#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generic, GPU/graph‑friendly L‑BFGS with Powell damping and Armijo+cubic line search.

Now supports **three precision modes** for optimizer internals:
- **"fp64"**: store & compute in float64 (most robust; best on CPU / data-center GPUs).
- **"fp32"**: store & compute in float32 (fastest arithmetic; stricter tolerances recommended).
- **"hybrid"**: store vectors (θ, g, d, S, Y) in float32, but compute all
  **reductions & scalars in float64** (sᵀy, sᵀs, yᵀy, ρ, γ, two-loop reductions, Powell, etc.).
  ρ is stored in float64. This is ideal for RTX GPUs: FP32 memory/bandwidth with FP64-quality curvature.

Loss/model can be its own precision (e.g., FP32); only optimizer internals are controlled here.

Python 3.7+, TensorFlow 2.x. Tested with TF 2.10–2.16.
"""
from __future__ import annotations
import os, time, random
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
import tensorflow as tf
from keras_compat import keras

TensorLike = Union[tf.Tensor, tf.Variable]
ParamStruct = Union[TensorLike, Sequence[TensorLike]]


def pick_vars_device(run_device: str) -> str:
    # run_device is 'gpu' or 'cpu' from your sweep
    if run_device.lower() == 'gpu' and tf.config.list_physical_devices('GPU'):
        return '/GPU:0'
    return '/CPU:0'


# ---------------------------------
# Precision / determinism utilities
# ---------------------------------
class Precision:
    FP32 = "fp32"
    FP64 = "fp64"
    MIXED = "mixed"


def set_model_precision(precision: str):
    if precision == Precision.MIXED:
        from keras import mixed_precision
        mixed_precision.set_global_policy("mixed_float16")
    elif precision == Precision.FP64:
        keras.backend.set_floatx("float64")
    else:
        keras.backend.set_floatx("float32")


def _mixed_policy_compute_dtype() -> Optional[str]:
    # Works with standalone keras and tf.keras, old/new APIs.
    try:
        from keras import mixed_precision as km
        pol = km.global_policy()
    except Exception:
        try:
            from keras import mixed_precision as km
            try:
                pol = km.global_policy()
            except Exception:
                pol = km.experimental.global_policy()
        except Exception:
            return None
    return getattr(pol, "compute_dtype", None)


def _mixed_fp16_enabled() -> bool:
    # Loss scaling is only needed for fp16, not usually for bfloat16.
    return _mixed_policy_compute_dtype() == "float16"


def set_global_determinism(seed: int, determinism=False):
    os.environ["PYTHONHASHSEED"] = str(seed)
    # set determinism to False for slightly better speed
    if determinism:
        os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            pass
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    try:
        keras.utils.set_random_seed(seed)
    except Exception:
        pass


# -----------------------------
# Numeric tolerance helper (dtype-aware + profiles)
# -----------------------------

def num_tol(dtype: tf.dtypes.DType, profile: str = "default"):
    """Return dtype-scaled tolerances.
    profile="default" for usual settings; profile="strict" (recommended for fp32) raises thresholds.
    Keys: denom_atol, sTy_fac, cos_tol, gamma_min, powell_kappa, adam_eps, cubic_det
    """
    eps = tf.constant(np.finfo(dtype.as_numpy_dtype).eps, dtype)
    if dtype == tf.float32:
        # eps = tf.constant(np.finfo(np.float32).eps, tf.float32)
        if profile == "strict":
            return {
                "denom_atol": tf.constant(1000.0, dtype) * eps,  # ~3e-4
                "sTy_fac": tf.constant(1000.0, dtype) * eps,  # ~3e-4
                "cos_tol": tf.maximum(1e-4, tf.constant(1000.0, dtype) * eps),
                "gamma_min": tf.sqrt(eps),  # ~3e-4
                "powell_kappa": tf.constant(10000.0, dtype) * eps,  # ~3e-3
                "adam_eps": tf.constant(1e-6, dtype),
                "cubic_det": 1e-4 * tf.ones([], dtype),
            }
        else:
            return {
                "denom_atol": tf.constant(100.0, dtype) * eps,  # ~1e-5
                "sTy_fac": tf.constant(100.0, dtype) * eps,  # ~1e-5
                "cos_tol": tf.constant(100.0, dtype) * eps,  # ~1e-5
                "gamma_min": tf.sqrt(eps),  # ~3e-4
                "powell_kappa": tf.constant(1000.0, dtype) * eps,  # ~1e-4
                "adam_eps": tf.constant(1e-8, dtype),
                "cubic_det": tf.pow(eps, 0.75),  # ~1e-5
            }
    else:
        # eps = tf.constant(np.finfo(np.float64).eps, tf.float64)
        # fp64 uses default; strict not needed
        return {
            "denom_atol": tf.constant(100.0, dtype) * eps,  # ~2e-14
            "sTy_fac": tf.constant(100.0, dtype) * eps,  # ~2e-14
            "cos_tol": tf.constant(100.0, dtype) * eps,  # ~2e-14
            "gamma_min": tf.sqrt(eps),  # ~1.5e-8
            "powell_kappa": tf.constant(1000.0, dtype) * eps,  # ~2e-13
            "adam_eps": tf.constant(1e-12, dtype),
            "cubic_det": tf.pow(eps, 0.75),  # ~1e-12
        }


# --------------------
# Flattening utilities
# --------------------
@dataclass(eq=False)
class Seg:
    start: int
    size: int
    shape: Tuple[int, ...]
    dtype: tf.dtypes.DType


@dataclass(eq=False)
class ParamSpec:
    """Precomputed mapping between a structured param list and a flat vector."""
    segs: List[Seg]
    total: int

    @staticmethod
    def from_params(params: Sequence[TensorLike]) -> "ParamSpec":
        segs: List[Seg] = []
        start = 0
        for p in params:
            size = int(np.prod(p.shape))
            segs.append(Seg(start=start, size=size, shape=tuple(int(s) for s in p.shape), dtype=p.dtype))
            start += size
        return ParamSpec(segs=segs, total=start)

    def flatten(self, params: Sequence[TensorLike], to_dtype: tf.dtypes.DType) -> tf.Tensor:
        flat = [tf.reshape(tf.cast(p, to_dtype), [-1]) for p in params]
        return tf.concat(flat, axis=0)

    def unflatten(self, theta: tf.Tensor, dtypes: Optional[Sequence[tf.dtypes.DType]] = None) -> List[tf.Tensor]:
        out: List[tf.Tensor] = []
        offset = 0
        for i, seg in enumerate(self.segs):
            tt = theta[offset: offset + seg.size]
            dt = seg.dtype if dtypes is None else dtypes[i]
            out.append(tf.reshape(tf.cast(tt, dt), seg.shape))
            offset += seg.size
        return out

    def assign_to_vars(self, theta: tf.Tensor, vars_list: Sequence[tf.Variable]):
        offset = 0
        for seg, v in zip(self.segs, vars_list):
            tt = theta[offset: offset + seg.size]
            v.assign(tf.reshape(tf.cast(tt, v.dtype), v.shape))
            offset += seg.size


# -----------------------------
# Diagnostics / warnings (graph)
# -----------------------------
@tf.function
def warn_if_non_finite(name: tf.Tensor, tensor: tf.Tensor):
    finite = tf.reduce_all(tf.math.is_finite(tensor))
    tf.cond(tf.logical_not(finite), lambda: tf.print("[WARN] non-finite in", name, "=>", tensor), lambda: tf.no_op())


@tf.function
def warn_if_small_denominator(name: tf.Tensor, denom: tf.Tensor, eps: tf.Tensor):
    bad = tf.reduce_any(tf.abs(denom) <= eps)
    tf.cond(bad, lambda: tf.print("[WARN] tiny denom in", name, ":", denom), lambda: tf.no_op())


# -----------------------------
# L‑BFGS memory (store vs compute dtypes)
# -----------------------------
@dataclass(eq=False)
class LBFGSMemory:
    m: int
    n: int
    dtype_store: tf.dtypes.DType
    dtype_rho: tf.dtypes.DType
    dtype_compute: tf.dtypes.DType
    suppress_warn: bool = False
    use_xla: bool = False  # <-- new
    debug_print: bool = False  # <-- new
    device: str = '/CPU:0'

    def __post_init__(self):
        ds = self.dtype_store
        with tf.device(self.device):
            self.S = tf.Variable(tf.zeros([self.m, self.n], dtype=ds), trainable=False, name='lbfgs_S')
            self.Y = tf.Variable(tf.zeros([self.m, self.n], dtype=ds), trainable=False, name='lbfgs_Y')
            self.rho = tf.Variable(tf.zeros([self.m], dtype=self.dtype_rho), trainable=False, name='lbfgs_rho')
            self.len = tf.Variable(tf.cast(0, self.dtype_compute),
                                   dtype=self.dtype_compute, trainable=False, name='lbfgs_len')

    @tf.function
    def clear(self):
        self.S.assign(tf.zeros_like(self.S))
        self.Y.assign(tf.zeros_like(self.Y))
        self.rho.assign(tf.zeros_like(self.rho))
        self.len.assign(tf.cast(0, self.len.dtype))

    @tf.function
    def push(self, s_comp: tf.Tensor, y_comp: tf.Tensor):
        """Push pair to memory.
        s_comp, y_comp are in compute dtype; we store S,Y in store dtype; ρ is computed in compute dtype.
        """
        dc = self.dtype_compute
        ds = self.dtype_store
        tol = num_tol(dc)["denom_atol"]
        n = tf.shape(self.S)[1]
        s_c = tf.reshape(tf.cast(s_comp, dc), [n])
        y_c = tf.reshape(tf.cast(y_comp, dc), [n])
        # compute rho in compute dtype for robustness
        ys = tf.tensordot(y_c, s_c, axes=1)

        # Only build print logic when debugging AND not compiling with XLA
        denom_atol = num_tol(dc)["denom_atol"]

        if (not self.use_xla) and self.debug_print and (not self.suppress_warn):
            bad = tf.less(tf.abs(ys), denom_atol)

            def _do_print():
                return tf.print("[WARN] tiny denom in y^T s (rho):", ys)

            tf.cond(bad, _do_print, lambda: tf.no_op())
        rho_new = tf.cast(1.0, dc) / (ys + denom_atol)
        # store S,Y in store dtype (roll window)
        s_s = tf.reshape(tf.cast(s_c, ds), [n])
        y_s = tf.reshape(tf.cast(y_c, ds), [n])
        self.S.assign(tf.concat([self.S[1:], tf.expand_dims(s_s, 0)], axis=0))
        self.Y.assign(tf.concat([self.Y[1:], tf.expand_dims(y_s, 0)], axis=0))
        # store rho in rho dtype (usually compute dtype)
        self.rho.assign(tf.concat([self.rho[1:], tf.reshape(tf.cast(rho_new, self.dtype_rho), [1])], axis=0))
        new_len = tf.minimum(
            tf.add(self.len, tf.cast(1, self.len.dtype)),
            tf.cast(tf.shape(self.S)[0], self.len.dtype)
        )
        self.len.assign(new_len)


# -----------------------------
# Powell damping (compute dtype)
# -----------------------------
@tf.function
def powell_damp_pair(s_c: tf.Tensor, y_c: tf.Tensor, c: float, gamma_c: tf.Tensor) -> Tuple[
    tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    d = s_c.dtype
    tol = num_tol(d)
    dmax = lambda x: tf.maximum(x, tol["denom_atol"])  # denom guard
    zero = tf.constant(0.0, d)
    one = tf.constant(1.0, d)
    c = tf.cast(c, d)
    ss = tf.tensordot(s_c, s_c, 1)
    yy = tf.tensordot(y_c, y_c, 1)
    sTy = tf.tensordot(s_c, y_c, 1)
    # flip y if negative curvature
    y_c = tf.where(sTy < zero, -y_c, y_c)
    sTy = tf.where(sTy < zero, -sTy, sTy)
    # legacy tiny floor (kept): fl = ss * yy * 1e-9 (in compute dtype)
    fl = ss * yy * (tf.cast(1e-8, d) if s_c.dtype == tf.float32 else tf.cast(1e-9, d))
    tiny = sTy < fl
    a = tf.where(yy > zero, fl / dmax(yy), one)
    y_c = tf.where(tiny, a * y_c, y_c)
    sTy = tf.where(tiny, fl, sTy)
    # B0 ≈ γ^{-1} I
    inv_gamma = one / dmax(tf.cast(gamma_c, d))
    sBs = inv_gamma * ss
    cond_nodamp = sTy >= c * sBs
    theta = tf.where(cond_nodamp, one, (one - c) * sBs / dmax(sBs - sTy))
    y_bar = theta * y_c + (one - theta) * inv_gamma * s_c
    sTy_bar = tf.tensordot(s_c, y_bar, 1)
    return y_bar, theta, sTy, sTy_bar


# -----------------------------
# Two‑loop recursion (reductions in compute dtype)
# -----------------------------
@tf.function
def two_loop_direction(g_store: tf.Tensor, mem: LBFGSMemory, gamma_c: tf.Tensor,
                       d0_diag_c: Optional[tf.Tensor] = None) -> tf.Tensor:
    ds = mem.dtype_store
    dc = mem.dtype_compute
    # operate q in compute dtype for robust reductions
    q = tf.cast(g_store, dc)
    m = mem.m
    L = tf.cast(mem.len, tf.int32)
    alpha = tf.TensorArray(dc, size=m, clear_after_read=False)
    # backward
    for i in tf.range(m - 1, -1, delta=-1):
        valid = i >= (m - L)
        s_i = tf.cast(mem.S[i], dc)
        y_i = tf.cast(mem.Y[i], dc)
        rho_i = tf.cast(mem.rho[i], dc)
        a_i = rho_i * tf.tensordot(s_i, q, 1)
        a_i = tf.where(valid, a_i, tf.cast(0.0, dc))
        q = tf.where(valid, q - a_i * y_i, q)
        alpha = alpha.write(i, a_i)
    # H0
    if d0_diag_c is None:
        q = tf.cast(gamma_c, dc) * q
    else:
        q = tf.cast(d0_diag_c, dc) * q
    # forward
    for i in tf.range(0, m, delta=1):
        valid = i >= (m - L)
        s_i = tf.cast(mem.S[i], dc)
        y_i = tf.cast(mem.Y[i], dc)
        rho_i = tf.cast(mem.rho[i], dc)
        a_i = alpha.read(i)
        b_i = rho_i * tf.tensordot(y_i, q, 1)
        q = tf.where(valid, q + s_i * (a_i - b_i), q)
    # return direction in store dtype
    return -tf.cast(q, ds)


# -----------------------------
# Line search: Armijo + cubic (runs in store dtype)
# -----------------------------
@tf.function
def armijo_ok(f0: tf.Tensor, m0: tf.Tensor, f_a: tf.Tensor, a: tf.Tensor, c1: tf.Tensor) -> tf.Tensor:
    return f_a <= f0 + c1 * a * m0


@tf.function
def cubic_step_closed_form(f0: tf.Tensor, m0: tf.Tensor,
                           a_prev: tf.Tensor, f_prev: tf.Tensor,
                           a: tf.Tensor, f_a: tf.Tensor,
                           low_mult: tf.Tensor, high_mult: tf.Tensor,
                           denom_atol: tf.Tensor, cubic_det: tf.Tensor) -> tf.Tensor:
    d = f0.dtype
    low_mult = tf.cast(low_mult, d)
    high_mult = tf.cast(high_mult, d)
    denom_atol = tf.cast(denom_atol, d)
    cubic_det = tf.cast(cubic_det, d)
    # a_prev = tf.cast(a_prev, d)
    # f_prev = tf.cast(f_prev, d)
    m0 = tf.cast(m0, d)

    def default():
        return tf.clip_by_value(0.5 * a, low_mult * a, high_mult * a)

    cond_have_prev = tf.logical_and(tf.math.is_finite(a_prev), tf.math.is_finite(f_prev))

    def compute():
        r1 = f_prev - f0 - m0 * a_prev
        r2 = f_a - f0 - m0 * a
        M = tf.reshape(tf.stack([a_prev * a_prev, a_prev * a_prev * a_prev,
                                 a * a, a * a * a], axis=0), [2, 2])
        rhs = tf.stack([r1, r2])
        det = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
        small = tf.abs(det) < cubic_det

        def solve_then_min():
            cd = tf.linalg.solve(M, tf.reshape(rhs, [2, 1]))
            c = cd[0, 0]
            d3 = cd[1, 0]
            A = 3.0 * d3
            B = 2.0 * c
            C = m0
            disc = B * B - 4.0 * A * C
            bad = tf.logical_or(~tf.math.is_finite(disc), disc <= 0.0)

            def fallback():
                return default()

            def roots():
                sqrt_disc = tf.sqrt(disc + tf.cast(0.0, d))
                safe = lambda x: x + denom_atol
                t1 = (-B + sqrt_disc) / safe(2.0 * A)
                t2 = (-B - sqrt_disc) / safe(2.0 * A)
                t_candidates = tf.stack([t1, t2])
                mask = tf.logical_and(tf.math.is_finite(t_candidates), t_candidates > 0.0)
                t_candidates = tf.where(mask, t_candidates, tf.fill(tf.shape(t_candidates), default()))
                half = 0.5 * a
                idx = tf.argmin(tf.abs(t_candidates - half))
                t_new = tf.gather(t_candidates, idx)
                t_new = tf.clip_by_value(t_new, low_mult * a, high_mult * a)
                t_new = tf.where(tf.logical_or(~tf.math.is_finite(t_new), t_new <= 0.0), default(), t_new)
                return t_new

            return tf.cond(bad, fallback, roots)

        return tf.cond(small, default, solve_then_min)

    return tf.cond(cond_have_prev, compute, default)


# -----------------------------
# Config
# -----------------------------
@dataclass
class LBFGSConfig:
    max_iters: int = 200
    mem: int = 10
    c1: float = 1e-4
    powell_c: float = 0.2
    ls_max_steps: int = 8
    alpha_init: float = 1.0
    cub_clip: Tuple[float, float] = (0.1, 2.5)
    curvature_cos_tol: float = 1e-6
    print_every: int = 100  # outer-loop logging interval (iterations)
    # Adam-style H0
    use_adam_h0: bool = False
    beta2: float = 0.999
    diag_eps: float = 0.0  # 0 => use dtype default from num_tol
    # Inner compiled chunk size
    chunk_size: int = 100
    # Precision mode for optimizer internals: 'fp64' | 'fp32' | 'hybrid'
    lbfgs_mode: str = 'fp64'
    # If lbfgs_mode='fp32', use stricter tolerances
    fp32_strict: bool = True
    debug_print: bool = False
    run_device: str = 'gpu'


# -----------------------------
# Loss/param adapter
# -----------------------------
class LossAdapter:
    """
    Bridges between (theta_flat) <-> your loss function.
    Modes:
      - mode='struct': call loss_fn(params_struct); grads optional.
      - mode='assign': assign theta into provided variables, then call loss_fn() with no args; grads optional.
    """

    def __init__(self,
                 loss_fn: Callable,
                 params: Union[Sequence[TensorLike], tf.Tensor],
                 mode: str = 'struct',
                 opt_dtype: tf.dtypes.DType = tf.float64,
                 use_xla: bool = False):
        self.loss_fn = loss_fn
        self.mode = mode
        self.opt_dtype = opt_dtype  # EXPECTS store dtype of theta
        self.use_xla = use_xla
        # Decide once: do we need loss scaling? (only when mixed policy uses fp16 compute)
        self._do_loss_scaling = _mixed_fp16_enabled()
        # Constant captured by tf.function; matches optimizer store dtype
        self._loss_scale = tf.constant(2.0 ** 15, dtype=self.opt_dtype) if self._do_loss_scaling \
            else tf.constant(1.0, dtype=self.opt_dtype)

        # Normalize params
        if isinstance(params, (list, tuple)):
            self.params_list: Optional[List[TensorLike]] = list(params)
            self.spec = ParamSpec.from_params(self.params_list)
            theta0 = self.spec.flatten(self.params_list, to_dtype=opt_dtype)
        else:
            self.params_list = None
            self.spec = ParamSpec([Seg(0, int(params.shape[0]), (int(params.shape[0]),), params.dtype)],
                                  int(params.shape[0]))
            theta0 = tf.cast(tf.reshape(params, [-1]), opt_dtype)

        self.theta0 = theta0

        # Build wrappers
        if mode == 'struct':
            @tf.function(jit_compile=use_xla)
            def loss_only(theta: tf.Tensor) -> tf.Tensor:
                plist = self.spec.unflatten(theta)
                out = self.loss_fn(plist)
                if isinstance(out, (list, tuple)):
                    loss = out[0]
                else:
                    loss = out
                return tf.cast(loss, self.opt_dtype)

            @tf.function(jit_compile=use_xla)
            def loss_and_grad(theta: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
                plist = self.spec.unflatten(theta)

                # If user returns (loss, grads_struct) we bypass Tape and scaling
                out = self.loss_fn(plist)
                if isinstance(out, (list, tuple)) and len(out) == 2:
                    loss, grads_struct = out
                    gflat = self.spec.flatten(grads_struct, to_dtype=self.opt_dtype)
                    return tf.cast(loss, self.opt_dtype), tf.cast(gflat, self.opt_dtype)

                with tf.GradientTape() as tape:
                    for p in plist:
                        tape.watch(p)
                    # Compute loss in float32 for stability, then cast to opt dtype for the optimizer math
                    loss_val = tf.cast(self.loss_fn(plist), tf.float32)
                    loss_cast = tf.cast(loss_val, self.opt_dtype)

                    # Manual loss scaling (only if mixed+fp16 is active)
                    loss_for_grad = loss_cast * tf.cast(self._loss_scale, loss_cast.dtype)

                grads = tape.gradient(loss_for_grad, plist)

                # Unscale grads if we scaled the loss, also replace Nones with zeros (rare but safe)
                if self._do_loss_scaling:
                    grads = [tf.zeros_like(p) if g is None else g / self._loss_scale for g, p in zip(grads, plist)]
                else:
                    grads = [tf.zeros_like(p) if g is None else g for g, p in zip(grads, plist)]

                gflat = self.spec.flatten(grads, to_dtype=self.opt_dtype)
                return tf.cast(loss_cast, self.opt_dtype), tf.cast(gflat, self.opt_dtype)


        elif mode == 'assign':
            assert self.params_list is not None, "assign mode requires a params list of Variables"

            # Accept anything variable-like (has .assign). This is robust across TF versions.

            def _require_assignable(x):
                if hasattr(x, "assign"):
                    return x  # tf.Variable / ResourceVariable / Keras variable
                raise TypeError(f"mode='assign' requires variables with .assign; got {type(x).__name__}")

            vars_list = [_require_assignable(p) for p in self.params_list]

            @tf.function(jit_compile=use_xla)
            def loss_only(theta: tf.Tensor) -> tf.Tensor:
                self.spec.assign_to_vars(theta, vars_list)
                out = self.loss_fn()
                if isinstance(out, (list, tuple)):
                    loss = out[0]
                else:
                    loss = out
                return tf.cast(loss, self.opt_dtype)

            @tf.function(jit_compile=use_xla)
            def loss_and_grad(theta: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
                self.spec.assign_to_vars(theta, vars_list)

                # If user returns (loss, grads_struct) we bypass Tape and scaling
                out = self.loss_fn()
                if isinstance(out, (list, tuple)) and len(out) == 2:
                    loss, grads_struct = out
                    gflat = self.spec.flatten(grads_struct, to_dtype=self.opt_dtype)
                    return tf.cast(loss, self.opt_dtype), tf.cast(gflat, self.opt_dtype)

                with tf.GradientTape() as tape:
                    # Variables are auto-watched, no need for tape.watch(vars_list)
                    loss_val = tf.cast(self.loss_fn(), tf.float32)  # stable reduction dtype
                    loss_cast = tf.cast(loss_val, self.opt_dtype)  # match optimizer math
                    loss_for_grad = loss_cast * self._loss_scale  # scale if mixed+fp16

                grads = tape.gradient(loss_for_grad, vars_list)

                if self._do_loss_scaling:
                    grads = [tf.zeros_like(v) if g is None else g / tf.cast(self._loss_scale, g.dtype)
                             for g, v in zip(grads, vars_list)]
                else:
                    grads = [tf.zeros_like(v) if g is None else g for g, v in zip(grads, vars_list)]

                gflat = self.spec.flatten(grads, to_dtype=self.opt_dtype)
                return tf.cast(loss_cast, self.opt_dtype), tf.cast(gflat, self.opt_dtype)

        else:
            raise ValueError("mode must be 'struct' or 'assign'")

        self.loss_only = loss_only
        self.loss_and_grad = loss_and_grad


# -----------------------------
# Main L‑BFGS optimizer (inner/outer)
# -----------------------------
class LBFGS:
    def __init__(self,
                 loss_adapter: LossAdapter,
                 cfg: LBFGSConfig,
                 lbfgs_mode: Optional[str] = None,
                 use_xla: bool = False):
        self.cfg = cfg
        self.debug_print = cfg.debug_print
        # If debugging, do NOT use XLA (avoids tf.print/string in compiled clusters)
        self.use_xla = bool(use_xla and not self.debug_print)
        mode = (lbfgs_mode or cfg.lbfgs_mode).lower()
        if mode not in ("fp64", "fp32", "hybrid"):
            raise ValueError("lbfgs_mode must be 'fp64', 'fp32', or 'hybrid'")
        self.mode = mode
        # store/compute dtypes
        if mode == "fp64":
            self.store_dtype = tf.float64
            self.compute_dtype = tf.float64
            self.tol_profile = "default"
        elif mode == "fp32":
            self.store_dtype = tf.float32
            self.compute_dtype = tf.float32
            self.tol_profile = "strict" if cfg.fp32_strict else "default"
        else:  # hybrid
            self.store_dtype = tf.float32
            self.compute_dtype = tf.float64
            self.tol_profile = "default"

        self.adapter = loss_adapter  # expects store dtype
        self.use_xla = use_xla

        self.var_device = pick_vars_device(cfg.run_device)
        with tf.device(self.var_device):
            # Parameters/state
            self.theta = tf.Variable(tf.cast(loss_adapter.theta0, self.store_dtype), trainable=False)
            self.n = int(self.theta.shape[0])
            self.mem = LBFGSMemory(m=cfg.mem, n=self.n,
                                   dtype_store=self.store_dtype,
                                   dtype_rho=self.compute_dtype,
                                   dtype_compute=self.compute_dtype,
                                   suppress_warn=not self.debug_print,
                                   use_xla=self.use_xla,
                                   debug_print=self.debug_print,
                                   device=self.var_device)
            # Adam diag state (kept in compute dtype for stability)
            self.adam_v = tf.Variable(tf.zeros([self.n], dtype=self.compute_dtype), trainable=False)
            self.adam_t = tf.Variable(0, dtype=tf.int64, trainable=False)

        print('mem devices:', self.mem.S.device, self.mem.Y.device, self.mem.rho.device, self.mem.len.device)

        @tf.function(jit_compile=use_xla)
        def _adam_diag(gflat_store: tf.Tensor, beta2: tf.Tensor, eps_c: tf.Tensor) -> tf.Tensor:
            g_c = tf.cast(gflat_store, self.compute_dtype)
            self.adam_t.assign_add(1)
            self.adam_v.assign(beta2 * self.adam_v + (1.0 - beta2) * tf.square(g_c))
            t = tf.cast(self.adam_t, self.compute_dtype)
            denom = 1.0 - tf.pow(beta2, t) + num_tol(self.compute_dtype)["denom_atol"]
            v_hat = self.adam_v / denom
            return 1.0 / tf.sqrt(v_hat + eps_c)  # compute dtype

        self._adam_diag = _adam_diag

        # Initial loss & grad (store dtype)
        f0, g0 = self.adapter.loss_and_grad(self.theta)
        self.f = tf.Variable(tf.cast(f0, self.store_dtype), trainable=False)
        self.g = tf.Variable(tf.cast(g0, self.store_dtype), trainable=False)
        self.gamma = tf.Variable(tf.cast(1.0, self.compute_dtype), trainable=False)
        self.mem.clear()

        # Build inner compiled K-step runner
        self._build_inner_runner()

    def _build_inner_runner(self):
        cfg = self.cfg
        ds = self.store_dtype
        dc = self.compute_dtype
        adapter = self.adapter
        mem = self.mem
        use_xla = self.use_xla
        # Precision/tolerance mapping per requested logic:
        # 1) fp64: store=compute=fp64, fp64 tolerances (default)
        # 2) fp32: store=compute=fp32, **strict** tolerances
        # 3) hybrid: store=fp32 (default tol), compute=fp64 (default tol)
        if self.mode == "fp64":
            tol_store = num_tol(ds, "default")  # ds==fp64
            tol_comp = num_tol(dc, "default")  # dc==fp64
        elif self.mode == "fp32":
            tol_store = num_tol(ds, "strict")  # ds==fp32
            tol_comp = num_tol(dc, "strict")  # dc==fp32
        else:  # hybrid
            tol_store = num_tol(ds, "default")  # ds==fp32
            tol_comp = num_tol(dc, "default")  # dc==fp64

        @tf.function(jit_compile=use_xla)
        def loss_only(theta):
            return adapter.loss_only(theta)

        @tf.function(jit_compile=use_xla)
        def loss_and_grad(theta):
            return adapter.loss_and_grad(theta)

        @tf.function(jit_compile=use_xla)
        def line_search(theta, f0, g, d) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
            alpha0 = tf.cast(cfg.alpha_init, ds)
            c1 = tf.cast(cfg.c1, ds)
            low_mult = tf.cast(cfg.cub_clip[0], ds)
            high_mult = tf.cast(cfg.cub_clip[1], ds)
            ls_max = tf.cast(cfg.ls_max_steps, tf.int32)
            m0 = tf.tensordot(g, d, 1)  # store dtype

            def fix_descent():
                return -g, -tf.tensordot(g, g, 1)

            d, m0 = tf.cond(m0 >= 0.0, fix_descent, lambda: (d, m0))

            def cond(i, alpha, a_prev, f_prev, best_f, best_a, accepted):
                return tf.logical_and(i < ls_max, tf.logical_not(accepted))

            def body(i, alpha, a_prev, f_prev, best_f, best_a, accepted):
                f_a = loss_only(theta + alpha * d)
                ok = armijo_ok(f0, m0, f_a, alpha, c1)
                better = f_a < best_f
                best_f = tf.where(better, f_a, best_f)
                best_a = tf.where(better, alpha, best_a)
                accepted = tf.logical_or(accepted, ok)
                alpha_next = tf.cond(ok,
                                     lambda: alpha,
                                     lambda: cubic_step_closed_form(f0, m0, a_prev, f_prev, alpha, f_a, low_mult,
                                                                    high_mult, tol_comp["denom_atol"],
                                                                    tol_comp["cubic_det"]))
                a_prev_next = tf.cond(ok, lambda: a_prev, lambda: alpha_next)
                f_prev_next = tf.cond(ok, lambda: f_prev, lambda: tf.where(tf.math.is_finite(f_a), f_a, f_prev))
                return i + 1, alpha_next, a_prev_next, f_prev_next, best_f, best_a, accepted

            i0 = tf.constant(0, tf.int32)
            a_prev0 = tf.cast(tf.constant(float('nan')), ds)
            f_prev0 = tf.cast(tf.constant(float('nan')), ds)
            best_f0 = tf.identity(f0)
            best_a0 = tf.cast(0.0, ds)
            accepted0 = tf.constant(False)

            i, alpha, a_prev, f_prev, best_f, best_a, accepted = tf.while_loop(
                cond, body, (i0, alpha0, a_prev0, f_prev0, best_f0, best_a0, accepted0),
                maximum_iterations=ls_max)

            alpha_final = tf.where(best_f < f0, best_a, alpha)
            return alpha_final, tf.cast(accepted, tf.int32), i, m0, d, best_f

        @tf.function(jit_compile=use_xla)
        def inner_k_steps(theta, f, g, gamma, k: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor,
        tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
            # returns: theta,f,g,gamma, alpha_last, ls_iters_last, mem_len, sTy_bar_last, cos_sy_last
            it = tf.constant(0, tf.int32)

            def cond(it, theta, f, g, gamma, alpha_last, ls_last, mem_len, sTy_bar_last, cos_sy_last):
                return it < k

            def body(it, theta, f, g, gamma, alpha_last, ls_last, mem_len, sTy_bar_last, cos_sy_last):
                # H0 diag (optional) in compute dtype
                d0_diag_c = None
                if cfg.use_adam_h0:
                    cfg_de = tf.cast(cfg.diag_eps, dc)
                    eps_adam = tf.cond(cfg_de > 0, lambda: cfg_de, lambda: tol_comp["adam_eps"])
                    d0_diag_c = self._adam_diag(g, tf.cast(cfg.beta2, dc), eps_adam)

                d = two_loop_direction(g, mem, gamma, d0_diag_c)  # returns store dtype
                alpha, accepted, ls_iters, m0, d_used, best_f = line_search(theta, f, g, d)
                theta_next = theta + tf.cast(alpha, ds) * d
                f_next, g_next = loss_and_grad(theta_next)

                # ds = self.store_dtype inside inner_k_steps
                rel = tf.constant(1e-6, ds) if ds == tf.float64 else tf.constant(5e-4, ds)
                abs_ = tf.constant(1e-9, ds) if ds == tf.float64 else tf.constant(1e-6, ds)

                # tolerance scales with magnitude; protects near-zero losses too
                tol = tf.maximum(abs_, rel * tf.maximum(tf.abs(f), tf.abs(f_next)))

                # accept if Armijo accepted OR if the change is within tolerance
                accept_bool = tf.logical_or(accepted>0, f_next <= f + tol)

                def accept():
                    return theta_next, f_next, g_next

                def reject():
                    return theta, f, g

                theta_next, f_next, g_next = tf.cond(accept_bool, accept, reject)

                def do_pair_update():
                    # s,y in compute dtype (hybrid safe)
                    s_c = tf.cast(theta_next - theta, dc)
                    y_c = tf.cast(g_next - g, dc)
                    y_bar_c, theta_mix, sTy_raw, sTy_bar = powell_damp_pair(s_c, y_c, cfg.powell_c, gamma)

                    # angle and curvature checks (compute dtype)
                    ss = tf.tensordot(s_c, s_c, 1)
                    yybar = tf.tensordot(y_bar_c, y_bar_c, 1)
                    denom = tf.sqrt(ss * (yybar + tol_comp["denom_atol"]) + tol_comp["denom_atol"])
                    cos_sy = sTy_bar / denom
                    curv_thr = tol_comp["sTy_fac"] * tf.sqrt(ss * (yybar + tol_comp["denom_atol"]))
                    cos_thr = tf.maximum(tf.cast(cfg.curvature_cos_tol, dc), tol_comp["cos_tol"])
                    pair_ok = tf.logical_and(sTy_bar > curv_thr, cos_sy >= cos_thr)
                    tiny_gate = tf.cast(10.0, dc) * tol_comp["denom_atol"]
                    pair_ok = tf.logical_and(pair_ok, sTy_bar > tiny_gate)

                    def push_pair():
                        mem.push(s_c, y_bar_c)
                        return tf.maximum(sTy_bar / (yybar + tol_comp["denom_atol"]), tol_comp["gamma_min"])

                    gamma_next_local = tf.cond(pair_ok, push_pair, lambda: gamma)
                    return gamma_next_local, sTy_bar, cos_sy

                def skip_pair_update():
                    # No state changes; return well-shaped placeholders for logging
                    return gamma, tf.cast(0.0, dc), tf.cast(0.0, dc)

                gamma_next, sTy_bar, cos_sy = tf.cond(accept_bool, do_pair_update, skip_pair_update)

                # advance (keep store dtype for theta/f/g)
                return (it + 1, theta_next, f_next, g_next, gamma_next,
                        alpha, ls_iters, tf.cast(mem.len, tf.int32), sTy_bar, cos_sy)

            it, theta, f, g, gamma, alpha_last, ls_last, mem_len, sTy_bar_last, cos_sy_last = tf.while_loop(
                cond, body,
                loop_vars=(it, theta, f, g, gamma,
                           tf.cast(0.0, ds), tf.cast(0, tf.int32), tf.cast(0, tf.int32), tf.cast(0.0, dc),
                           tf.cast(0.0, dc)),
                maximum_iterations=k)
            return theta, f, g, gamma, alpha_last, ls_last, mem_len, sTy_bar_last, cos_sy_last

        self._loss_only = loss_only
        self._loss_and_grad = loss_and_grad
        self._line_search = line_search
        self._inner_k_steps = inner_k_steps

    # ---------- public API ----------
    def minimize_chunked(self, total_iters: int, chunk_size: Optional[int] = None,
                         print_every: Optional[int] = None) -> None:
        if chunk_size is None:
            chunk_size = self.cfg.chunk_size
        if print_every is None:
            print_every = self.cfg.print_every

        t0 = time.perf_counter()
        # Iter 0 already computed in __init__
        print(
            f"Iter 0 : loss= {float(self.f.numpy())} , g_norm= {float(tf.linalg.norm(self.g).numpy())} , mem= {int(self.mem.len.numpy())} , t= 0.000s")

        total_done = 0
        while total_done < total_iters:
            k = min(chunk_size, total_iters - total_done)
            theta, f, g, gamma, alpha_last, ls_last, mem_len, sTy_bar_last, cos_sy_last = self._inner_k_steps(
                self.theta, self.f, self.g, self.gamma, tf.cast(k, tf.int32))
            # update captured state vars
            self.theta.assign(theta);
            self.f.assign(f);
            self.g.assign(g);
            self.gamma.assign(gamma)
            total_done += k
            # Logging
            if (total_done % print_every) == 0 or total_done == total_iters:
                t_now = time.perf_counter() - t0
                print(
                    f"Iter {total_done:>4} : loss= {float(f.numpy())} , alpha= {float(alpha_last.numpy()):g} , "
                    f"ls= {int(ls_last.numpy())} , mem= {int(mem_len.numpy())} , cos_sy= {float(cos_sy_last.numpy())} , "
                    f"sTy_bar= {float(sTy_bar_last.numpy())} , t= {t_now:.3f}s")

    # Convenience accessors
    def get_theta(self) -> tf.Tensor:
        return tf.identity(self.theta)


# -----------------------------
# Demo with a small MLP + MSE
# -----------------------------

def std_normal_cdf(z: tf.Tensor) -> tf.Tensor:
    z = tf.convert_to_tensor(z)
    return 0.5 * (1.0 + tf.math.erf(z / tf.sqrt(tf.cast(2.0, z.dtype))))


def make_data(n: int, a: float = 0.0, b: float = 1.0, seed: int = 0,
              dtype: tf.dtypes.DType = tf.float32) -> Tuple[tf.Tensor, tf.Tensor]:
    rng = np.random.default_rng(seed)
    t = rng.uniform(0.0, 1.0, size=(n, 1)).astype(np.float32)
    x = rng.uniform(-4.0, 4.0, size=(n, 1)).astype(np.float32)
    s = np.sqrt(1.0 - t).astype(np.float32)
    X = np.concatenate([t, x, s], axis=1).astype(np.float32)
    z = (x - a) / (b * s)
    z_tf = tf.convert_to_tensor(z, dtype=dtype)
    y = tf.cast(std_normal_cdf(z_tf), dtype)
    return tf.convert_to_tensor(X, dtype=dtype), y


def build_mlp(seed: int = 0, output_dtype: str = 'float32') -> tf.keras.Model:
    k = tf.keras.initializers.GlorotUniform
    z = tf.keras.initializers.Zeros()
    inp = tf.keras.Input(shape=(3,))
    s = 64
    h = tf.keras.layers.Dense(s, activation='tanh', kernel_initializer=k(seed + 1), bias_initializer=z)(inp)
    h = tf.keras.layers.Dense(s, activation='tanh', kernel_initializer=k(seed + 2), bias_initializer=z)(h)
    h = tf.keras.layers.Dense(s, activation='tanh', kernel_initializer=k(seed + 3), bias_initializer=z)(h)
    out = tf.keras.layers.Dense(1, activation='sigmoid', kernel_initializer=k(seed + 4), bias_initializer=z,
                                dtype=output_dtype)(h)
    return tf.keras.Model(inputs=inp, outputs=out)


def demo_main():
    # Repro & GPU niceties
    seed = 1
    set_global_determinism(seed)
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass

    # Data & model
    N = 32768
    X, Y = make_data(N, seed=seed, dtype=tf.float32)
    set_model_precision(Precision.FP32)  # model in FP32
    model = build_mlp(seed=seed)

    # Loss function (assign mode): assign theta -> model vars, compute MSE
    vars_list = model.trainable_variables

    def loss_fn():
        pred = tf.cast(model(X, training=False), Y.dtype)
        e = tf.cast(Y, pred.dtype) - pred
        return tf.reduce_mean(tf.square(e))  # MSE

    # === Choose optimizer mode here ===
    # mode = 'fp64'  # store=compute=float64
    mode = 'fp32'  # store=compute=float32 (uses strict tolerances by default)
    # mode = 'hybrid'  # store=float32, compute=float64 (recommended for RTX)

    store_dtype = tf.float64 if mode == 'fp64' else tf.float32
    adapter = LossAdapter(loss_fn=loss_fn, params=vars_list, mode='assign', opt_dtype=store_dtype, use_xla=False)

    cfg = LBFGSConfig(max_iters=260, mem=10, print_every=30, chunk_size=30,
                      use_adam_h0=True, beta2=0.999, diag_eps=0.0,  # 0 -> dtype default
                      alpha_init=1.0, ls_max_steps=8, c1=1e-4, cub_clip=(0.1, 2.5), powell_c=0.2,
                      lbfgs_mode=mode, fp32_strict=True)

    opt = LBFGS(adapter, cfg, lbfgs_mode=mode, use_xla=False)

    # Outer loop (prints every 100 iters). For debugging: set chunk_size=1
    opt.minimize_chunked(total_iters=cfg.max_iters, chunk_size=cfg.chunk_size, print_every=cfg.print_every)

    # if __name__ == "__main__":
    demo_main()
