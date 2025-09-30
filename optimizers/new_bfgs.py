#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generic, GPU/graph‑friendly L‑BFGS with Powell damping and Armijo+cu​bic line search.

Key features
- Accepts **any loss function** and **parameters** (flat tensor or list/tuple of tensors/variables).
- Loss API can return `loss` **or** `(loss, grads)`; otherwise grads computed via `tf.GradientTape`.
- Optimizer works on a **flat parameter vector**; unflatten/assign happen **only when needed** (loss evals).
- Inner compiled function runs **K iterations entirely on GPU** (`@tf.function` + `tf.while_loop`).
- Outer Python loop calls the inner in **chunks** (e.g., every 100 or 500 iters) to print/log.
- Independent precision for model/loss vs optimizer: LBFGS math in fp32/fp64; loss can use its own dtype.
- Armijo‑only line search with **cubic interpolation** (no gradient evals inside LS).
- Powell damping exactly as requested.
- Adam‑style diagonal H₀ (optional).
- Determinism hooks + seeded demo.

Python 3.7+, TensorFlow 2.x. Tested with TF 2.10–2.16.
"""
from __future__ import annotations
import os, time, math, random
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import tensorflow as tf

TensorLike = Union[tf.Tensor, tf.Variable]
ParamStruct = Union[TensorLike, Sequence[TensorLike]]

# ---------------------------------
# Precision / determinism utilities
# ---------------------------------
class Precision:
    FP32 = "fp32"
    FP64 = "fp64"
    MIXED = "mixed"


def set_model_precision(precision: str):
    if precision == Precision.MIXED:
        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy("mixed_float16")
    elif precision == Precision.FP64:
        tf.keras.backend.set_floatx("float64")
    else:
        tf.keras.backend.set_floatx("float32")


def set_global_determinism(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    try:
        tf.keras.utils.set_random_seed(seed)
    except Exception:
        pass


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
# L‑BFGS memory (float64 state)
# -----------------------------
@dataclass(eq=False)
class LBFGSMemory:
    m: int
    n: int

    def __post_init__(self):
        self.S = tf.Variable(tf.zeros([self.m, self.n], dtype=tf.float64), trainable=False)
        self.Y = tf.Variable(tf.zeros([self.m, self.n], dtype=tf.float64), trainable=False)
        self.rho = tf.Variable(tf.zeros([self.m], dtype=tf.float64), trainable=False)
        self.len = tf.Variable(0, dtype=tf.int32, trainable=False)

    @tf.function
    def clear(self):
        self.S.assign(tf.zeros_like(self.S))
        self.Y.assign(tf.zeros_like(self.Y))
        self.rho.assign(tf.zeros_like(self.rho))
        self.len.assign(0)

    @tf.function
    def push(self, s: tf.Tensor, y: tf.Tensor):
        # s,y are [n] float64
        n = tf.shape(self.S)[1]
        s = tf.reshape(tf.cast(s, tf.float64), [n])
        y = tf.reshape(tf.cast(y, tf.float64), [n])
        # roll
        self.S.assign(tf.concat([self.S[1:], tf.expand_dims(s, 0)], axis=0))
        self.Y.assign(tf.concat([self.Y[1:], tf.expand_dims(y, 0)], axis=0))
        ys = tf.tensordot(y, s, axes=1)
        warn_if_small_denominator(tf.constant("y^T s (rho)"), ys, tf.constant(1e-18, tf.float64))
        rho_new = 1.0 / (ys + 1e-18)
        self.rho.assign(tf.concat([self.rho[1:], tf.reshape(rho_new, [1])], axis=0))
        self.len.assign(tf.minimum(self.len + 1, tf.shape(self.S)[0]))


# -----------------------------
# Powell damping (float64)
# -----------------------------
@tf.function
def powell_damp_pair(s: tf.Tensor, y: tf.Tensor, c: float, gamma: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    s = tf.cast(s, tf.float64); y = tf.cast(y, tf.float64); gamma = tf.cast(gamma, tf.float64); c = tf.cast(c, tf.float64)
    zero = tf.constant(0.0, tf.float64); one = tf.constant(1.0, tf.float64)
    eps = tf.constant(1e-16, tf.float64); tiny_fac = tf.constant(1e-9, tf.float64)
    ss = tf.tensordot(s, s, 1); yy = tf.tensordot(y, y, 1); sTy = tf.tensordot(s, y, 1)
    y = tf.where(sTy < zero, -y, y); sTy = tf.where(sTy < zero, -sTy, sTy)
    fl = ss * yy * tiny_fac; tiny = sTy < fl
    a = tf.where(yy > zero, fl / (yy + eps), one)
    y = tf.where(tiny, a * y, y); sTy = tf.where(tiny, fl, sTy)
    inv_gamma = one / (gamma + eps); sBs = inv_gamma * ss
    cond_nodamp = sTy >= c * sBs
    theta = tf.where(cond_nodamp, one, (one - c) * sBs / tf.maximum(sBs - sTy, eps))
    y_bar = theta * y + (one - theta) * inv_gamma * s
    sTy_bar = tf.tensordot(s, y_bar, 1)
    return y_bar, theta, sTy, sTy_bar, tf.cast(tiny, tf.float64)


# -----------------------------
# Two‑loop recursion
# -----------------------------
@tf.function
def two_loop_direction(g: tf.Tensor, mem: LBFGSMemory, gamma: tf.Tensor,
                       d0_diag: Optional[tf.Tensor] = None) -> tf.Tensor:
    g = tf.cast(g, tf.float64)
    q = tf.identity(g)
    m = mem.m; L = mem.len
    alpha = tf.TensorArray(tf.float64, size=m, clear_after_read=False)
    # backward
    for i in tf.range(m - 1, -1, delta=-1):
        valid = i >= (m - L)
        s_i = mem.S[i]; y_i = mem.Y[i]; rho_i = mem.rho[i]
        a_i = rho_i * tf.tensordot(s_i, q, 1)
        a_i = tf.where(valid, a_i, 0.0)
        q = tf.where(valid, q - a_i * y_i, q)
        alpha = alpha.write(i, a_i)
    # H0
    if d0_diag is None:
        q = gamma * q
    else:
        q = d0_diag * q
    # forward
    for i in tf.range(0, m, delta=1):
        valid = i >= (m - L)
        s_i = mem.S[i]; y_i = mem.Y[i]; rho_i = mem.rho[i]
        a_i = alpha.read(i)
        b_i = rho_i * tf.tensordot(y_i, q, 1)
        q = tf.where(valid, q + s_i * (a_i - b_i), q)
    return -q


# -----------------------------
# Line search: Armijo + cubic
# -----------------------------
@tf.function
def armijo_ok(f0: tf.Tensor, m0: tf.Tensor, f_a: tf.Tensor, a: tf.Tensor, c1: tf.Tensor) -> tf.Tensor:
    return f_a <= f0 + c1 * a * m0


@tf.function
def cubic_step_closed_form(f0: tf.Tensor, m0: tf.Tensor,
                           a_prev: tf.Tensor, f_prev: tf.Tensor,
                           a: tf.Tensor, f_a: tf.Tensor,
                           low_mult: tf.Tensor, high_mult: tf.Tensor) -> tf.Tensor:
    opt_dtype = f0.dtype
    def default():
        return tf.clip_by_value(0.5 * a, low_mult * a, high_mult * a)
    cond_have_prev = tf.logical_and(tf.math.is_finite(a_prev), tf.math.is_finite(f_prev))
    def compute():
        r1 = f_prev - f0 - m0 * a_prev
        r2 = f_a   - f0 - m0 * a
        M = tf.reshape(tf.stack([a_prev*a_prev, a_prev*a_prev*a_prev,
                                 a*a,           a*a*a], axis=0), [2,2])
        rhs = tf.stack([r1, r2])
        det = M[0,0]*M[1,1] - M[0,1]*M[1,0]
        small = tf.abs(det) < tf.cast(1e-22, opt_dtype)
        def solve_then_min():
            cd = tf.linalg.solve(M, tf.reshape(rhs, [2,1]))
            c = cd[0,0]; d = cd[1,0]
            A = 3.0*d; B = 2.0*c; C = m0
            disc = B*B - 4.0*A*C
            bad = tf.logical_or(~tf.math.is_finite(disc), disc <= 0.0)
            def fallback():
                return default()
            def roots():
                sqrt_disc = tf.sqrt(disc + tf.cast(0.0, opt_dtype))
                t1 = (-B + sqrt_disc) / (2.0*A + tf.cast(1e-30, opt_dtype))
                t2 = (-B - sqrt_disc) / (2.0*A + tf.cast(1e-30, opt_dtype))
                t_candidates = tf.stack([t1, t2])
                mask = tf.logical_and(tf.math.is_finite(t_candidates), t_candidates > 0.0)
                t_candidates = tf.where(mask, t_candidates, tf.fill(tf.shape(t_candidates), default()))
                half = 0.5*a
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
    print_every: int = 100  # outer-loop logging interval (in iterations)
    full_graph_ls: bool = True
    # Adam-style H0
    use_adam_h0: bool = False
    beta2: float = 0.999
    diag_eps: float = 1e-8
    # Inner compiled chunk size
    chunk_size: int = 100


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
        self.opt_dtype = opt_dtype
        self.use_xla = use_xla

        # Normalize params
        if isinstance(params, (list, tuple)):
            self.params_list: Optional[List[TensorLike]] = list(params)
            self.spec = ParamSpec.from_params(self.params_list)
            theta0 = self.spec.flatten(self.params_list, to_dtype=opt_dtype)
        else:
            self.params_list = None
            self.spec = ParamSpec([Seg(0, int(params.shape[0]), (int(params.shape[0]),), params.dtype)], int(params.shape[0]))
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
                return tf.cast(loss, opt_dtype)

            @tf.function(jit_compile=use_xla)
            def loss_and_grad(theta: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
                plist = self.spec.unflatten(theta)
                out = self.loss_fn(plist)
                if isinstance(out, (list, tuple)) and len(out) == 2:
                    loss, grads_struct = out
                    gflat = self.spec.flatten(grads_struct, to_dtype=opt_dtype)
                    return tf.cast(loss, opt_dtype), tf.cast(gflat, opt_dtype)
                # compute grads ourselves
                with tf.GradientTape() as tape:
                    for p in plist:
                        tape.watch(p)
                    loss = self.loss_fn(plist)
                grads = tape.gradient(loss, plist)
                gflat = self.spec.flatten(grads, to_dtype=opt_dtype)
                return tf.cast(loss, opt_dtype), tf.cast(gflat, opt_dtype)

        elif mode == 'assign':
            assert self.params_list is not None, "assign mode requires a params list of Variables"
            vars_list = [tf.convert_to_tensor(p) if not isinstance(p, tf.Variable) else p for p in self.params_list]

            @tf.function(jit_compile=use_xla)
            def loss_only(theta: tf.Tensor) -> tf.Tensor:
                self.spec.assign_to_vars(theta, vars_list)
                out = self.loss_fn()
                if isinstance(out, (list, tuple)):
                    loss = out[0]
                else:
                    loss = out
                return tf.cast(loss, opt_dtype)

            @tf.function(jit_compile=use_xla)
            def loss_and_grad(theta: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
                self.spec.assign_to_vars(theta, vars_list)
                out = self.loss_fn()
                if isinstance(out, (list, tuple)) and len(out) == 2:
                    loss, grads_struct = out
                    gflat = self.spec.flatten(grads_struct, to_dtype=opt_dtype)
                    return tf.cast(loss, opt_dtype), tf.cast(gflat, opt_dtype)
                with tf.GradientTape() as tape:
                    tape.watch(vars_list)
                    loss = self.loss_fn()
                grads = tape.gradient(loss, vars_list)
                gflat = self.spec.flatten(grads, to_dtype=opt_dtype)
                return tf.cast(loss, opt_dtype), tf.cast(gflat, opt_dtype)
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
                 lbfgs_precision: str = Precision.FP64,
                 use_xla: bool = False):
        self.cfg = cfg
        self.opt_dtype = tf.float64 if lbfgs_precision == Precision.FP64 else tf.float32
        self.adapter = loss_adapter
        self.theta = tf.Variable(tf.cast(loss_adapter.theta0, self.opt_dtype), trainable=False)
        self.n = int(self.theta.shape[0])
        self.mem = LBFGSMemory(m=cfg.mem, n=self.n)
        # Adam diag state
        self.adam_v = tf.Variable(tf.zeros([self.n], dtype=self.opt_dtype), trainable=False)
        self.adam_t = tf.Variable(0, dtype=tf.int64, trainable=False)
        self.use_xla = use_xla

        @tf.function(jit_compile=use_xla)
        def _adam_diag(gflat: tf.Tensor, beta2: tf.Tensor, eps: tf.Tensor) -> tf.Tensor:
            self.adam_t.assign_add(1)
            self.adam_v.assign(beta2 * self.adam_v + (1.0 - beta2) * tf.square(gflat))
            t = tf.cast(self.adam_t, self.opt_dtype)
            v_hat = self.adam_v / (1.0 - tf.pow(beta2, t) + tf.cast(1e-16, self.opt_dtype))
            return tf.cast(1.0 / tf.sqrt(v_hat + eps), tf.float64)
        self._adam_diag = _adam_diag

        # Initial loss & grad
        f0, g0 = self.adapter.loss_and_grad(self.theta)
        self.f = tf.Variable(f0, trainable=False)
        self.g = tf.Variable(g0, trainable=False)
        self.gamma = tf.Variable(tf.cast(1.0, tf.float64), trainable=False)
        self.mem.clear()

        # Build inner compiled K-step runner
        self._build_inner_runner()

    def _build_inner_runner(self):
        cfg = self.cfg
        opt_dtype = self.opt_dtype
        adapter = self.adapter
        mem = self.mem
        use_xla = self.use_xla

        @tf.function(jit_compile=use_xla)
        def loss_only(theta):
            return adapter.loss_only(theta)

        @tf.function(jit_compile=use_xla)
        def loss_and_grad(theta):
            return adapter.loss_and_grad(theta)

        @tf.function(jit_compile=use_xla)
        def line_search(theta, f0, g, d) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
            alpha0 = tf.cast(cfg.alpha_init, opt_dtype)
            c1 = tf.cast(cfg.c1, opt_dtype)
            low_mult = tf.cast(cfg.cub_clip[0], opt_dtype)
            high_mult = tf.cast(cfg.cub_clip[1], opt_dtype)
            ls_max = tf.cast(cfg.ls_max_steps, tf.int32)
            m0 = tf.tensordot(g, d, 1)
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
                                     lambda: cubic_step_closed_form(f0, m0, a_prev, f_prev, alpha, f_a, low_mult, high_mult))
                a_prev_next = tf.cond(ok, lambda: a_prev, lambda: alpha_next)
                f_prev_next = tf.cond(ok, lambda: f_prev, lambda: tf.where(tf.math.is_finite(f_a), f_a, f_prev))
                return i+1, alpha_next, a_prev_next, f_prev_next, best_f, best_a, accepted

            i0 = tf.constant(0, tf.int32)
            a_prev0 = tf.cast(tf.constant(float('nan')), opt_dtype)
            f_prev0 = tf.cast(tf.constant(float('nan')), opt_dtype)
            best_f0 = tf.identity(f0)
            best_a0 = tf.cast(0.0, opt_dtype)
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
                # H0 diag (optional)
                d0_diag = None
                if cfg.use_adam_h0:
                    d0_diag = self._adam_diag(g, tf.cast(cfg.beta2, opt_dtype), tf.cast(cfg.diag_eps, opt_dtype))
                d = two_loop_direction(g, mem, gamma, d0_diag)
                alpha, accepted, ls_iters, m0, d_used, best_f = line_search(theta, f, g, d)
                theta_next = theta + tf.cast(alpha, opt_dtype) * d
                f_next, g_next = loss_and_grad(theta_next)
                s = tf.cast(theta_next - theta, tf.float64)
                y = tf.cast(g_next - g, tf.float64)
                y_bar, theta_mix, sTy_raw, sTy_bar, tiny = powell_damp_pair(s, y, cfg.powell_c, gamma)
                cos_sy = (sTy_bar / (tf.sqrt(tf.tensordot(s, s, 1) * (tf.tensordot(y_bar, y_bar, 1) + 1e-18) + 1e-18)))
                pair_ok = tf.logical_and(sTy_bar > 1e-12, cos_sy >= cfg.curvature_cos_tol)
                def push_pair():
                    mem.push(s, y_bar)
                    # gamma update
                    return tf.cast(tf.maximum(sTy_bar / (tf.tensordot(y_bar, y_bar, 1) + 1e-18), 1e-8), tf.float64)
                gamma_next = tf.cond(pair_ok, push_pair, lambda: gamma)
                # advance
                return (it+1, theta_next, f_next, g_next, gamma_next,
                        alpha, ls_iters, mem.len, sTy_bar, cos_sy)

            it, theta, f, g, gamma, alpha_last, ls_last, mem_len, sTy_bar_last, cos_sy_last = tf.while_loop(
                cond, body,
                loop_vars=(it, theta, f, g, gamma,
                           tf.cast(0.0, opt_dtype), tf.cast(0, tf.int32), tf.cast(0, tf.int32), tf.cast(0.0, tf.float64), tf.cast(0.0, tf.float64)),
                maximum_iterations=k)
            return theta, f, g, gamma, alpha_last, ls_last, mem_len, sTy_bar_last, cos_sy_last

        self._loss_only = loss_only
        self._loss_and_grad = loss_and_grad
        self._line_search = line_search
        self._inner_k_steps = inner_k_steps

    # ---------- public API ----------
    def minimize_chunked(self, total_iters: int, chunk_size: Optional[int] = None, print_every: Optional[int] = None) -> None:
        if chunk_size is None:
            chunk_size = self.cfg.chunk_size
        if print_every is None:
            print_every = self.cfg.print_every

        t0 = time.perf_counter()
        # Iter 0 already computed in __init__
        print(f"Iter 0 : loss= {float(self.f.numpy())} , g_norm= {float(tf.linalg.norm(self.g).numpy())} , mem= {int(self.mem.len.numpy())} , t= 0.000s")

        total_done = 0
        while total_done < total_iters:
            k = min(chunk_size, total_iters - total_done)
            theta, f, g, gamma, alpha_last, ls_last, mem_len, sTy_bar_last, cos_sy_last = self._inner_k_steps(
                self.theta, self.f, self.g, self.gamma, tf.cast(k, tf.int32))
            # update captured state vars
            self.theta.assign(theta); self.f.assign(f); self.g.assign(g); self.gamma.assign(gamma)
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
    y = std_normal_cdf(tf.convert_to_tensor(z, dtype=tf.float32)).numpy().astype(np.float32)
    return tf.convert_to_tensor(X, dtype=dtype), tf.convert_to_tensor(y, dtype=dtype)


def build_mlp(seed: int = 0) -> tf.keras.Model:
    k = tf.keras.initializers.GlorotUniform
    z = tf.keras.initializers.Zeros()
    inp = tf.keras.Input(shape=(3,))
    h = tf.keras.layers.Dense(16, activation='tanh', kernel_initializer=k(seed+1), bias_initializer=z)(inp)
    h = tf.keras.layers.Dense(16, activation='tanh', kernel_initializer=k(seed+2), bias_initializer=z)(h)
    h = tf.keras.layers.Dense(16, activation='tanh', kernel_initializer=k(seed+3), bias_initializer=z)(h)
    out = tf.keras.layers.Dense(1, activation='sigmoid', kernel_initializer=k(seed+4), bias_initializer=z)(h)
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
    set_model_precision(Precision.FP32)
    model = build_mlp(seed=seed)

    # Loss function (assign mode): assign theta -> model vars, compute MSE
    vars_list = model.trainable_variables
    spec = ParamSpec.from_params(vars_list)

    def loss_fn():
        pred = tf.cast(model(X, training=False), Y.dtype)
        e = tf.cast(Y, pred.dtype) - pred
        return tf.reduce_mean(tf.square(e))  # MSE

    adapter = LossAdapter(loss_fn=loss_fn, params=vars_list, mode='assign', opt_dtype=tf.float64, use_xla=False)

    cfg = LBFGSConfig(max_iters=260, mem=10, print_every=30, chunk_size=30,
                      use_adam_h0=True, beta2=0.999, diag_eps=1e-8,
                      alpha_init=1.0, ls_max_steps=8, c1=1e-4, cub_clip=(0.1, 2.5), powell_c=0.2)

    opt = LBFGS(adapter, cfg, lbfgs_precision=Precision.FP64, use_xla=False)

    # Outer loop (prints every 100 iters). For debugging: set chunk_size=1
    opt.minimize_chunked(total_iters=cfg.max_iters, chunk_size=cfg.chunk_size, print_every=cfg.print_every)


demo_main()
