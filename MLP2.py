#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
import math
import dataclasses
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os, random


# ==========================
# Plot helpers
# ==========================

def plot_convergence_AB(curve_a: List[Dict[str, Any]], curve_b: List[Dict[str, Any]], skip_a=True):
    def unpack(curve):
        ts = [r['t'] for r in curve]
        ls = [r['loss'] for r in curve]
        return ts, ls

    if not skip_a: ta, la = unpack(curve_a)
    tb, lb = unpack(curve_b)
    plt.figure()
    if not skip_a: plt.plot(ta, la, label='A: L-BFGS')
    plt.plot(tb, lb, label='B: L-BFGS + Adam diag')
    plt.xlabel('Wall time (s)')
    plt.ylabel('√ mse')
    plt.yscale('log')
    plt.title('Convergence (θ training)')
    plt.legend();
    plt.grid(True, which='both', ls=':')


def plot_convergence_dual(curve: List[Dict[str, Any]], title='Ω Convergence'):
    """Left axis: √MSE; Right axis: penalty (Jac or Var)."""
    t = [r['t'] for r in curve]
    sqrt_mse = [r['sqrt_mse'] for r in curve]
    pen = [r['pen'] for r in curve]
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(t, sqrt_mse, label='√MSE')
    ax2.plot(t, pen, linestyle='--', label='Penalty')
    ax1.set_xlabel('Wall time (s)')
    ax1.set_ylabel('√MSE');
    ax1.set_yscale('log');
    ax1.grid(True, which='both', ls=':')
    ax2.set_ylabel('Penalty');
    ax2.set_yscale('log')
    lines, labels = [], []
    for ax in (ax1, ax2):
        L = ax.get_lines();
        lines += L;
        labels += [l.get_label() for l in L]
    fig.legend(lines, labels, loc='upper right')
    fig.suptitle(title)
    fig.tight_layout()


def plot_slices(model_a: tf.keras.Model, model_b: tf.keras.Model, a: float = 0.0, b: float = 1.0, skip_a=True):
    t_vals = [0.1, 0.5, 0.9]
    x = np.linspace(-4, 4, 401).astype(np.float32).reshape(-1, 1)
    plt.figure(figsize=(10, 7))
    for i, t0 in enumerate(t_vals, 1):
        t = np.full_like(x, t0, dtype=np.float32)
        s = np.sqrt(1.0 - t).astype(np.float32)
        X = np.concatenate([t, x, s], axis=1).astype(np.float32)
        if not skip_a: ya = model_a(X, training=False).numpy().reshape(-1)
        yb = model_b(X, training=False).numpy().reshape(-1)
        z = (x - a) / (b * s)
        y_true = std_normal_cdf(tf.convert_to_tensor(z)).numpy().reshape(-1)
        plt.subplot(3, 1, i)
        plt.plot(x.reshape(-1), y_true, label=f'True Φ, t={t0}')
        if not skip_a: plt.plot(x.reshape(-1), ya, label='Model A')
        plt.plot(x.reshape(-1), yb, label='Ω model')
        plt.grid(True, ls=':');
        plt.legend()
    plt.suptitle('Predictions vs Φ at fixed t')


def update_H0_from_grad(g, v, beta2, eps, target_med):
    v = beta2 * v + (1 - beta2) * (g * g)
    v_hat = v
    d0 = 1.0 / (np.sqrt(v_hat) + eps)
    med = np.median(d0)
    if med > 0: d0 *= (target_med / med)
    d0 = np.clip(d0, 1e-8, 1e+2)
    return v, d0


# ==========================
# Target & data
# ==========================

def std_normal_cdf(z: tf.Tensor) -> tf.Tensor:
    return 0.5 * (1.0 + tf.math.erf(z / tf.sqrt(tf.constant(2.0, dtype=z.dtype))))


def make_dataset(n: int, seed: int = 0, a: float = 0.0, b: float = 1.0) -> Tuple[tf.Tensor, tf.Tensor]:
    rng = np.random.default_rng(seed)
    t = rng.uniform(0.0, 1.0, size=(n, 1)).astype(np.float32)
    x = rng.uniform(-4.0, 4.0, size=(n, 1)).astype(np.float32)
    s = np.sqrt(1.0 - t).astype(np.float32)
    X = np.concatenate([t, x, s], axis=1).astype(np.float32)
    z = (x - a) / (b * s)
    y = std_normal_cdf(tf.convert_to_tensor(z)).numpy().astype(np.float32)
    return tf.convert_to_tensor(X), tf.convert_to_tensor(y)


# ==========================
# Model f (θ-parameterized MLP)
# ==========================

def build_mlp() -> tf.keras.Model:
    inp = tf.keras.Input(shape=(3,), dtype=tf.float32)
    h = tf.keras.layers.Dense(16, activation='tanh')(inp)
    h = tf.keras.layers.Dense(16, activation='tanh')(h)
    h = tf.keras.layers.Dense(16, activation='tanh')(h)
    out = tf.keras.layers.Dense(1, activation='sigmoid')(h)
    return tf.keras.Model(inputs=inp, outputs=out)


ARCH_F = [3, 16, 16, 16, 1]


def count_f_params(arch=ARCH_F) -> int:
    P = 0
    for din, dout in zip(arch[:-1], arch[1:]): P += din * dout + dout
    return P  # 625 for [3,16,16,16,1]


def _unpack_theta(theta_flat: tf.Tensor, arch=ARCH_F):
    idx = 0;
    params = []
    for din, dout in zip(arch[:-1], arch[1:]):
        wsz, bsz = din * dout, dout
        W = tf.reshape(theta_flat[idx:idx + wsz], [din, dout]);
        idx += wsz
        b = tf.reshape(theta_flat[idx:idx + bsz], [dout]);
        idx += bsz
        params.append((W, b))
    return params


def _unpack_theta_batched(theta_all: tf.Tensor, arch=ARCH_F):
    N = tf.shape(theta_all)[0]
    idx = 0;
    params = []
    for din, dout in zip(arch[:-1], arch[1:]):
        wsz, bsz = din * dout, dout
        W = tf.reshape(theta_all[:, idx:idx + wsz], [N, din, dout]);
        idx += wsz
        B = tf.reshape(theta_all[:, idx:idx + bsz], [N, dout]);
        idx += bsz
        params.append((W, B))
    return params


def f_forward_single_theta(X: tf.Tensor, theta_flat: tf.Tensor, arch=ARCH_F) -> tf.Tensor:
    params = _unpack_theta(theta_flat, arch)
    h = X
    for (W, b) in params[:-1]: h = tf.tanh(tf.matmul(h, W) + b)
    W, b = params[-1];
    y = tf.matmul(h, W) + b
    return tf.math.sigmoid(y)


def f_forward_per_sample_theta(X: tf.Tensor, theta_all: tf.Tensor, arch=ARCH_F) -> tf.Tensor:
    params = _unpack_theta_batched(theta_all, arch)
    h = X
    for (W, B) in params[:-1]:
        h = tf.tanh(tf.einsum('ni,nio->no', h, W) + B)
    W, B = params[-1]
    y = tf.einsum('ni,nio->no', h, W) + B
    return tf.math.sigmoid(y)


# ==========================
# Pack/Unpack for any Keras model
# ==========================

def pack_variables(vars_list: List[tf.Variable]) -> tf.Tensor:
    return tf.concat([tf.reshape(v, [-1]) for v in vars_list], axis=0)


def unpack_to_variables(flat: tf.Tensor, vars_list: List[tf.Variable]) -> None:
    offset = 0
    for v in vars_list:
        size = tf.size(v)
        new_vals = tf.reshape(flat[offset:offset + size], v.shape)
        v.assign(tf.cast(new_vals, v.dtype))
        offset += size


# ==========================
# θ loss
# ==========================

def mse_loss(model: tf.keras.Model, X: tf.Tensor, Y: tf.Tensor) -> tf.Tensor:
    pred = model(X, training=False)
    return tf.reduce_mean(tf.square(pred - Y))


# ==========================
# L-BFGS memory & two-loop
# ==========================
@dataclasses.dataclass
class LBFGSMemory:
    m: int = 10
    S: List[np.ndarray] = dataclasses.field(default_factory=list)
    Y: List[np.ndarray] = dataclasses.field(default_factory=list)

    def clear(self):
        self.S.clear();
        self.Y.clear()

    def push(self, s: np.ndarray, y: np.ndarray):
        self.S.append(s.astype(np.float64));
        self.Y.append(y.astype(np.float64))
        if len(self.S) > self.m: self.S.pop(0); self.Y.pop(0)

    def two_loop(self, g: np.ndarray, gamma: float, d0=None) -> np.ndarray:
        S, Y = self.S, self.Y
        q = g.astype(np.float64).copy();
        alpha, rho = [], []
        for s, y in zip(reversed(S), reversed(Y)):
            r = 1.0 / (np.dot(y, s) + 1e-18);
            rho.append(r)
            a = r * np.dot(s, q);
            alpha.append(a);
            q -= a * y
        q *= gamma if d0 is None else d0
        for (s, y, r, a) in zip(S, Y, reversed(rho), reversed(alpha)):
            b = r * np.dot(y, q);
            q += s * (a - b)
        return q


# ==========================
# Powell damping
# ==========================
def powell_damp_pair(s: np.ndarray, y: np.ndarray, c: float, gamma: float) -> Tuple[
    np.ndarray, float, float, float, bool]:
    ss, yy = np.dot(s, s), np.dot(y, y)
    sBs = (1.0 / max(gamma, 1e-16)) * ss
    sy = float(np.dot(s, y))
    if sy < 0: y = -y
    asy = abs(sy)
    fl = ss * yy * 1e-9;
    tiny = False
    if asy < fl:
        tiny = True;
        a = fl / yy;
        y = a * y;
        asy = fl
    if asy >= c * sBs: return y, 1.0, sy, asy, tiny
    theta = (1.0 - c) * sBs / max(sBs - asy, 1e-16)
    y_bar = theta * y + (1.0 - theta) * (1.0 / max(gamma, 1e-16)) * s
    return y_bar, theta, sy, float(np.dot(s, y_bar)), tiny


# ==========================
# Line search
# ==========================
def armijo_condition(f0: float, m0: float, f_a: float, a: float, c1: float) -> bool:
    return f_a <= f0 + c1 * a * m0


def cubic_step(f0, m0, a_prev, f_prev, a, f_a, low_mult, high_mult) -> float:
    if a_prev is None or f_prev is None:
        return float(np.clip(0.5 * a, low_mult * a, high_mult * a))
    try:
        A = np.array([[a_prev ** 2, a_prev, 1.0],
                      [a ** 2, a, 1.0],
                      [0.0, 1.0, 0.0]], dtype=float)
        b = np.array([f_prev - f0, f_a - f0, m0], dtype=float)
        coef = np.linalg.lstsq(A, b, rcond=None)[0]
        a_new = -coef[1] / (2.0 * coef[0]) if abs(coef[0]) > 1e-18 else 0.5 * a
    except Exception:
        a_new = 0.5 * a
    a_new = float(np.clip(a_new, low_mult * a, high_mult * a))
    if not np.isfinite(a_new) or a_new <= 0: a_new = float(np.clip(0.5 * a, low_mult * a, high_mult * a))
    return a_new


# ==========================
# Config & Runner (θ training)
# ==========================
@dataclasses.dataclass
class LBFGSConfig:
    max_iters: int = 200
    mem: int = 10
    c1: float = 1e-4
    powell_c: float = 0.2
    ls_max_steps: int = 4
    alpha_init: float = 1.0
    cub_clip: Tuple[float, float] = (0.1, 2.5)
    quad_clip: Tuple[float, float] = (0.1, 2.5)
    sigma_es: float = 1e-2
    sigma_decay: float = 0.4
    sigma_min: float = 1e-8
    curvature_cos_tol: float = 1e-6
    diag_true_grad: bool = False
    adam_diagonal: bool = False


class LBFGSRunner:
    def __init__(self, model: tf.keras.Model, X: tf.Tensor, Y: tf.Tensor, cfg: LBFGSConfig, seed: int = 0):
        self.model = model;
        self.X = X;
        self.Y = Y;
        self.cfg = cfg
        self.mem = LBFGSMemory(m=cfg.mem);
        self.rng = np.random.default_rng(seed)
        self.vars = model.trainable_variables

    # ----- IO -----
    def get_x_tf64(self) -> tf.Tensor:
        return tf.cast(pack_variables(self.vars), tf.float64)

    def set_x_tf32(self, x_np64: np.ndarray) -> None:
        unpack_to_variables(tf.cast(tf.convert_to_tensor(x_np64), tf.float32), self.vars)

    @tf.function(jit_compile=False)
    def _tf_loss(self) -> tf.Tensor:  # θ training uses plain MSE
        return mse_loss(self.model, self.X, self.Y)

    def loss(self) -> float:
        return float(self._tf_loss().numpy())

    def loss_at_x(self, x_np64: np.ndarray) -> float:
        self.set_x_tf32(x_np64);
        return self.loss()

    def true_grad(self) -> np.ndarray:
        with tf.GradientTape() as tape:
            loss = self._tf_loss()
        grads = tape.gradient(loss, self.vars)
        flat = tf.concat([tf.reshape(g, [-1]) for g in grads], axis=0)
        return flat.numpy().astype(np.float64)

    # ----- Directions -----
    def direction(self, g: np.ndarray, gamma: float, d0=None) -> np.ndarray:
        if len(self.mem.S) == 0: return -((gamma if d0 is None else d0) * g)
        return -self.mem.two_loop(g, gamma, d0)

    # ----- Line search -----
    def line_search(self, theta: np.ndarray, f0: float, g: np.ndarray, d: np.ndarray,
                    mode: str, m0: Optional[float] = None) -> Tuple[float, bool, List[Tuple[float, float, bool]], int]:
        cfg = self.cfg
        if m0 is None: m0 = float(np.dot(g, d))
        if m0 >= 0: d = -g.copy(); m0 = float(np.dot(g, d))
        alpha = cfg.alpha_init;
        a_prev = None;
        f_prev = None
        best_f, best_a = f0, 0;
        trace = [];
        accepted = False;
        count = 0
        for _ in range(cfg.ls_max_steps):
            count += 1
            theta_trial = theta + alpha * d
            self.set_x_tf32(tf.convert_to_tensor(theta_trial))
            f_a = self.loss()
            ok = armijo_condition(f0, m0, f_a, alpha, cfg.c1)
            trace.append((alpha, f_a, ok))
            if f_a < best_f: best_f, best_a = f_a, alpha
            if ok:
                if best_f < f0: alpha = best_a
                accepted = True;
                break
            alpha = cubic_step(f0, m0, a_prev, f_prev, alpha, f_a, *cfg.cub_clip)
            a_prev, f_prev = alpha, f_a
        if not accepted:
            if best_f < f0 and best_a is not None:
                alpha = best_a;
                accepted = True
            else:
                alpha = 1e-4 / (np.linalg.norm(d) + 1e-12)
        return float(alpha), accepted, trace, count

    # ----- Run (used by θ training) -----
    def run_autodiff(self) -> List[Dict[str, Any]]:
        cfg = self.cfg;
        history: List[Dict[str, Any]] = [];
        t0 = time.perf_counter()
        theta = self.get_x_tf64().numpy();
        f0 = self.loss();
        g = self.true_grad();
        gamma = 1.0
        print(f"Iter 0: loss={f0:.6f}")
        history.append(dict(iter=0, t=0.0, loss=f0))
        d_prev = None;
        v = np.zeros_like(theta);
        d0 = None;
        beta2, eps = 0.999, 1e-8
        if cfg.adam_diagonal: v, d0 = update_H0_from_grad(g, v, beta2, eps, gamma)
        total_ls_iter = 0
        for it in range(1, cfg.max_iters + 1):
            d = self.direction(g, gamma, d0)
            angle_pi = float('nan')
            if d_prev is not None:
                nd = np.linalg.norm(d);
                ndp = np.linalg.norm(d_prev)
                if nd > 0 and ndp > 0:
                    c = np.clip(np.dot(d_prev, d) / (ndp * nd), -1.0, 1.0)
                    angle_pi = float(np.arccos(c) / np.pi)
            d_prev = d.copy()
            m0 = float(np.dot(g, d))
            alpha, accepted, ls_trace, ls_iter = self.line_search(theta, f0, g, d, mode='autodiff', m0=m0)
            total_ls_iter += ls_iter
            theta_next = theta + alpha * d
            self.set_x_tf32(tf.convert_to_tensor(theta_next))
            f_next = self.loss();
            g_next = self.true_grad()
            s = (theta_next - theta).astype(np.float64);
            y = (g_next - g).astype(np.float64)
            y_bar, theta_mix, sTy, sTyb, _ = powell_damp_pair(s, y, cfg.powell_c, gamma)
            cos_sy = float(sTyb / (np.linalg.norm(s) * (np.linalg.norm(y_bar) + 1e-18) + 1e-18))
            pair_accepted = False
            if sTyb > 1e-12 and cos_sy >= cfg.curvature_cos_tol:
                self.mem.push(s, y_bar);
                pair_accepted = True
                gamma = max(float(sTyb / (np.dot(y_bar, y_bar) + 1e-18)), 1e-8)
                if cfg.adam_diagonal: v, d0 = update_H0_from_grad(g, v, beta2, eps, gamma)
            theta, f0, g = theta_next, f_next, g_next
            t_now = time.perf_counter() - t0;
            sqrt_f = math.sqrt(f_next)
            rec = dict(iter=it, t=t_now, loss=sqrt_f, alpha=alpha, armijo=accepted,
                       g_norm=float(np.linalg.norm(g)), d_dot_g=float(np.dot(d, g)),
                       sTy=sTy, sTy_bar=sTyb, cos_sy=cos_sy, s_norm=float(np.linalg.norm(s)),
                       ybar_norm=float(np.linalg.norm(y_bar)), mem=len(self.mem.S), gamma=gamma,
                       ls_trace=ls_trace)
            history.append(rec)
            if it % 10 == 0:
                print(
                    f"Iter {it}: t={t_now:7.3f}s, , angle={angle_pi:.4f}, loss={sqrt_f:.5f}, d.g>0:{m0 > 0}, "
                    f"armijo:{accepted}, alpha={alpha:.3e}, iter={total_ls_iter}, |g|={rec['g_norm']:.3e}, "
                    f"neg_sTy={sTy < 0}, powel={theta_mix != 1.0}, pair ok:{pair_accepted}, mem={len(self.mem.S)}")
        return history


# ==========================
# g model: (X,T)->θ and Ω runner
# ==========================
def build_g(P: int) -> tf.keras.Model:
    inp = tf.keras.Input(shape=(4,), dtype=tf.float32)  # [t, x, s, T]
    h = tf.keras.layers.Dense(16, activation='tanh')(inp)
    h = tf.keras.layers.Dense(16, activation='tanh')(h)
    h = tf.keras.layers.Dense(16, activation='tanh')(h)
    out = tf.keras.layers.Dense(P, activation=None)(h)
    return tf.keras.Model(inp, out)


# ----- Penalties -----
def variance_penalty(theta_all: tf.Tensor) -> tf.Tensor:
    """mean_i mean_n (g_i(n) - mean_n g_i)**2  (scalar)"""
    mean_over_n = tf.reduce_mean(theta_all, axis=0, keepdims=True)  # [1,P]
    centered = theta_all - mean_over_n  # [N,P]
    pen = tf.reduce_mean(tf.reduce_mean(tf.square(centered), axis=0))  # mean over n, then mean over i
    return pen


def jacobian_frob_penalty_hutch(g_model: tf.keras.Model, XT: tf.Tensor,
                                probes: int = 1, penalty_subsample: int = 0) -> tf.Tensor:
    """
    Hutchinson VJP estimator of ||J||_F^2: E_v ||J^T v||^2, with v in R^P.
    Differentiable wrt ω; memory-safe. Optionally subsample rows for the penalty.
    """
    N = tf.shape(XT)[0]
    if penalty_subsample and penalty_subsample > 0:
        M = tf.minimum(tf.constant(penalty_subsample, dtype=tf.int32), N)
        idx = tf.random.uniform((M,), 0, N, dtype=tf.int32)
        XT_pen = tf.gather(XT, idx, axis=0)
        scale = tf.cast(N, tf.float32) / tf.cast(M, tf.float32)
    else:
        XT_pen = XT;
        scale = tf.constant(1.0, tf.float32)

    pen = tf.constant(0.0, dtype=XT.dtype)
    P = g_model.output_shape[-1]
    for _ in range(int(max(1, probes))):
        v = tf.random.normal([P], dtype=XT.dtype)  # probe in output space
        with tf.GradientTape() as tape_in:
            tape_in.watch(XT_pen)
            theta_pen = g_model(XT_pen, training=True)  # [M,P]
            s = tf.tensordot(theta_pen, v, axes=[[1], [0]])  # [M]
        JT_v = tape_in.gradient(s, XT_pen)  # [M,4]
        pen += tf.reduce_sum(tf.square(JT_v))
    return scale * (pen / float(max(1, probes)))


def jacobian_frob_penalty_exact_jvp(g_model: tf.keras.Model, XT: tf.Tensor, chunk: int = 2048) -> tf.Tensor:
    """
    Exact sum_j ||J col_j||^2 via 4 forward-mode JVPs (one per input coord), chunked.
    NOTE: Higher-order gradients wrt ω through ForwardAccumulator can depend on TF version.
    If you see None grads, switch to the Hutchinson estimator above.
    """
    pen = tf.constant(0.0, dtype=XT.dtype)
    N = tf.shape(XT)[0]
    for start in tf.range(0, N, delta=chunk):
        end = tf.minimum(start + chunk, N)
        XTc = XT[start:end]
        for j in range(4):
            vj = tf.one_hot(j, 4, dtype=XT.dtype);
            vj = tf.broadcast_to(vj, tf.shape(XTc))
            with tf.autodiff.ForwardAccumulator(primals=XTc, tangents=vj) as acc:
                theta_c = g_model(XTc, training=True)  # [m,P]
            col = acc.jvp(theta_c)  # [m,P]
            pen += tf.reduce_sum(tf.square(col))
    return pen


class LBFGSRunnerOmega(LBFGSRunner):
    """
    Ω-runner reuses the L-BFGS machinery but overrides loss/grad to:
      loss = MSE + lam * (Jacobian Frobenius penalty  OR  Var(θ) penalty)
    Options:
      - per_sample_theta: True → f(x_n; θ_n), False → f(X; θ̄)
      - penalty_type: 'jac' or 'var'
      - jac_mode: 'hutch' (default) or 'exact'
      - hutch_probes, penalty_subsample, exact_chunk
    """

    def __init__(self, g_model: tf.keras.Model, X: tf.Tensor, Y: tf.Tensor,
                 cfg: LBFGSConfig, lam: float = 1e-6, per_sample_theta: bool = True,
                 penalty_type: str = 'jac', jac_mode: str = 'hutch',
                 hutch_probes: int = 1, penalty_subsample: int = 0, exact_chunk: int = 2048,
                 seed: int = 0):
        super().__init__(g_model, X, Y, cfg, seed)
        self.vars = g_model.trainable_variables
        self.P = count_f_params(ARCH_F)
        self.lam = tf.constant(lam, dtype=tf.float32)
        self.per_sample_theta = bool(per_sample_theta)
        self.penalty_type = penalty_type  # 'jac' or 'var'
        self.jac_mode = jac_mode  # 'hutch' or 'exact'
        self.hutch_probes = int(hutch_probes)
        self.penalty_subsample = int(penalty_subsample)
        self.exact_chunk = int(exact_chunk)
        # caches for logging
        self._last_mse_val = 0.0
        self._last_pen_val = 0.0
        self._last_pen_name = 'jac' if penalty_type == 'jac' else 'var'
        self.with_penalty = lam != 0

    # ---- components as tensors (no .numpy here) ----
    def _components_tensor(self) -> Tuple[tf.Tensor, tf.Tensor]:
        XT = tf.concat([self.X, self.Y], axis=1)  # [N,4]
        theta_all = self.model(XT, training=True)  # [N,P]
        if self.per_sample_theta:
            y_hat = f_forward_per_sample_theta(self.X, theta_all, ARCH_F)
        else:
            theta_bar = tf.reduce_mean(theta_all, axis=0)  # [P]
            y_hat = f_forward_single_theta(self.X, theta_bar, ARCH_F)
        mse = tf.reduce_mean(tf.square(y_hat - self.Y))
        if self.with_penalty:
            if self.penalty_type == 'var':
                pen = variance_penalty(theta_all)
            else:
                if self.jac_mode == 'exact':
                    pen = jacobian_frob_penalty_exact_jvp(self.model, XT, chunk=self.exact_chunk)
                else:
                    pen = jacobian_frob_penalty_hutch(self.model, XT, probes=self.hutch_probes,
                                                      penalty_subsample=self.penalty_subsample)
        else:
            pen = None
        return mse, pen

    # ---- scalar loss (eager) with caching for logging ----
    def loss(self) -> float:
        mse_t, pen_t = self._components_tensor()
        self._last_mse_val = float(mse_t.numpy())
        if pen_t is None:
            self._last_pen_val = 0.0
            total = self._last_mse_val
        else:
            self._last_pen_val = float(pen_t.numpy())
            total = self._last_mse_val + float(self.lam.numpy()) * self._last_pen_val
        return total

    # ---- true grad wrt ω (uses the same components) ----
    def true_grad(self) -> np.ndarray:
        with tf.GradientTape() as tape:
            mse_t, pen_t = self._components_tensor()
            if pen_t is None:
                loss_t=mse_t
            else:
                loss_t = mse_t + self.lam * pen_t
        grads = tape.gradient(loss_t, self.vars)
        flat = tf.concat([tf.reshape(g, [-1]) for g in grads], axis=0)
        return flat.numpy().astype(np.float64)

    # ---- run with original print line, splitting the loss ----
    def run_autodiff(self) -> List[Dict[str, Any]]:
        cfg = self.cfg;
        history: List[Dict[str, Any]] = [];
        t0 = time.perf_counter()
        theta = self.get_x_tf64().numpy()
        f0_total = self.loss()  # caches _last_mse_val, _last_pen_val
        g = self.true_grad();
        gamma = 1.0
        print(
            f"Iter 0: total={f0_total:.6f}, mse={self._last_mse_val:.6f}, {self._last_pen_name}={self._last_pen_val:.6f}")
        history.append(dict(iter=0, t=0.0, sqrt_mse=math.sqrt(self._last_mse_val),
                            pen=self._last_pen_val, total=f0_total))
        d_prev = None;
        v = np.zeros_like(theta);
        d0 = None;
        beta2, eps = 0.999, 1e-8
        if cfg.adam_diagonal: v, d0 = update_H0_from_grad(g, v, beta2, eps, gamma)
        total_ls_iter = 0
        for it in range(1, cfg.max_iters + 1):
            d = self.direction(g, gamma, d0)
            angle_pi = float('nan')
            if d_prev is not None:
                nd = np.linalg.norm(d);
                ndp = np.linalg.norm(d_prev)
                if nd > 0 and ndp > 0:
                    c = np.clip(np.dot(d_prev, d) / (ndp * nd), -1.0, 1.0)
                    angle_pi = float(np.arccos(c) / np.pi)
            d_prev = d.copy()
            m0 = float(np.dot(g, d))
            alpha, accepted, ls_trace, ls_iter = self.line_search(theta, f0_total, g, d, mode='autodiff', m0=m0)
            total_ls_iter += ls_iter
            theta_next = theta + alpha * d
            self.set_x_tf32(tf.convert_to_tensor(theta_next))
            f_next_total = self.loss()  # updates caches
            g_next = self.true_grad()
            s = (theta_next - theta).astype(np.float64);
            y = (g_next - g).astype(np.float64)
            y_bar, theta_mix, sTy, sTyb, _ = powell_damp_pair(s, y, cfg.powell_c, gamma)
            cos_sy = float(sTyb / (np.linalg.norm(s) * (np.linalg.norm(y_bar) + 1e-18) + 1e-18))
            pair_accepted = False
            if sTyb > 1e-12 and cos_sy >= cfg.curvature_cos_tol:
                self.mem.push(s, y_bar);
                pair_accepted = True
                gamma = max(float(sTyb / (np.dot(y_bar, y_bar) + 1e-18)), 1e-8)
                if cfg.adam_diagonal: v, d0 = update_H0_from_grad(g, v, beta2, eps, gamma)
            theta, f0_total, g = theta_next, f_next_total, g_next
            t_now = time.perf_counter() - t0
            sqrt_mse = math.sqrt(self._last_mse_val)
            rec = dict(iter=it, t=t_now, sqrt_mse=sqrt_mse, pen=self._last_pen_val,
                       total=f_next_total, alpha=alpha, armijo=accepted,
                       g_norm=float(np.linalg.norm(g)), d_dot_g=float(np.dot(d, g)),
                       sTy=sTy, sTy_bar=sTyb, cos_sy=cos_sy, s_norm=float(np.linalg.norm(s)),
                       ybar_norm=float(np.linalg.norm(y_bar)), mem=len(self.mem.S), gamma=gamma,
                       ls_trace=ls_trace)
            history.append(rec)
            if it % 10 == 0:
                # Preserve your print, but split 'loss' into sqrt_mse + lam*pen
                print(
                    f"Iter {it}: t={t_now:7.3f}s, , angle={angle_pi:.4f}, "
                    f"loss={sqrt_mse:.5f}+lam*{self._last_pen_val:.5f}, d.g>0:{m0 > 0}, armijo:{accepted}, "
                    f"alpha={alpha:.3e}, iter={total_ls_iter}, |g|={rec['g_norm']:.3e}, "
                    f"neg_sTy={sTy < 0}, powel={theta_mix != 1.0}, pair ok:{pair_accepted}, mem={len(self.mem.S)}")
        return history


# ==========================
# Ω display helpers
# ==========================
class ThetaMeanPredictor(tf.keras.Model):
    def __init__(self, g_model: tf.keras.Model, X: tf.Tensor, Y: tf.Tensor):
        super().__init__()
        XT = tf.concat([X, Y], axis=1);
        theta_all = g_model(XT, training=False)
        self.theta_bar = tf.reduce_mean(theta_all, axis=0)

    def call(self, X, training=False):
        return f_forward_single_theta(X, self.theta_bar, ARCH_F)


class OmegaPerSampleDisplay(tf.keras.Model):
    """For plotting when Ω was trained per-sample: use true T on slice grid."""

    def __init__(self, g_model: tf.keras.Model, a: float, b: float):
        super().__init__();
        self.g = g_model;
        self.a = float(a);
        self.b = float(b)

    def call(self, X, training=False):
        t = tf.reshape(X[:, 0], [-1, 1]);
        x = tf.reshape(X[:, 1], [-1, 1]);
        s = tf.reshape(X[:, 2], [-1, 1])
        z = (x - self.a) / (self.b * s);
        T_true = std_normal_cdf(z)
        XT = tf.concat([t, x, s, T_true], axis=1)
        theta_all = self.g(XT, training=False)
        return f_forward_per_sample_theta(X, theta_all, ARCH_F)


# ==========================
# Main
# ==========================
def demo():
    seed = 1
    random.seed(seed);
    np.random.seed(seed);
    tf.random.set_seed(seed)

    # Config
    N = int(32768 / 8)  # your default
    A_PARAM = 0.0;
    B_PARAM = 1.0
    SHOW_PLOTS = True

    X, Y = make_dataset(N, seed=1, a=A_PARAM, b=B_PARAM)

    model_a = build_mlp();
    model_b = build_mlp()
    model_b.set_weights([w.copy() for w in model_a.get_weights()])  # identical init

    I = 2000

    cfg_a = LBFGSConfig(max_iters=I, mem=20, c1=1e-4, powell_c=0.2, ls_max_steps=10,
                        alpha_init=1.0, cub_clip=(0.1, 2.5), quad_clip=(0.1, 2.5),
                        sigma_es=1e-3, sigma_decay=0.4, sigma_min=1e-8,
                        curvature_cos_tol=1e-1, diag_true_grad=False, adam_diagonal=False)

    cfg_b = LBFGSConfig(max_iters=I, mem=20, c1=1e-4, powell_c=0.2, ls_max_steps=10,
                        alpha_init=1.0, cub_clip=(0.1, 2.5), quad_clip=(0.1, 2.5),
                        sigma_es=1e-3, sigma_decay=0.4, sigma_min=1e-8,
                        curvature_cos_tol=1e-1, diag_true_grad=False, adam_diagonal=True)

    skip_A = True
    if not skip_A:
        print("\n=== Mode A: L-BFGS (θ, Armijo cubic) ===")
        runner_a = LBFGSRunner(model_a, X, Y, cfg_a, seed=123)
        hist_a = runner_a.run_autodiff()

    print("\n=== Mode B: L-BFGS (θ, Armijo cubic + Adam diag)===")
    runner_b = LBFGSRunner(model_b, X, Y, cfg_b, seed=123)
    hist_b = runner_b.run_autodiff()

    # ===== Ω training (g -> θ) =====
    P = count_f_params(ARCH_F)
    lam = 0  # 1e-6  # same λ used for either penalty

    # Build four g models (same init pairs for fairness)
    g_per_jac = build_g(P)
    g_mean_jac = build_g(P);
    g_mean_jac.set_weights([w.copy() for w in g_per_jac.get_weights()])
    g_per_var = build_g(P)
    g_mean_var = build_g(P);
    g_mean_var.set_weights([w.copy() for w in g_per_var.get_weights()])

    cfg_omega = LBFGSConfig(max_iters=I, mem=20, c1=1e-4, powell_c=0.2, ls_max_steps=10,
                            alpha_init=1.0, cub_clip=(0.1, 2.5), quad_clip=(0.1, 2.5),
                            sigma_es=1e-3, sigma_decay=0.4, sigma_min=1e-8,
                            curvature_cos_tol=1e-1, diag_true_grad=False, adam_diagonal=False)

    print("\n=== Ω1: per-sample θ + Jac penalty (Hutch) ===")
    lam=0 # 1e-6
    runner_O1 = LBFGSRunnerOmega(g_per_jac, X, Y, cfg_omega, lam=lam,
                                 per_sample_theta=True, penalty_type='jac',
                                 jac_mode='hutch', hutch_probes=1, penalty_subsample=1024, seed=123)
    hist_O1 = runner_O1.run_autodiff()

    #print("\n=== Ω2: mean θ + Jac penalty (Hutch) ===")
    #runner_O2 = LBFGSRunnerOmega(g_mean_jac, X, Y, cfg_omega, lam=lam,
    #                             per_sample_theta=False, penalty_type='jac',
    #                             jac_mode='hutch', hutch_probes=1, penalty_subsample=1024, seed=123)
    #hist_O2 = runner_O2.run_autodiff()


    print("\n=== Ω3: per-sample θ + Var(θ) penalty ===")
    lam = 0# 1e-2
    runner_O3 = LBFGSRunnerOmega(g_per_var, X, Y, cfg_omega, lam=lam,
                                 per_sample_theta=True, penalty_type='var', seed=123)
    hist_O3 = runner_O3.run_autodiff()

    #print("\n=== Ω4: mean θ + Var(θ) penalty ===")
    #runner_O4 = LBFGSRunnerOmega(g_mean_var, X, Y, cfg_omega, lam=lam,
    #                             per_sample_theta=False, penalty_type='var', seed=123)
    #hist_O4 = runner_O4.run_autodiff()

    # Build predictors for plotting
    pred_O1 = OmegaPerSampleDisplay(g_per_jac, a=A_PARAM, b=B_PARAM)  # per-sample
    #pred_O2 = ThetaMeanPredictor(g_mean_jac, X, Y)  # mean-θ
    pred_O3 = OmegaPerSampleDisplay(g_per_var, a=A_PARAM, b=B_PARAM)  # per-sample
    #pred_O4 = ThetaMeanPredictor(g_mean_var, X, Y)  # mean-θ

    # Report final θ-trained results
    if not skip_A: tA, lA = hist_a[-1]['t'], hist_a[-1]['loss']
    tB, lB = hist_b[-1]['t'], hist_b[-1]['loss']
    print("\nFinal results (θ-trained):")
    if not skip_A: print(f"Mode A: loss={lA:.6f}, wall_time={tA:.3f}s")
    print(f"Mode B: loss={lB:.6f}, wall_time={tB:.3f}s")

    # Plain MSE on training set for Ω models (apples-to-apples)
    mse_O1 = float(mse_loss(pred_O1, X, Y).numpy())
    #mse_O2 = float(mse_loss(pred_O2, X, Y).numpy())
    mse_O3 = float(mse_loss(pred_O3, X, Y).numpy())
    #mse_O4 = float(mse_loss(pred_O4, X, Y).numpy())

    print("\nFinal results (Ω-trained) — plain √MSE on train:")
    print(f"Ω1 per+Jac:  √MSE={math.sqrt(mse_O1):.6f}")
    #print(f"Ω2 mean+Jac: √MSE={math.sqrt(mse_O2):.6f}")
    print(f"Ω3 per+Var:  √MSE={math.sqrt(mse_O3):.6f}")
    #print(f"Ω4 mean+Var: √MSE={math.sqrt(mse_O4):.6f}")

    if SHOW_PLOTS:
        # θ plots
        if skip_A:
            plot_convergence_AB(hist_b, hist_b, skip_a=True)
        else:
            plot_convergence_AB(hist_a, hist_b, skip_a=False)

        # Ω dual-axis plots
        plot_convergence_dual(hist_O1, title='Ω1 per-sample θ + Jac')
        plot_convergence_dual(hist_O2, title='Ω2 mean θ + Jac')
        plot_convergence_dual(hist_O3, title='Ω3 per-sample θ + Var(θ)')
        plot_convergence_dual(hist_O4, title='Ω4 mean θ + Var(θ)')

        # Slice comparisons (e.g., compare θ-trained A with Ω2 mean-θ)
        ref_A = model_b if skip_A else model_a
        plot_slices(ref_A, pred_O2, a=A_PARAM, b=B_PARAM, skip_a=False)
        plt.show()

demo()