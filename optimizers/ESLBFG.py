#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clean L-BFGS (Powell-damped) for a small TF MLP in two modes:
  (A) Autodiff gradient + cubic Armijo line search
  (B) ES gradient (antithetic Rademacher, CRN only within k/k+1) + quadratic function-only Armijo

Adds rich per-iteration diagnostics and returns full `history` for both modes.

Target: y = Φ((x - a) / (b * sqrt(1 - t))).
Inputs: (t, x, sqrt(1-t)), t~U(0,1), x~U(-4,4). Network: (3,16,16,16,1), tanh, sigmoid.

Tested with: Python 3.9+, TensorFlow 2.15+.
"""
import time
import math
import dataclasses
from typing import List, Tuple, Callable, Optional, Dict, Any

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os, random


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
# Model
# ==========================

def build_mlp() -> tf.keras.Model:
    inp = tf.keras.Input(shape=(3,), dtype=tf.float32)
    h = tf.keras.layers.Dense(16, activation='tanh')(inp)
    h = tf.keras.layers.Dense(16, activation='tanh')(h)
    h = tf.keras.layers.Dense(16, activation='tanh')(h)
    out = tf.keras.layers.Dense(1, activation='sigmoid')(h)
    return tf.keras.Model(inputs=inp, outputs=out)


# ==========================
# Flatten/pack parameters
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
# Loss and helpers
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
        self.S.append(s.astype(np.float64))
        self.Y.append(y.astype(np.float64))
        if len(self.S) > self.m:
            self.S.pop(0);
            self.Y.pop(0)

    def two_loop(self, g: np.ndarray, gamma: float) -> np.ndarray:
        S, Y = self.S, self.Y
        q = g.astype(np.float64).copy()
        alpha, rho = [], []
        for s, y in zip(reversed(S), reversed(Y)):
            r = 1.0 / (np.dot(y, s) + 1e-18)
            rho.append(r)
            a = r * np.dot(s, q)
            alpha.append(a)
            q -= a * y
        q *= gamma
        for (s, y, r, a) in zip(S, Y, reversed(rho), reversed(alpha)):
            b = r * np.dot(y, q)
            q += s * (a - b)
        return q


# ==========================
# Powell damping
# ==========================

def powell_damp_pair(s: np.ndarray, y: np.ndarray, c: float, gamma: float) -> Tuple[np.ndarray, float]:
    """Return y_bar and theta (mixing factor)."""
    sBs = (1.0 / max(gamma, 1e-16)) * np.dot(s, s)
    sy = float(np.dot(s, y))
    if sy >= c * sBs:
        return y, 1.0
    theta = (1.0 - c) * sBs / max(sBs - sy, 1e-16)
    y_bar = theta * y + (1.0 - theta) * (1.0 / max(gamma, 1e-16)) * s
    return y_bar, theta


# ==========================
# Line searches
# ==========================

def armijo_condition(f0: float, m0: float, f_a: float, a: float, c1: float) -> bool:
    return f_a <= f0 + c1 * a * m0


def cubic_step(f0, m0, a_prev, f_prev, a, f_a, low_mult, high_mult) -> float:
    """Cubic using f0,m0 at 0 and (a_prev,f_prev),(a,f_a)."""
    if a_prev is None or f_prev is None:
        return float(np.clip(0.5 * a, low_mult * a, high_mult * a))
    try:
        A = np.array([[a_prev ** 2, a_prev, 1.0],
                      [a ** 2, a, 1.0],
                      [0.0, 1.0, 0.0]], dtype=float)
        b = np.array([f_prev - f0, f_a - f0, m0], dtype=float)
        coef = np.linalg.lstsq(A, b, rcond=None)[0]
        if abs(coef[0]) > 1e-18:
            a_new = -coef[1] / (2.0 * coef[0])
        else:
            a_new = 0.5 * a
    except Exception:
        a_new = 0.5 * a
    a_new = float(np.clip(a_new, low_mult * a, high_mult * a))
    if not np.isfinite(a_new) or a_new <= 0:
        a_new = float(np.clip(0.5 * a, low_mult * a, high_mult * a))
    return a_new


def quadratic_step_function_only(a_prev, f_prev, a, f_a, low_mult, high_mult) -> float:
    if a_prev is None or f_prev is None:
        return float(np.clip(0.5 * a, low_mult * a, high_mult * a))
    denom = (a - a_prev)
    if abs(denom) < 1e-18:
        return float(np.clip(0.5 * a, low_mult * a, high_mult * a))
    slope_secant = (f_a - f_prev) / denom
    a_new = a - 0.5 * slope_secant * denom
    a_new = float(np.clip(a_new, low_mult * a, high_mult * a))
    if not np.isfinite(a_new) or a_new <= 0:
        a_new = float(np.clip(0.5 * a, low_mult * a, high_mult * a))
    return a_new


# ==========================
# ES gradient (antithetic Rademacher)
# ==========================

def es_grad_rademacher(f: Callable[[tf.Tensor], float], theta: tf.Tensor, sigma: float,
                       rng: np.random.Generator, eps: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Return g_hat, eps used. g_hat = ((f(θ+σ ε) - f(θ-σ ε))/(2σ)) * ε, ε_i∈{-1,+1}."""
    one_side=True

    d = int(theta.shape[0])
    if eps is None:
        eps = rng.integers(0, 2, size=d).astype(np.int8) * 2 - 1  # ±1
        eps = eps.astype(np.float64)
    eps_tf = tf.convert_to_tensor(eps, dtype=theta.dtype)
    f_plus = float(f(tf.cast(theta + sigma * eps_tf, theta.dtype)))
    f_minus = float(f(tf.cast(theta - sigma * eps_tf, theta.dtype)))
    coeff = (f_plus - f_minus) / (2.0 * sigma)
    g = coeff * eps
    return g.astype(np.float64), eps


# ==========================
# Config & Runner
# ==========================
@dataclasses.dataclass
class LBFGSConfig:
    max_iters: int = 200
    mem: int = 10
    c1: float = 1e-4
    powell_c: float = 0.2
    ls_max_steps: int = 15
    alpha_init: float = 1.0
    cub_clip: Tuple[float, float] = (0.1, 2.5)
    quad_clip: Tuple[float, float] = (0.1, 2.5)
    sigma_es: float = 1e-2
    sigma_decay: float = 0.4  # σ_k = max(σ_min, σ0 / (1+k)^decay)
    sigma_min: float = 1e-4
    curvature_cos_tol: float = 1e-6  # accept pair only if cos(s,y_bar) >= tol
    diag_true_grad: bool = False  # compute true grad for diagnostics only


class LBFGSRunner:
    def __init__(self, model: tf.keras.Model, X: tf.Tensor, Y: tf.Tensor, cfg: LBFGSConfig, seed: int = 0):
        self.model = model
        self.X = X
        self.Y = Y
        self.cfg = cfg
        self.mem = LBFGSMemory(m=cfg.mem)
        self.rng = np.random.default_rng(seed)
        self.vars = model.trainable_variables

    # ----- Parameter IO -----
    def get_theta(self) -> tf.Tensor:
        return tf.cast(pack_variables(self.vars), tf.float64)

    def set_theta(self, theta: tf.Tensor) -> None:
        unpack_to_variables(tf.cast(theta, tf.float32), self.vars)

    @tf.function(jit_compile=False)
    def _tf_loss(self) -> tf.Tensor:
        return mse_loss(self.model, self.X, self.Y)

    def loss_numpy(self) -> float:
        return float(self._tf_loss().numpy())

    def loss_at_theta(self, theta_flat: tf.Tensor) -> float:
        theta_save = self.get_theta()
        try:
            self.set_theta(theta_flat)
            return self.loss_numpy()
        finally:
            self.set_theta(theta_save)

    def true_grad(self) -> np.ndarray:
        with tf.GradientTape() as tape:
            loss = self._tf_loss()
        grads = tape.gradient(loss, self.vars)
        flat = tf.concat([tf.reshape(g, [-1]) for g in grads], axis=0)
        return flat.numpy().astype(np.float64)

    # ----- Direction -----
    def direction(self, g: np.ndarray, gamma: float) -> np.ndarray:
        if len(self.mem.S) == 0:
            return -gamma * g
        return -self.mem.two_loop(g, gamma)

    # ----- Line search (shared) -----
    def line_search(self, theta: np.ndarray, f0: float, g: np.ndarray, d: np.ndarray,
                    mode: str, m0: Optional[float] = None) -> Tuple[float, bool, List[Tuple[float, float, bool]]]:
        cfg = self.cfg
        if m0 is None:
            m0 = float(np.dot(g, d))
        if m0 >= 0:
            # fallback to steepest descent if needed
            d = -g.copy();
            m0 = float(np.dot(g, d))
        alpha = cfg.alpha_init
        a_prev, f_prev = None, None
        best_f, best_a = math.inf, None
        trace = []
        accepted = False
        for _ in range(cfg.ls_max_steps):
            theta_trial = theta + alpha * d
            self.set_theta(tf.convert_to_tensor(theta_trial))
            f_a = self.loss_numpy()
            ok = armijo_condition(f0, m0, f_a, alpha, cfg.c1)
            trace.append((alpha, f_a, ok))
            if f_a < best_f:
                best_f, best_a = f_a, alpha
            if ok:
                accepted = True
                break
            #alpha = cubic_step(f0, m0, a_prev, f_prev, alpha, f_a, *cfg.cub_clip)

            if mode == 'autodiff':
                alpha = cubic_step(f0, m0, a_prev, f_prev, alpha, f_a, *cfg.cub_clip)
            else:
                alpha = quadratic_step_function_only(a_prev, f_prev, alpha, f_a, *cfg.quad_clip)

            a_prev, f_prev = alpha, f_a
        if not accepted:
            if best_f < f0 and best_a is not None:
                alpha = best_a
                accepted = True  # accept best decrease even if Armijo failed
            else:
                # very small step to escape
                alpha = 1e-4 / (np.linalg.norm(d) + 1e-12)
        return float(alpha), accepted, trace

    # ----- Runs -----
    def run_autodiff(self) -> List[Dict[str, Any]]:
        cfg = self.cfg
        history: List[Dict[str, Any]] = []
        t0 = time.perf_counter()

        theta = self.get_theta().numpy()
        f0 = self.loss_numpy()
        g = self.true_grad()
        gamma = 1.0
        print(f"Iter 0: loss={f0:.6f}")
        history.append(dict(iter=0, t=0.0, loss=f0))

        for it in range(1, cfg.max_iters + 1):
            d = self.direction(g, gamma)
            m0 = float(np.dot(g, d))
            alpha, accepted, ls_trace = self.line_search(theta, f0, g, d, mode='autodiff', m0=m0)
            theta_next = theta + alpha * d
            self.set_theta(tf.convert_to_tensor(theta_next))
            f_next = self.loss_numpy()
            g_next = self.true_grad()

            s = (theta_next - theta).astype(np.float64)
            y = (g_next - g).astype(np.float64)
            y_bar, theta_mix = powell_damp_pair(s, y, cfg.powell_c, gamma)

            sTy = float(np.dot(s, y))
            sTyb = float(np.dot(s, y_bar))
            cos_sy = float(np.dot(s, y_bar) / (np.linalg.norm(s) * (np.linalg.norm(y_bar) + 1e-18) + 1e-18))

            if sTyb > 1e-12 and cos_sy >= cfg.curvature_cos_tol:
                self.mem.push(s, y_bar)
            if np.dot(y, y) > 0:
                gamma = max(float(np.dot(s, y) / np.dot(y, y)), 1e-8)

            theta, f0, g = theta_next, f_next, g_next
            t_now = time.perf_counter() - t0

            rec = dict(iter=it, t=t_now, loss=f_next, alpha=alpha, armijo=accepted,
                       g_norm=float(np.linalg.norm(g)), d_dot_g=float(np.dot(d, g)),
                       sTy=sTy, sTy_bar=sTyb, cos_sy=cos_sy, s_norm=float(np.linalg.norm(s)),
                       ybar_norm=float(np.linalg.norm(y_bar)), mem=len(self.mem.S), gamma=gamma,
                       ls_trace=ls_trace)
            history.append(rec)
            if it%10==0:
                print(
                    f"Iter {it}: t={t_now:7.3f}s, loss={f_next:.6f}, alpha={alpha:.3e}, |g|={rec['g_norm']:.3e}, mem={len(self.mem.S)}")
            if np.linalg.norm(g) < 1e-6:
                break
        return history

    def run_es(self) -> List[Dict[str, Any]]:
        cfg = self.cfg
        history: List[Dict[str, Any]] = []
        t0 = time.perf_counter()

        theta = self.get_theta().numpy()
        f0 = self.loss_numpy()
        sigma0 = cfg.sigma_es
        sigma = sigma0
        g_es, eps = es_grad_rademacher(lambda th: self.loss_at_theta(th),
                                       tf.convert_to_tensor(theta, tf.float64),
                                       sigma, self.rng, eps=None)
        gamma = 1.0
        print(f"Iter 0: loss={f0:.6f}")
        history.append(dict(iter=0, t=0.0, loss=f0))

        for it in range(1, cfg.max_iters + 1):
            # Anneal σ
            sigma = max(cfg.sigma_min, sigma0 / ((1.0 + it) ** cfg.sigma_decay))

            d = self.direction(g_es, gamma)
            m0 = float(np.dot(g_es, d))
            alpha, accepted, ls_trace = self.line_search(theta, f0, g_es, d, mode='es', m0=m0)
            theta_next = theta + alpha * d
            self.set_theta(tf.convert_to_tensor(theta_next))
            f_next = self.loss_numpy()

            # ES gradient at new point with SAME eps (CRN within step)
            g_es_next, _ = es_grad_rademacher(lambda th: self.loss_at_theta(th),
                                              tf.convert_to_tensor(theta_next, tf.float64),
                                              sigma, self.rng, eps=eps)

            # New eps for next iteration baseline
            g_es_baseline, eps_new = es_grad_rademacher(lambda th: self.loss_at_theta(th),
                                                        tf.convert_to_tensor(theta_next, tf.float64),
                                                        sigma, self.rng, eps=None)

            # Diagnostics: true grad if enabled
            if cfg.diag_true_grad:
                g_true = self.true_grad()
                g_true_norm = float(np.linalg.norm(g_true))
                d_dot_true = float(np.dot(d, g_true))
                g_err = float(np.linalg.norm(g_es_next - g_true))
            else:
                g_true_norm = float('nan');
                d_dot_true = float('nan');
                g_err = float('nan')

            s = (theta_next - theta).astype(np.float64)
            y = (g_es_next - g_es).astype(np.float64)
            sTy = float(np.dot(s, y))

            # Update H0 scaling
            if np.dot(y, y) > 0:
                gamma = max(float(np.dot(s, y) / np.dot(y, y)), 1e-8)

            # Powell damping and acceptance
            y_bar, theta_mix = powell_damp_pair(s, y, cfg.powell_c, gamma)
            sTyb = float(np.dot(s, y_bar))
            cos_sy = float(np.dot(s, y_bar) / (np.linalg.norm(s) * (np.linalg.norm(y_bar) + 1e-18) + 1e-18))
            if sTyb > 1e-12 and cos_sy >= cfg.curvature_cos_tol:
                self.mem.push(s, y_bar)

            theta, f0, g_es, eps = theta_next, f_next, g_es_baseline, eps_new
            t_now = time.perf_counter() - t0

            rec = dict(iter=it, t=t_now, loss=f_next, alpha=alpha, armijo=accepted,
                       g_es_norm=float(np.linalg.norm(g_es_baseline)), g_true_norm=g_true_norm,
                       g_err=g_err, d_dot_true=d_dot_true, m0=m0, sigma=sigma,
                       sTy=sTy, sTy_bar=sTyb, cos_sy=cos_sy, s_norm=float(np.linalg.norm(s)),
                       ybar_norm=float(np.linalg.norm(y_bar)), mem=len(self.mem.S), gamma=gamma,
                       ls_trace=ls_trace)
            history.append(rec)
            if it%10==0:
                print(
                    f"Iter {it}: t={t_now:7.3f}s, loss={f_next:.6f}, alpha={alpha:.3e}, |ES g|={rec['g_es_norm']:.3e}, |true g|={g_true_norm:.3e}, |err|={g_err:.3e}, mem={len(self.mem.S)}")

        return history


# ==========================
# Plot helpers
# ==========================

def plot_convergence(curve_a: List[Dict[str, Any]], curve_b: List[Dict[str, Any]], skip_a=True):
    def unpack(curve):
        ts = [r['t'] for r in curve]
        ls = [r['loss'] for r in curve]
        return ts, ls

    if not skip_a: ta, la = unpack(curve_a)
    tb, lb = unpack(curve_b)
    plt.figure()
    if not skip_a: plt.plot(ta, la, label='A: autodiff+cubic')
    plt.plot(tb, lb, label='B: ES+quadratic')
    plt.xlabel('Wall time (s)')
    plt.ylabel('MSE loss')
    plt.yscale('log')
    plt.title('Convergence')
    plt.legend();
    plt.grid(True, which='both', ls=':')


def plot_slices(model_a: tf.keras.Model, model_b: tf.keras.Model, a: float = 0.0, b: float = 1.0, skip_a=True):
    t_vals = [0.1, 0.5, 0.9]
    x = np.linspace(-4, 4, 401).astype(np.float32).reshape(-1, 1)
    plt.figure(figsize=(10, 7))
    for i, t0 in enumerate(t_vals, 1):
        t = np.full_like(x, t0, dtype=np.float32)
        s = np.sqrt(1.0 - t)
        X = np.concatenate([t, x, s], axis=1).astype(np.float32)
        if not skip_a: ya = model_a(X, training=False).numpy().reshape(-1)
        yb = model_b(X, training=False).numpy().reshape(-1)
        z = (x - a) / (b * s)
        y_true = std_normal_cdf(tf.convert_to_tensor(z)).numpy().reshape(-1)
        plt.subplot(3, 1, i)
        plt.plot(x.reshape(-1), y_true, label=f'True Φ, t={t0}')
        if not skip_a: plt.plot(x.reshape(-1), ya, label='Model A')
        plt.plot(x.reshape(-1), yb, label='Model B')
        plt.grid(True, ls=':');
        plt.legend()
    plt.suptitle('Predictions vs Φ at fixed t')


# ==========================
# Main
# ==========================
if __name__ == "__main__":

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Config
    N = int(32768/16)
    A_PARAM = 0.0
    B_PARAM = 1.0
    SHOW_PLOTS = True

    X, Y = make_dataset(N, seed=42, a=A_PARAM, b=B_PARAM)

    model_a = build_mlp()
    model_b = build_mlp()

    # Ensure identical initialization
    model_b.set_weights([w.copy() for w in model_a.get_weights()])

    cfg_a = LBFGSConfig(max_iters=200, mem=10, c1=1e-4, powell_c=0.2, ls_max_steps=15,
                      alpha_init=1.0, cub_clip=(0.1, 2.5), quad_clip=(0.1, 2.5),
                      sigma_es=1e-2, sigma_decay=0.4, sigma_min=1e-4,
                      curvature_cos_tol=1e-6, diag_true_grad=False)

    cfg_b = LBFGSConfig(max_iters=200, mem=10, c1=1e-4, powell_c=0.2, ls_max_steps=15,
                      alpha_init=1.0, cub_clip=(0.1, 2.5), quad_clip=(0.1, 2.5),
                      sigma_es=1e-2, sigma_decay=0.4, sigma_min=1e-4,
                      curvature_cos_tol=1e-6, diag_true_grad=False)

    skip_auto_diff = False
    if not skip_auto_diff:
        print("\n=== Mode A: Autodiff + Cubic Armijo ===")
        runner_a = LBFGSRunner(model_a, X, Y, cfg_a, seed=123)
        hist_a = runner_a.run_autodiff()
        #hist_a = runner_a.run_es()

    print("\n=== Mode B: ES (CRN) + Quadratic Armijo (function-only) ===")
    runner_b = LBFGSRunner(model_b, X, Y, cfg_b, seed=123)
    hist_b = runner_b.run_es()

    if not skip_auto_diff: tA, lA = hist_a[-1]['t'], hist_a[-1]['loss']
    tB, lB = hist_b[-1]['t'], hist_b[-1]['loss']
    print("\nFinal results:")
    if not skip_auto_diff: print(f"Mode A: loss={lA:.6f}, wall_time={tA:.3f}s")
    print(f"Mode B: loss={lB:.6f}, wall_time={tB:.3f}s")

    if SHOW_PLOTS:
        if skip_auto_diff:
            hist_a = hist_b
        plot_convergence(hist_a, hist_b,skip_auto_diff)
        plot_slices(model_a, model_b, a=A_PARAM, b=B_PARAM, skip_a=skip_auto_diff)
        plt.show()
