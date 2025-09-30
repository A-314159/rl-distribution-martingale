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
    if not skip_a: plt.plot(ta, la, label='A: L-BFGS')
    plt.plot(tb, lb, label='B: L-BFGS with Adam diagonal')
    plt.xlabel('Wall time (s)')
    plt.ylabel('√ mse')
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


def update_H0_from_grad(g, v, beta2, eps, target_med):
    v = beta2 * v + (1 - beta2) * (g * g)
    v_hat = v  # bias-correction optional for large t
    d0 = 1.0 / (np.sqrt(v_hat) + eps)
    # normalize to match scalar scale
    med = np.median(d0)
    if med > 0:
        d0 *= (target_med / med)
    # clip for safety
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
# Model
# ==========================

def build_mlp() -> tf.keras.Model:
    inp = tf.keras.Input(shape=(3,), dtype=tf.float32)
    h = tf.keras.layers.Dense(16, activation='tanh')(inp)
    h = tf.keras.layers.Dense(16, activation='tanh')(h)
    h = tf.keras.layers.Dense(16, activation='tanh')(h)
    #h = tf.keras.layers.Dense(16, activation='tanh')(h)
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
        self.S.clear()
        self.Y.clear()

    def push(self, s: np.ndarray, y: np.ndarray):
        self.S.append(s.astype(np.float64))
        self.Y.append(y.astype(np.float64))
        if len(self.S) > self.m:
            self.S.pop(0)
            self.Y.pop(0)

    def two_loop(self, g: np.ndarray, gamma: float, d0=None) -> np.ndarray:
        S, Y = self.S, self.Y
        q = g.astype(np.float64).copy()
        alpha, rho = [], []
        for s, y in zip(reversed(S), reversed(Y)):
            r = 1.0 / (np.dot(y, s) + 1e-18)
            rho.append(r)
            a = r * np.dot(s, q)
            alpha.append(a)
            q -= a * y
        if d0 is None:
            q *= gamma
        else:
            q *= d0
        for (s, y, r, a) in zip(S, Y, reversed(rho), reversed(alpha)):
            b = r * np.dot(y, q)
            q += s * (a - b)
        return q


# ==========================
# Powell damping
# ==========================

def powell_damp_pair(s: np.ndarray, y: np.ndarray, c: float, gamma: float) -> Tuple[
    np.ndarray, float, float, float, bool]:
    """Return y_bar and theta (mixing factor)."""
    ss, yy = np.dot(s, s), np.dot(y, y)
    sBs = (1.0 / max(gamma, 1e-16)) * ss
    sy = float(np.dot(s, y))
    # flip y if neg curvature
    if sy < 0:
        y = -y
    asy = abs(sy)

    # floor sy if tiny
    fl = ss * yy * 1e-9
    tiny = False
    if asy < fl:
        tiny = True
        a = fl / yy
        y = a * y
        asy = fl

    if asy >= c * sBs:
        return y, 1.0, sy, asy, tiny
    theta = (1.0 - c) * sBs / max(sBs - asy, 1e-16)
    y_bar = theta * y + (1.0 - theta) * (1.0 / max(gamma, 1e-16)) * s
    return y_bar, theta, sy, float(np.dot(s, y_bar)), tiny


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


def fd_slope_along(f_at, x, d, h=None):
    """
    Approximate φ'(0) where φ(α) = f(x + α d).
    Uses symmetric finite difference: (f(x+h d) - f(x-h d)) / (2h).

    f_at : callable taking a flat θ and returning a scalar loss
    x    : np.ndarray (flat params)
    d    : np.ndarray (search direction)
    h    : optional scalar step in α-space (not along unit d; along d itself)
    """
    nd = float(np.linalg.norm(d))
    if nd == 0.0:
        return 0.0

    # Choose a conservative h so that the actual perturbation ||h d|| is small.
    # Works well in practice:
    if h is None:
        # target absolute parameter step ~ 1e-3 relative to x scale
        target_step = 1e-3 * (1.0 + float(np.linalg.norm(x)) / (1.0 + x.size))
        h = min(1e-3, target_step / nd)

    f_plus = f_at(x + h * d)
    f_minus = f_at(x - h * d)
    return float((f_plus - f_minus) / (2.0 * h))


# ==========================
# y
# ==========================
def dg(f: Callable[[np.ndarray], float], x_np64: np.ndarray, x_next_np64, f_cur: float, f_next: float, sigma: float,
       rng: np.random.Generator, normal=False, one_side=False, ES=True, K: int = 1) -> \
        np.ndarray:
    d = int(x_np64.shape[0])
    g_next = np.zeros(d, dtype=np.float64)
    g = np.zeros(d, dtype=np.float64)
    sig = sigma if one_side else 2 * sigma
    if not ES: sig = 1.0 / sig

    for k in range(K):
        eps_np64 = rng.normal(0, 1, size=d) if normal else rng.integers(0, 2, size=d).astype(np.int8) * 2 - 1  # ±1
        eps_np64 = eps_np64.astype(np.float64)
        if one_side:
            df_next = f(x_next_np64 + sigma * eps_np64) - f_next
            df = f_cur - f(x_np64 - sigma * eps_np64)
        else:
            df_next = f(x_next_np64 + sigma * eps_np64) - f(x_next_np64 - sigma * eps_np64)
            df = f(x_np64 + sigma * eps_np64) - f(x_np64 - sigma * eps_np64)
        if ES:
            g_next += df_next * eps_np64
            g += df * eps_np64
        else:
            g_next += np.divide(df_next, eps_np64)
            g += np.divide(df, eps_np64)
    return (g_next - g) / K / sig


# ==========================
# ES gradient
# ==========================
def f_g(f: Callable[[np.ndarray], float], x_np64: np.ndarray, sigma: float,
        rng: np.random.Generator, normal=False, one_side=False, ES=True, K=1) -> Tuple[float, np.ndarray]:
    n = int(x_np64.shape[0])
    g_np64 = np.zeros(n, np.float64)
    f_mid = f(x_np64)
    for k in range(K):
        esp_np64 = rng.normal(0, 1, size=n) if normal else rng.integers(0, 2, size=n).astype(np.int8) * 2 - 1  # ±1
        esp_np64 = esp_np64.astype(np.float64)
        f_plus = f(x_np64 + sigma * esp_np64)

        if not one_side:
            f_minus = f(x_np64 - sigma * esp_np64)
            d = 2 * sigma
        else:
            f_minus = f_mid
            d = sigma
        if ES:
            g_np64 += (f_plus - f_minus) / d * esp_np64
        else:
            g_np64 += np.divide(f_plus - f_minus, d * esp_np64)

    return f_mid, g_np64 / K


def should_accept_pair(s: np.ndarray,
                       y_bar: np.ndarray,
                       gamma: float,
                       c: float,
                       eps_curv: float = 1e-12,
                       cos_tol_relaxed: float = -0.10,  # <= 0 means very permissive
                       relax_factor: float = 0.5) -> bool:
    """
    Accept if the *damped* pair has enough positive curvature vs the Powell floor.
    Returns (accepted, sTyb, curv_floor, cos_sy).
    """
    s = s.astype(np.float64, copy=False)
    y_bar = y_bar.astype(np.float64, copy=False)
    sTyb = float(np.dot(s, y_bar))
    if sTyb <= eps_curv:
        return False  # , sTyb, 0.0, 0.0

    s_norm = float(np.linalg.norm(s)) + 1e-18
    yb_norm = float(np.linalg.norm(y_bar)) + 1e-18
    cos_sy = sTyb / (s_norm * yb_norm)

    # Powell curvature floor (using B0 ≈ gamma^{-1} I)
    curv_floor = c * (np.dot(s, s) / max(gamma, 1e-16))

    # Accept if we’re at least a relaxed fraction of the floor, and the cosine isn’t terrible
    if sTyb >= relax_factor * curv_floor and cos_sy >= cos_tol_relaxed:
        return True  #, sTyb, curv_floor, cos_sy
    return False  #, sTyb, curv_floor, cos_sy


# ==========================
# Config & Runner
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
    sigma_decay: float = 0.4  # σ_k = max(σ_min, σ0 / (1+k)^decay)
    sigma_min: float = 1e-8
    curvature_cos_tol: float = 1e-6  # accept pair only if cos(s,y_bar) >= tol
    diag_true_grad: bool = False  # compute true grad for diagnostics only
    adam_diagonal: bool = False


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
    def get_x_tf64(self) -> tf.Tensor:
        return tf.cast(pack_variables(self.vars), tf.float64)

    def set_x_tf32(self, x_np64: np.ndarray) -> None:
        unpack_to_variables(tf.cast(tf.convert_to_tensor(x_np64), tf.float32), self.vars)

    @tf.function(jit_compile=False)
    def _tf_loss(self) -> tf.Tensor:
        return mse_loss(self.model, self.X, self.Y)

    def loss(self) -> float:
        return float(self._tf_loss().numpy())

    def loss_at_x(self, x_np64: np.ndarray) -> float:
        self.set_x_tf32(x_np64)
        return self.loss()

    def true_grad(self) -> np.ndarray:
        with tf.GradientTape() as tape:
            loss = self._tf_loss()
        grads = tape.gradient(loss, self.vars)
        flat = tf.concat([tf.reshape(g, [-1]) for g in grads], axis=0)
        return flat.numpy().astype(np.float64)

    # ----- Direction -----
    def direction(self, g: np.ndarray, gamma: float, d0=None) -> np.ndarray:
        if len(self.mem.S) == 0:
            if d0 is None:
                return -gamma * g
            else:
                return -d0 * g
        return -self.mem.two_loop(g, gamma, d0)

    # ----- Line search (shared) -----
    def line_search(self, theta: np.ndarray, f0: float, g: np.ndarray, d: np.ndarray,
                    mode: str, m0: Optional[float] = None) -> Tuple[float, bool, List[Tuple[float, float, bool]], int]:
        cfg = self.cfg
        if m0 is None:
            m0 = float(np.dot(g, d))
        if m0 >= 0:
            # fallback to steepest descent if needed
            d = -g.copy()
            m0 = float(np.dot(g, d))
        alpha = cfg.alpha_init
        a_prev, f_prev = None, None
        best_f, best_a = f0, 0
        trace = []
        accepted = False
        count = 0
        for i in range(cfg.ls_max_steps):
            count += 1
            theta_trial = theta + alpha * d
            self.set_x_tf32(tf.convert_to_tensor(theta_trial))
            f_a = self.loss()
            ok = armijo_condition(f0, m0, f_a, alpha, cfg.c1)
            trace.append((alpha, f_a, ok))
            if f_a < best_f:
                best_f, best_a = f_a, alpha
            if ok:
                # new
                if best_f < f0:
                    alpha = best_a  # When armijo is met, we take the best alpha even if it was such that armijo was not met
                accepted = True
                break

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
        return float(alpha), accepted, trace, count

    # ----- Runs -----
    def run_autodiff(self) -> List[Dict[str, Any]]:
        cfg = self.cfg
        history: List[Dict[str, Any]] = []
        t0 = time.perf_counter()

        theta = self.get_x_tf64().numpy()
        f0 = self.loss()
        g = self.true_grad()
        gamma = 1.0
        print(f"Iter 0: loss={np.sqrt(f0):.6f}")
        history.append(dict(iter=0, t=0.0, loss=f0))
        d_prev = None

        # state
        v = np.zeros_like(theta)  # EMA of g^2
        d0 = None
        beta2, eps = 0.999, 1e-8

        if cfg.adam_diagonal:        v, d0 = update_H0_from_grad(g, v, beta2, eps, gamma)
        total_ls_iter = 0
        for it in range(1, cfg.max_iters + 1):
            d = self.direction(g, gamma, d0)
            angle_pi = float('nan')
            if d_prev is not None:
                nd = np.linalg.norm(d)
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
            f_next = self.loss()
            g_next = self.true_grad()

            s = (theta_next - theta).astype(np.float64)
            y = (g_next - g).astype(np.float64)

            y_bar, theta_mix, sTy, sTyb, _ = powell_damp_pair(s, y, cfg.powell_c, gamma)
            cos_sy = float(sTyb / (np.linalg.norm(s) * (np.linalg.norm(y_bar) + 1e-18) + 1e-18))

            pair_accepted = False
            if sTyb > 1e-12 and cos_sy >= cfg.curvature_cos_tol:
                self.mem.push(s, y_bar)
                pair_accepted = True
                gamma = max(float(sTyb / (np.dot(y_bar, y_bar) + 1e-18)), 1e-8)
                if cfg.adam_diagonal:        v, d0 = update_H0_from_grad(g, v, beta2, eps, gamma)

            theta, f0, g = theta_next, f_next, g_next
            t_now = time.perf_counter() - t0
            sqrt_f = math.sqrt(f_next)

            rec = dict(iter=it, t=t_now, loss=sqrt_f, alpha=alpha, armijo=accepted,
                       g_norm=float(np.linalg.norm(g)), d_dot_g=float(np.dot(d, g)),
                       sTy=sTy, sTy_bar=sTyb, cos_sy=cos_sy, s_norm=float(np.linalg.norm(s)),
                       ybar_norm=float(np.linalg.norm(y_bar)), mem=len(self.mem.S), gamma=gamma,
                       ls_trace=ls_trace)
            history.append(rec)
            if it % 1 == 0:
                print(
                    f"Iter {it}: t={t_now:7.3f}s, , angle={angle_pi:.5f}, loss={sqrt_f:.6f}, d.g>0:{m0 > 0}, armijo:{accepted}, alpha={alpha:.4e}, iter={total_ls_iter}, |g|={rec['g_norm']:.4e}, neg_sTy={sTy < 0}, powel={theta_mix != 1.0}, pair ok:{pair_accepted}, mem={len(self.mem.S)}")

            #if np.linalg.norm(g) < 1e-6:
            #    break
        return history

    def run_es(self) -> List[Dict[str, Any]]:
        cfg = self.cfg
        history: List[Dict[str, Any]] = []
        t0 = time.perf_counter()

        sigma0 = cfg.sigma_es
        sigma = sigma0

        normal = False
        ES = True
        one_side = False

        K_g = 256
        K_y = 256
        x_np64 = self.get_x_tf64().numpy()
        gamma = 1.0

        if not ES: normal = False
        f, g = f_g(lambda th: self.loss_at_x(th), x_np64, sigma, self.rng, normal=normal, one_side=one_side, ES=ES,
                   K=K_g)

        print(f"Iter 0: loss={f:.6f}")
        history.append(dict(iter=0, t=0.0, loss=f))
        d_prev = None
        for it in range(1, cfg.max_iters + 1):

            d = self.direction(g, gamma)

            # Compute angle of direction change
            angle_pi = float('nan')
            if d_prev is not None:
                nd = np.linalg.norm(d)
                ndp = np.linalg.norm(d_prev)
                if nd > 0 and ndp > 0:
                    c = np.clip(np.dot(d_prev, d) / (ndp * nd), -1.0, 1.0)
                    angle_pi = float(np.arccos(c) / np.pi)
            d_prev = d.copy()

            m0 = float(np.dot(g, d))
            #m0 = fd_slope_along(lambda x: self.loss_at_x(x), x_np64, d)
            alpha, accepted, ls_trace, ls_iter = self.line_search(x_np64, f, g, d, mode='es', m0=m0)
            if not accepted:
                g = self.true_grad()
                #self.mem.clear()
                print('true gradient')
                continue

            x_next_np64 = x_np64 + alpha * d

            # Anneal σ from time to time and compute gradient at next theta
            sigma_next = max(cfg.sigma_min, sigma0 / ((1.0 + it) ** cfg.sigma_decay)) if it % 10 == 0 else sigma
            f_next, g = f_g(lambda x: self.loss_at_x(x), x_next_np64, sigma_next, self.rng, normal=normal,
                            one_side=one_side, ES=ES, K=K_y)

            # Compute s and y using the current sigma
            s = x_next_np64 - x_np64
            y = dg(lambda x: self.loss_at_x(x), x_np64, x_next_np64, f, f_next, sigma, self.rng, normal=normal,
                   one_side=one_side, ES=ES, K=K_g)

            s_norm = float(np.linalg.norm(s)) + 1e-18
            y_norm = float(np.linalg.norm(y)) + 1e-18
            cos_raw = float(np.dot(s, y)) / (s_norm * y_norm)
            #ratio=np.where(y!=0,np.divide(s,y),0)

            # Powell damping and acceptance
            y_bar, theta_mix, sTy, sTyb, tiny = powell_damp_pair(s, y, cfg.powell_c, gamma)
            cos_sy = float(sTyb / (np.linalg.norm(s) * (np.linalg.norm(y_bar) + 1e-18) + 1e-18))
            over_parallel = cos_raw > 0.95

            new_acceptance = True
            pair_accepted = False
            if not new_acceptance:
                if sTyb > 1e-12 and cos_sy >= cfg.curvature_cos_tol:
                    self.mem.push(s, y_bar)
                    pair_accepted = True
                    gamma = max(float(sTyb / (np.dot(y_bar, y_bar) + 1e-18)), 1e-8)
            else:
                pair_accepted = should_accept_pair(s, y_bar, gamma, cfg.powell_c) and not over_parallel
                if pair_accepted:
                    self.mem.push(s, y_bar)
                    gamma = max(float(sTyb / (np.dot(y_bar, y_bar) + 1e-18)), 1e-8)
            d_gd_like = d + gamma * g  # or g for autodiff mode
            gd_like_ratio = float(np.linalg.norm(d_gd_like) / (np.linalg.norm(gamma * g) + 1e-18))

            x_np64, f, sigma = x_next_np64, f_next, sigma_next
            t_now = time.perf_counter() - t0

            rec = dict(iter=it, t=t_now, loss=f_next, alpha=alpha, armijo=accepted,
                       g_es_norm=float(np.linalg.norm(g)), m0=m0, sigma=sigma,
                       sTy=sTy, sTy_bar=sTyb, cos_sy=cos_sy, s_norm=float(np.linalg.norm(s)),
                       ybar_norm=float(np.linalg.norm(y_bar)), mem=len(self.mem.S), gamma=gamma,
                       ls_trace=ls_trace)
            history.append(rec)
            if it % 1 == 0:
                print(
                    f"Iter {it}: t={t_now:7.3f}s, , angle={angle_pi:.4f}, loss={f_next:.6f}, d.g>0:{m0 > 0}, gd_like={gd_like_ratio:.3e}, cos_raw={cos_raw:.3f}, over_par={over_parallel},armijo:{accepted}, alpha={alpha:.3e}, iter={ls_iter}, sigma={sigma:.3e}, |g|={rec['g_es_norm']:.3e}, neg_sTy={sTy < 0}, TINY={tiny}, powel={theta_mix != 1.0}, cos(sy)={cos_sy:.3e}, pair ok:{pair_accepted}, mem={len(self.mem.S)}")

        return history


# ==========================
# Main
# ==========================
def demo():
    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Config
    N = int(32768 / 8)
    A_PARAM = 0.0
    B_PARAM = 1.0
    SHOW_PLOTS = True
    ITER = 200

    X, Y = make_dataset(N, seed=seed, a=A_PARAM, b=B_PARAM)

    model_a = build_mlp()
    model_b = build_mlp()

    # First 2 rows of data
    print("X[:2] =", X[:2].numpy())
    print("Y[:2] =", Y[:2].numpy())

    # A tiny weight fingerprint (sum and first 5 scalars)
    w0 = tf.concat([tf.reshape(v, [-1]) for v in model_a.weights], axis=0)
    print("w-sum =", float(tf.reduce_sum(w0)))
    print("w[:5] =", w0[:5].numpy())

    # Ensure identical initialization
    model_b.set_weights([w.copy() for w in model_a.get_weights()])

    cfg_a = LBFGSConfig(max_iters=ITER, mem=20, c1=1e-4, powell_c=0.2, ls_max_steps=10,
                        alpha_init=1.0, cub_clip=(0.1, 2.5), quad_clip=(0.1, 2.5),
                        sigma_es=1e-3, sigma_decay=0.4, sigma_min=1e-8,
                        curvature_cos_tol=1e-1, diag_true_grad=False, adam_diagonal=False)

    cfg_b = LBFGSConfig(max_iters=ITER, mem=20, c1=1e-4, powell_c=0.2, ls_max_steps=10,
                        alpha_init=1.0, cub_clip=(0.1, 2.5), quad_clip=(0.1, 2.5),
                        sigma_es=1e-3, sigma_decay=0.4, sigma_min=1e-8,
                        curvature_cos_tol=1e-1, diag_true_grad=False, adam_diagonal=True)

    skip_auto_diff = False
    if not skip_auto_diff:
        print("\n=== Mode A: L-BFGS (Armijo cubic) ===")
        runner_a = LBFGSRunner(model_a, X, Y, cfg_a, seed=123)
        hist_a = runner_a.run_autodiff()
        #hist_a = runner_a.run_es()

    print("\n=== Mode B: L-FBGS (Armijo cubic, with Adam diagonal)===")
    runner_b = LBFGSRunner(model_b, X, Y, cfg_b, seed=123)
    #hist_b = runner_b.run_es()
    hist_b = runner_b.run_autodiff()

    if not skip_auto_diff: tA, lA = hist_a[-1]['t'], hist_a[-1]['loss']
    tB, lB = hist_b[-1]['t'], hist_b[-1]['loss']
    print("\nFinal results:")
    if not skip_auto_diff: print(f"Mode A: loss={lA:.6f}, wall_time={tA:.3f}s")
    print(f"Mode B: loss={lB:.6f}, wall_time={tB:.3f}s")

    if SHOW_PLOTS:
        if skip_auto_diff:
            hist_a = hist_b
        plot_convergence(hist_a, hist_b, skip_auto_diff)
        plot_slices(model_a, model_b, a=A_PARAM, b=B_PARAM, skip_a=skip_auto_diff)
        plt.show()


run = True
if run: demo()
