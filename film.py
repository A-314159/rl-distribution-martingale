#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Full script with:
- θ-direct training (A: plain L-BFGS, B: L-BFGS + Adam-diagonal)
- Hyper-net (Ω-Fullθ): Δθ around an anchor θ*, with gated [Jacobian / Var / weight-decay] penalties,
  optional nearest-neighbor use in TRAIN and/or INFERENCE (val/OOS), and optional fixed-point refinement.
- FiLM (Ω-FiLM): joint θ + ω with gated [Var(J,β) and/or Jacobian(g)] + weight decay on g.

Key design:
* For Ω-Fullθ: g([X, T]) → Δθ ∈ R^P, f is a batched MLP parameterized by θ = θ* + Δθ.
* For FiLM: g(X) → per-layer (γ, β), applied to hidden pre-activations of f.

All penalties are **gated**:
- If ratio_jac == 0, no Jacobian compute/grad occurs.
- If ratio_var == 0, no Var compute/grad occurs.
- If wd_g == 0, no weight-decay compute/grad occurs.

Nearest-neighbor controls:
- TRAIN: You can feed g([X_i, T_nn(i)]) instead of g([X_i, T_i]) to simulate "borrowed" targets.
- INFER (val/OOS):
    • with fixed-point (FP): you can seed t^(0) with the neighbor’s T (or use a default seed);
    • without FP: you can directly borrow the neighbor’s θ to predict val, else compute θ from g([Xv, Tv]).
"""

import time, math, dataclasses, random
from typing import List, Tuple, Callable, Optional, Dict, Any, Sequence

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# =========================================================
# Repro
# =========================================================
def set_seed(seed=1):
    random.seed(seed);
    np.random.seed(seed);
    tf.random.set_seed(seed)


# =========================================================
# Data
#   X = [t, x, s], s = sqrt(1 - t) if USE_S else X = [t, x]
#   Y = Φ((x-a)/(b*s))
# =========================================================
def std_normal_cdf(z: tf.Tensor) -> tf.Tensor:
    return 0.5 * (1.0 + tf.math.erf(z / tf.sqrt(tf.constant(2.0, dtype=z.dtype))))


def make_dataset(n: int, seed: int, a: float = 0.0, b: float = 1.0, use_s: bool = True) -> Tuple[tf.Tensor, tf.Tensor]:
    rng = np.random.default_rng(seed)
    t = rng.uniform(0.0, 1.0, size=(n, 1)).astype(np.float32)
    x = rng.uniform(-4.0, 4.0, size=(n, 1)).astype(np.float32)
    s = np.sqrt(1.0 - t).astype(np.float32)
    X = np.concatenate([t, x, s], axis=1).astype(np.float32) if use_s else np.concatenate([t, x], axis=1).astype(
        np.float32)
    z = (x - a) / (b * s)
    y = std_normal_cdf(tf.convert_to_tensor(z)).numpy().astype(np.float32)
    return tf.convert_to_tensor(X), tf.convert_to_tensor(y)


# =========================================================
# Architectures (configurable widths)
#   Build MLP with any number of hidden layers from widths list.
# =========================================================
def build_mlp(input_dim: int,
              widths: Sequence[int],
              out_dim: int = 1,
              hidden_activation="tanh",
              out_activation="sigmoid") -> tf.keras.Model:
    inp = tf.keras.Input(shape=(input_dim,), dtype=tf.float32)
    x = inp
    for w in widths:
        x = tf.keras.layers.Dense(int(w), activation=hidden_activation)(x)
    out = tf.keras.layers.Dense(out_dim, activation=out_activation)(x)
    return tf.keras.Model(inputs=inp, outputs=out)


@dataclasses.dataclass
class ArchConfig:
    """
    widths_std_f:   hidden widths for the baseline θ-MLP trained directly.
    widths_omega_f: hidden widths of f used by the hyper-net (defines P via specs).
    widths_omega_g: hidden widths of g in the hyper-net (outputs Δθ of length P).
    widths_film_f:  hidden widths of f in FiLM (defines how many (γ,β) dims).
    widths_film_g:  hidden widths of g in FiLM (outputs concatenated (γ,β)).
    """
    widths_std_f: Tuple[int, ...] = (16, 16, 16)
    widths_omega_f: Tuple[int, ...] = (16, 16, 16)
    widths_omega_g: Tuple[int, ...] = (32, 32)
    widths_film_f: Tuple[int, ...] = (16, 16, 16)
    widths_film_g: Tuple[int, ...] = (32, 32)


# =========================================================
# Regularization / penalties config
# =========================================================
@dataclasses.dataclass
class PenaltyConfig:
    """
    ratio_jac: auto-scale coefficient for Jacobian penalty; 0 disables it.
               lam_jac = ratio_jac * mse0 / jac0 (computed at first iteration).
    ratio_var: auto-scale coefficient for variance penalty on Δθ (hyper) or on mods (FiLM); 0 disables it.
               lam_var = ratio_var * mse0 / var0.
    jac_cols:  indices of input columns to include in Jacobian penalty.
               Hyper-net uses XT=[X,T]; indices refer to XT columns. Default penalizes (t,x) only.
               FiLM uses X; indices refer to X columns.
    jac_probes: # random probes for Hutchinson/RFD.
    jac_mode:   "rfd" (directional finite differences, forward-only; fast) or "hutch" (classic).
    rfd_eps:    step for RFD.
    wd_g:       absolute L2 weight decay on g’s Dense parameters (kernels + biases).
    """
    ratio_jac: float = 0.0
    ratio_var: float = 0.0
    jac_cols: Tuple[int, ...] = (0, 1)
    jac_probes: int = 2
    jac_mode: str = "rfd"  # "hutch" or "rfd"
    rfd_eps: float = 1e-3
    wd_g: float = 0.0


# =========================================================
# Inference & NN controls
# =========================================================
@dataclasses.dataclass
class InferenceConfig:
    """
    use_fp_refine:     if True, run fixed-point solve for t: t = f(x, θ* + g([x,t])) at inference (val).
    fp_max_iter/relax/tol: FP solver knobs.
    use_nn_seed_for_fp: if True, seed FP with neighbor’s T; else seed with f(x, θ*).
    neighbor_borrow_infer: if False and use_fp_refine=False → compute θ from g([Xv,Tv]) for val.
                           if True and use_fp_refine=False  → borrow θ from nearest training sample.
    neighbor_borrow_train: if True → TRAINING feeds g([X_i, T_nn(i)]) with nn excluding self.
    nn_metric:          "x", "tx", "x_scaled" (see knn helpers below).
    clip_lo/clip_hi:    clipping bounds for t within FP loop.
    """
    use_fp_refine: bool = True
    fp_max_iter: int = 20
    fp_relax: float = 1.0
    fp_tol: float = 1e-6
    use_nn_seed_for_fp: bool = True
    neighbor_borrow_infer: bool = False
    neighbor_borrow_train: bool = False
    nn_metric: str = "x_scaled"
    clip_lo: float = 1e-6
    clip_hi: float = 1.0 - 1e-6


# =========================================================
# θ pack/unpack + specs for f(.; θ)
# =========================================================
def pack_variables(vars_list: List[tf.Variable]) -> tf.Tensor:
    return tf.concat([tf.reshape(v, [-1]) for v in vars_list], axis=0)


def unpack_to_variables(flat: tf.Tensor, vars_list: List[tf.Variable]) -> None:
    offset = 0
    for v in vars_list:
        size = tf.size(v)
        new_vals = tf.reshape(flat[offset:offset + size], v.shape)
        v.assign(tf.cast(new_vals, v.dtype))
        offset += size


def layer_specs(input_dim: int, widths: Sequence[int], out_dim: int = 1) -> List[Tuple[int, int]]:
    dims = [input_dim] + list(widths) + [out_dim]
    return [(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]


def theta_size_from_specs(specs: List[Tuple[int, int]]) -> int:
    P = 0
    for din, dout in specs: P += din * dout + dout
    return P


def split_theta_layers(theta_all: tf.Tensor, specs: List[Tuple[int, int]]) -> List[Tuple[tf.Tensor, tf.Tensor]]:
    parts = []
    offset = 0
    N = tf.shape(theta_all)[0]
    for din, dout in specs:
        w_sz = din * dout
        w_flat = theta_all[:, offset:offset + w_sz];
        offset += w_sz
        b_flat = theta_all[:, offset:offset + dout];
        offset += dout
        W = tf.reshape(w_flat, [N, din, dout])
        b = b_flat
        parts.append((W, b))
    return parts


def f_batch_forward(X: tf.Tensor, theta_all: tf.Tensor, specs: List[Tuple[int, int]]) -> tf.Tensor:
    parts = split_theta_layers(theta_all, specs)
    h = X
    for i, (W, b) in enumerate(parts):
        a = tf.einsum('ni,nio->no', h, W) + b
        if i < len(parts) - 1:
            h = tf.math.tanh(a)
        else:
            h = tf.math.sigmoid(a)
    return h


# =========================================================
# FiLM helpers (mods = [γ₁,β₁, …, γ_L,β_L])
# =========================================================
def film_split_mods(mods: tf.Tensor, hidden_widths: Sequence[int]) -> Tuple[List[tf.Tensor], List[tf.Tensor]]:
    idx = 0;
    gammas, betas = [], []
    for d in hidden_widths:
        g = mods[:, idx:idx + d];
        idx += d
        b = mods[:, idx:idx + d];
        idx += d
        gammas.append(g);
        betas.append(b)
    return gammas, betas


def film_forward_from_vars(X: tf.Tensor,
                           f_model: tf.keras.Model,
                           mods: tf.Tensor,
                           hidden_widths: Sequence[int]) -> tf.Tensor:
    Ws, bs = [], []
    for l in f_model.layers:
        if isinstance(l, tf.keras.layers.Dense):
            Ws.append(l.kernel);
            bs.append(l.bias)
    gammas, betas = film_split_mods(mods, hidden_widths)
    h = X
    for i, (W, b) in enumerate(zip(Ws, bs)):
        a = tf.matmul(h, W) + b
        if i < len(hidden_widths):
            a = gammas[i] * a + betas[i]
            h = tf.math.tanh(a)
        else:
            h = tf.math.sigmoid(a)
    return h


# =========================================================
# Jacobian penalties: Hutchinson (tape) and RFD (forward-only)
# =========================================================
def jacobian_frob_penalty_hutch_subset(g_model: tf.keras.Model,
                                       Xin: tf.Tensor,  # inputs watched
                                       cols_in: Sequence[int],
                                       probes: int = 2) -> tf.Tensor:
    cols = tf.constant(cols_in, dtype=tf.int32)
    pen = 0.0
    for _ in range(probes):
        with tf.GradientTape() as tape:
            tape.watch(Xin)
            G = g_model(Xin, training=True)  # [N,P]
            z = tf.random.normal(tf.shape(G))
            v = tf.reduce_sum(G * z, axis=1)  # [N]
        grads = tape.gradient(v, Xin)  # [N,D]
        grads = tf.gather(grads, cols, axis=1)  # [N,|cols|]
        pen += tf.reduce_mean(tf.reduce_sum(grads * grads, axis=1))
    return pen / float(probes)


def jacobian_dir_penalty_rfd_subset(g_model: tf.keras.Model,
                                    Xin: tf.Tensor,
                                    cols_in: Sequence[int],
                                    probes: int = 2,
                                    eps: float = 1e-3) -> tf.Tensor:
    """
    RFD: E_r ||(g(X+eps r) - g(X-eps r))/(2 eps)||^2, with r supported on 'cols_in'.
    Avoids GradientTape on inputs; often faster.
    """
    N = tf.shape(Xin)[0];
    D = tf.shape(Xin)[1];
    C = len(cols_in)
    mask = tf.stack([tf.one_hot(c, D, dtype=Xin.dtype) for c in cols_in], axis=0)  # [C,D]
    pen = 0.0
    for _ in range(probes):
        r = tf.random.normal([N, C], dtype=Xin.dtype)
        r = r / (tf.norm(r, axis=1, keepdims=True) + 1e-12)
        r_full = tf.matmul(r, mask)  # [N,D]
        Xp = Xin + eps * r_full
        Xm = Xin - eps * r_full
        Gp = g_model(Xp, training=True)
        Gm = g_model(Xm, training=True)
        diff = (Gp - Gm) / (2.0 * eps)  # [N,P]
        pen += tf.reduce_mean(tf.reduce_sum(tf.square(diff), axis=1))
    return pen / float(probes)


def l2_on_model(model: tf.keras.Model) -> tf.Tensor:
    s = tf.constant(0.0, dtype=tf.float32)
    for l in model.layers:
        if isinstance(l, tf.keras.layers.Dense):
            if l.kernel is not None: s += tf.reduce_sum(tf.square(l.kernel))
            if l.bias is not None: s += tf.reduce_sum(tf.square(l.bias))
    return s


def variance_penalty(mat: tf.Tensor) -> tf.Tensor:
    mean = tf.reduce_mean(mat, axis=0, keepdims=True)
    return tf.reduce_mean(tf.square(mat - mean))


# =========================================================
# L-BFGS (same structure as before)
# =========================================================
@dataclasses.dataclass
class LBFGSConfig:
    max_iters: int = 200
    mem: int = 20
    c1: float = 1e-4
    powell_c: float = 0.2
    ls_max_steps: int = 10
    alpha_init: float = 1.0
    cub_clip: Tuple[float, float] = (0.1, 2.5)
    curvature_cos_tol: float = 1e-1
    adam_diagonal: bool = False


@dataclasses.dataclass
class LBFGSMemory:
    m: int = 10
    S: List[np.ndarray] = dataclasses.field(default_factory=list)
    Y: List[np.ndarray] = dataclasses.field(default_factory=list)

    def clear(self):
        self.S.clear(); self.Y.clear()

    def push(self, s: np.ndarray, y: np.ndarray):
        self.S.append(s.astype(np.float64));
        self.Y.append(y.astype(np.float64))
        if len(self.S) > self.m: self.S.pop(0); self.Y.pop(0)

    def two_loop(self, g: np.ndarray, gamma: float, d0=None) -> np.ndarray:
        S, Y = self.S, self.Y
        q = g.astype(np.float64).copy();
        alpha, rho = [], []
        for s, y in zip(reversed(S), reversed(Y)):
            r = 1.0 / (np.dot(y, s) + 1e-18)
            a = r * np.dot(s, q);
            alpha.append(a);
            rho.append(r);
            q -= a * y
        q *= (gamma if d0 is None else d0)
        for (s, y, r, a) in zip(S, Y, reversed(rho), reversed(alpha)):
            b = r * np.dot(y, q);
            q += s * (a - b)
        return q


def powell_damp_pair(s: np.ndarray, y: np.ndarray, c: float, gamma: float):
    ss, yy = np.dot(s, s), np.dot(y, y)
    sBs = (1.0 / max(gamma, 1e-16)) * ss
    sy = float(np.dot(s, y))
    if sy < 0: y = -y
    asy = abs(sy)
    fl = ss * yy * 1e-9
    if asy < fl:
        a = fl/yy # This modif made by gpt: fl / max(yy, 1e-18);
        y = a * y;
        asy = fl
    if asy >= c * sBs:
        return y, 1.0, sy, asy, False
    theta = (1.0 - c) * sBs / max(sBs - asy, 1e-16)
    #y_bar = theta * y + (1.0 / max(gamma, 1e-16)) * s  # mistake done by gpt
    y_bar = theta * y + (1.0 - theta) * (1.0 / max(gamma, 1e-16)) * s
    return y_bar, theta, sy, float(np.dot(s, y_bar)), True


def armijo_condition(f0: float, m0: float, f_a: float, a: float, c1: float) -> bool:
    return f_a <= f0 + c1 * a * m0


def cubic_step(f0, m0, a_prev, f_prev, a, f_a, low_mult, high_mult) -> float:
    if a_prev is None or f_prev is None:
        return float(np.clip(0.5 * a, low_mult * a, high_mult * a))
    try:
        A = np.array([[a_prev ** 2, a_prev, 1.0], [a ** 2, a, 1.0], [0.0, 1.0, 0.0]], dtype=float)
        b = np.array([f_prev - f0, f_a - f0, m0], dtype=float)
        coef = np.linalg.lstsq(A, b, rcond=None)[0]
        a_new = -coef[1] / (2.0 * coef[0]) if abs(coef[0]) > 1e-18 else 0.5 * a
    except Exception:
        a_new = 0.5 * a
    a_new = float(np.clip(a_new, low_mult * a, high_mult * a))
    if not np.isfinite(a_new) or a_new <= 0: a_new = float(np.clip(0.5 * a, low_mult * a, high_mult * a))
    return a_new


def update_H0_from_grad(g, v, beta2, eps, target_med):
    v = beta2 * v + (1 - beta2) * (g * g)
    d0 = 1.0 / (np.sqrt(v) + eps)
    med = np.median(d0);
    if med > 0:
        d0 *= (target_med / med)
    d0 = np.clip(d0, 1e-8, 1e+2)
    return v, d0


class LBFGSRunner:
    def __init__(self, vars_list: List[tf.Variable], cfg: LBFGSConfig,
                 loss_closure: Callable[[], Tuple[tf.Tensor, Dict[str, Any]]], seed: int = 0):
        self.vars = vars_list
        self.cfg = cfg
        self.loss_closure = loss_closure
        self.mem = LBFGSMemory(m=cfg.mem)
        self.rng = np.random.default_rng(seed)
        self._last_metrics: Dict[str, Any] = {}

    def get_x_tf64(self) -> tf.Tensor:
        return tf.cast(pack_variables(self.vars), tf.float64)

    def set_x_tf32(self, x_np64: np.ndarray) -> None:
        unpack_to_variables(tf.cast(tf.convert_to_tensor(x_np64), tf.float32), self.vars)

    def loss(self) -> float:
        loss, _ = self.loss_closure()
        return float(loss.numpy())

    def loss_at_x(self, x_np64: np.ndarray) -> float:
        self.set_x_tf32(x_np64);
        return self.loss()

    def compute_loss_grad(self) -> Tuple[float, np.ndarray, Dict[str, Any]]:
        with tf.GradientTape() as tape:
            loss, metrics = self.loss_closure()
        grads = tape.gradient(loss, self.vars)
        flat = tf.concat([tf.reshape(g, [-1]) for g in grads], axis=0)
        # floatify metrics
        fm = {}
        for k, v in metrics.items():
            if isinstance(v, tf.Tensor):
                fm[k] = float(v.numpy())
            else:
                fm[k] = v
        return float(loss.numpy()), flat.numpy().astype(np.float64), fm

    def direction(self, g: np.ndarray, gamma: float, d0=None) -> np.ndarray:
        if len(self.mem.S) == 0: return -(gamma if d0 is None else d0) * g
        return -self.mem.two_loop(g, gamma, d0)

    def line_search(self, theta: np.ndarray, f0: float, g: np.ndarray, d: np.ndarray, m0: Optional[float] = None):
        cfg = self.cfg
        if m0 is None: m0 = float(np.dot(g, d))
        if m0 >= 0:
            d = -g.copy();
            m0 = float(np.dot(g, d))
        alpha = cfg.alpha_init
        a_prev, f_prev = None, None
        best_f, best_a = f0, 0
        trace = [];
        accepted = False;
        count = 0
        for _ in range(cfg.ls_max_steps):
            count += 1
            theta_trial = theta + alpha * d
            f_a = self.loss_at_x(theta_trial)
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
                alpha = best_a; accepted = True
            else:
                alpha = 1e-4 / (np.linalg.norm(d) + 1e-12)
        return float(alpha), accepted, trace, count

    def run_autodiff(self, printer_name: str = "") -> List[Dict[str, Any]]:
        cfg = self.cfg
        history = [];
        t0 = time.perf_counter()
        theta = self.get_x_tf64().numpy()
        f0, g, met = self.compute_loss_grad()
        self._last_metrics = met
        gamma = 1.0;
        print(f"{printer_name}Iter 0: loss={met.get('rmse', math.sqrt(f0)):.6f}")
        history.append(dict(iter=0, t=0.0, loss=met.get('rmse', math.sqrt(f0))))
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
                    c = np.clip(np.dot(d_prev, d) / (ndp * nd), -1.0, 1.0);
                    angle_pi = float(np.arccos(c) / np.pi)
            d_prev = d.copy()
            m0 = float(np.dot(g, d))
            alpha, accepted, ls_trace, ls_iter = self.line_search(theta, f0, g, d, m0=m0)
            total_ls_iter += ls_iter
            theta_next = theta + alpha * d
            self.set_x_tf32(theta_next)
            f_next, g_next, met_next = self.compute_loss_grad()
            s = (theta_next - theta).astype(np.float64);
            y = (g_next - g).astype(np.float64)
            y_bar, theta_mix, sTy, sTyb, _ = powell_damp_pair(s, y, cfg.powell_c, gamma)
            cos_sy = float(sTyb / (np.linalg.norm(s) * (np.linalg.norm(y_bar) + 1e-18) + 1e-18))
            pair_accepted = False
            if sTyb > 1e-12 and cos_sy >= cfg.curvature_cos_tol:
                self.mem.push(s, y_bar);
                pair_accepted = True
                gamma = max(float(sTyb / (np.dot(y_bar, y_bar) + 1e-18)), 1e-8)
                if cfg.adam_diagonal: v, d0 = update_H0_from_grad(g_next, v, beta2, eps, gamma)
            theta, f0, g = theta_next, f_next, g_next
            t_now = time.perf_counter() - t0
            rec = dict(iter=it, t=t_now, loss=met_next.get('rmse', math.sqrt(f_next)), alpha=alpha, armijo=accepted,
                       g_norm=float(np.linalg.norm(g_next)), d_dot_g=float(np.dot(d, g_next)),
                       sTy=sTy, sTy_bar=sTyb, cos_sy=cos_sy, s_norm=float(np.linalg.norm(s)),
                       mem=len(self.mem.S), gamma=gamma, ls_trace=ls_trace)
            self._last_metrics = met_next
            rec.update(met_next)
            if it % 10 == 0:
                pen_str = ""
                if rec.get('lam_jac', 0) > 0 and ('jac' in rec):
                    pen_str += f"+{rec['lam_jac']:.2e}*{rec['jac']:.5g}"
                if rec.get('lam_var', 0) > 0 and ('var' in rec):
                    pen_str += ("" if not pen_str else " ") + f"+{rec['lam_var']:.2e}*{rec['var']:.5g}"
                if rec.get('wd_g', 0) > 0 and ('l2g' in rec):
                    pen_str += ("" if not pen_str else " ") + f"+{rec['wd_g']:.2e}*{rec['l2g']:.5g}"
                print(
                    f"Iter {it}: t={t_now:7.3f}s, , angle={angle_pi:.5f}, "
                    f"loss={rec.get('rmse', math.sqrt(f_next)):.6f}{(' ' + pen_str) if pen_str else ''}, "
                    f"d.g>0:{m0 > 0}, armijo:{accepted}, alpha={alpha:.4e}, "
                    f"iter={total_ls_iter}, |g|={rec['g_norm']:.4e}, "
                    f"neg_sTy={sTy < 0}, powel={theta_mix != 1.0}, pair ok:{pair_accepted}, mem={len(self.mem.S)}, "
                    f"oos={rec.get('oos', None)}"
                )
            history.append(rec)
        return history


# =========================================================
# Closures
# =========================================================
def mse_closure(model: tf.keras.Model, X: tf.Tensor, Y: tf.Tensor) -> Callable[[], Tuple[tf.Tensor, Dict[str, Any]]]:
    def _c():
        pred = model(X, training=True)
        mse = tf.reduce_mean(tf.square(pred - Y))
        return mse, dict(mse=float(mse.numpy()), rmse=float(tf.sqrt(mse).numpy()))

    return _c


# =========================================================
# KNN helpers
#   metric:
#     - "x": only column 1 (x)
#     - "tx": columns 0 and 1 (t and x), Euclidean
#     - "x_scaled": ((x - xk)/sqrt(1 - t_query))^2
# =========================================================
def knn_argmin(X_src: np.ndarray, X_dst: np.ndarray, metric: str = "x") -> np.ndarray:
    if metric == "x":
        src = X_src[:, [1]];
        dst = X_dst[:, [1]]
        dists = ((dst[:, None, :] - src[None, :, :]) ** 2).sum(axis=2)
    elif metric == "tx":
        src = X_src[:, [0, 1]];
        dst = X_dst[:, [0, 1]]
        dists = ((dst[:, None, :] - src[None, :, :]) ** 2).sum(axis=2)
    elif metric == "x_scaled":
        x_src = X_src[:, 1:2];
        x_dst = X_dst[:, 1:2]
        t_dst = X_dst[:, 0:1]
        s_q = np.sqrt(1.0 - np.clip(t_dst, 0.0, 1.0))
        dists = ((x_dst[:, None, :] - x_src[None, :, :]) / (s_q[:, None, :] + 1e-12)) ** 2
        dists = dists[..., 0]
    else:
        src = X_src;
        dst = X_dst
        dists = ((dst[:, None, :] - src[None, :, :]) ** 2).sum(axis=2)
    return dists.argmin(axis=1)


def knn_argmin_excl_self(X: np.ndarray, metric: str = "x") -> np.ndarray:
    """Nearest neighbor indices for X against itself, excluding the diagonal."""
    N = X.shape[0]
    if metric == "x":
        pts = X[:, [1]]
        d = (pts[:, None, :] - pts[None, :, :]) ** 2
    elif metric == "tx":
        pts = X[:, [0, 1]]
        d = ((pts[:, None, :] - pts[None, :, :]) ** 2).sum(axis=2, keepdims=False)
    elif metric == "x_scaled":
        x = X[:, 1:2]
        t = X[:, 0:1]
        s = np.sqrt(1.0 - np.clip(t, 0.0, 1.0))
        # distance from i to j uses s_i (query-side)
        d = ((x[:, None, :] - x[None, :, :]) / (s[:, None, :] + 1e-12)) ** 2
        d = d[..., 0]
    else:
        pts = X
        d = ((pts[:, None, :] - pts[None, :, :]) ** 2).sum(axis=2)
    np.fill_diagonal(d, np.inf)
    return d.argmin(axis=1)


# =========================================================
# Fixed-point refinement for Ω-Fullθ inference
# =========================================================
def fixed_point_predict_batch(Xq: tf.Tensor,
                              g_model: tf.keras.Model,
                              specs_f: List[Tuple[int, int]],
                              theta_star_vec: tf.Tensor,  # [P]
                              seed_t0: tf.Tensor,  # [Nq,1]
                              max_iter: int = 20,
                              relax: float = 1.0,
                              tol: float = 1e-6,
                              clip: Tuple[float, float] = (1e-6, 1.0 - 1e-6)) -> Tuple[tf.Tensor, int]:
    Nq = tf.shape(Xq)[0]
    P = int(theta_star_vec.shape[0])
    theta_star = tf.broadcast_to(theta_star_vec[None, :], [Nq, P])
    t = tf.clip_by_value(tf.identity(seed_t0), clip[0], clip[1])
    k_done = 0
    for k in range(max_iter):
        XT = tf.concat([Xq, t], axis=1)  # [Nq, D+1]
        delta = g_model(XT, training=False)  # [Nq,P]
        theta = theta_star + delta
        y_hat = f_batch_forward(Xq, theta, specs_f)
        diff = tf.reduce_max(tf.abs(y_hat - t))
        t = (1.0 - relax) * t + relax * y_hat
        t = tf.clip_by_value(t, clip[0], clip[1])
        k_done = k + 1
        if float(diff.numpy()) < tol:
            break
    return t, k_done


# =========================================================
# Ω-Fullθ (Δθ) closure
# =========================================================
class HyperNetFullTheta:
    def __init__(self,
                 X: tf.Tensor, Y: tf.Tensor,
                 g_model: tf.keras.Model,
                 f_specs: List[Tuple[int, int]],
                 theta_star_vec: tf.Tensor,  # [P]
                 pen: PenaltyConfig,
                 val_pair: Optional[Tuple[tf.Tensor, tf.Tensor, Optional[np.ndarray]]] = None,
                 infer_cfg: InferenceConfig = InferenceConfig(),
                 # NN toggles
                 nn_train_idx: Optional[np.ndarray] = None  # if provided and neighbor_borrow_train=True, use T_nn(i)
                 ):
        self.X, self.Y = X, Y
        self.g = g_model
        self.specs = f_specs
        self.P = theta_size_from_specs(f_specs)
        self.theta_star_vec = tf.convert_to_tensor(theta_star_vec, dtype=tf.float32)
        if int(self.theta_star_vec.shape[0]) != self.P:
            raise ValueError(
                f"[HyperNetFullTheta] θ* length {int(self.theta_star_vec.shape[0])} != P={self.P}. Ensure θ* comes from widths_omega_f.")
        self.pen = pen
        self.val_pair = val_pair
        self.infer_cfg = infer_cfg
        self.nn_train_idx = nn_train_idx
        self._auto_done = False
        self.lam_jac = 0.0
        self.lam_var = 0.0
        self._last_theta_all = None

    def _compute_oos(self, theta_all_train: Optional[tf.Tensor]) -> Optional[float]:
        if self.val_pair is None: return None
        Xv, Yv, nn_val2train = self.val_pair

        if self.infer_cfg.use_fp_refine:
            # Seed for FP
            if self.infer_cfg.use_nn_seed_for_fp:
                if nn_val2train is None:
                    idx = knn_argmin(self.X.numpy(), Xv.numpy(), metric=self.infer_cfg.nn_metric)
                else:
                    idx = nn_val2train
                seed_t0 = tf.gather(self.Y, tf.convert_to_tensor(idx, dtype=tf.int32), axis=0)
            else:
                # default seed from anchor-only f(x, θ*)
                Nq = tf.shape(Xv)[0]
                theta_star = tf.broadcast_to(self.theta_star_vec[None, :], [Nq, self.P])
                seed_t0 = f_batch_forward(Xv, theta_star, self.specs)

            t_hat, _ = fixed_point_predict_batch(
                Xq=Xv, g_model=self.g, specs_f=self.specs, theta_star_vec=self.theta_star_vec,
                seed_t0=seed_t0, max_iter=self.infer_cfg.fp_max_iter,
                relax=self.infer_cfg.fp_relax, tol=self.infer_cfg.fp_tol,
                clip=(self.infer_cfg.clip_lo, self.infer_cfg.clip_hi)
            )
            yv_hat = t_hat
        else:
            # No FP: either borrow neighbor θ from training, or compute θ via g([Xv,Tv])
            if self.infer_cfg.neighbor_borrow_infer and (nn_val2train is not None) and (theta_all_train is not None):
                idx = tf.convert_to_tensor(nn_val2train, dtype=tf.int32)
                theta_v = tf.gather(theta_all_train, idx, axis=0)
                yv_hat = f_batch_forward(Xv, theta_v, self.specs)
            else:
                XT_v = tf.concat([Xv, Yv], axis=1)
                delta_v = self.g(XT_v, training=False)
                P = int(self.theta_star_vec.shape[0])
                theta_v = delta_v + tf.broadcast_to(self.theta_star_vec[None, :], [tf.shape(Xv)[0], P])
                yv_hat = f_batch_forward(Xv, theta_v, self.specs)

        mse_oos = tf.reduce_mean(tf.square(yv_hat - Yv))
        return float(tf.sqrt(mse_oos).numpy())

    def closure(self) -> Callable[[], Tuple[tf.Tensor, Dict[str, Any]]]:
        def _c():
            # TRAIN-TIME NN option: feed T_nn(i) instead of T_i (if enabled)
            if self.infer_cfg.neighbor_borrow_train and (self.nn_train_idx is not None):
                T_feed = tf.gather(self.Y, tf.convert_to_tensor(self.nn_train_idx, dtype=tf.int32), axis=0)
                XT = tf.concat([self.X, T_feed], axis=1)
            else:
                XT = tf.concat([self.X, self.Y], axis=1)

            delta = self.g(XT, training=True)  # [N,P]
            N = tf.shape(self.X)[0]
            theta_star = tf.broadcast_to(self.theta_star_vec[None, :], [N, self.P])
            theta_all = delta + theta_star
            self._last_theta_all = theta_all

            pred = f_batch_forward(self.X, theta_all, self.specs)
            mse = tf.reduce_mean(tf.square(pred - self.Y))

            # One-time auto-λ
            if not self._auto_done:
                mse0 = float(mse.numpy())
                if self.pen.ratio_jac > 0.0:
                    cols = list(self.pen.jac_cols)
                    if self.pen.jac_mode == "hutch":
                        jac0 = float(
                            jacobian_frob_penalty_hutch_subset(self.g, XT, cols, probes=self.pen.jac_probes).numpy())
                    else:
                        jac0 = float(jacobian_dir_penalty_rfd_subset(self.g, XT, cols, probes=self.pen.jac_probes,
                                                                     eps=self.pen.rfd_eps).numpy())
                    self.lam_jac = (self.pen.ratio_jac * mse0) / (jac0 + 1e-12)
                if self.pen.ratio_var > 0.0:
                    var0 = float(variance_penalty(delta).numpy())
                    self.lam_var = (self.pen.ratio_var * mse0) / (var0 + 1e-12)
                self._auto_done = True

            total = mse
            var_val = tf.constant(0.0, dtype=mse.dtype)
            jac_val = tf.constant(0.0, dtype=mse.dtype)
            if self.pen.ratio_var > 0.0 and self.lam_var > 0.0:
                var_val = variance_penalty(delta)
                total = total + self.lam_var * var_val
            if self.pen.ratio_jac > 0.0 and self.lam_jac > 0.0:
                cols = list(self.pen.jac_cols)
                if self.pen.jac_mode == "hutch":
                    jac_val = jacobian_frob_penalty_hutch_subset(self.g, XT, cols, probes=self.pen.jac_probes)
                else:
                    jac_val = jacobian_dir_penalty_rfd_subset(self.g, XT, cols, probes=self.pen.jac_probes,
                                                              eps=self.pen.rfd_eps)
                total = total + self.lam_jac * jac_val

            l2g = tf.constant(0.0, dtype=mse.dtype)
            if self.pen.wd_g > 0.0:
                l2g = l2_on_model(self.g)
                total = total + self.pen.wd_g * l2g

            rmse = tf.sqrt(mse)
            metrics = dict(mse=mse, rmse=rmse, var=var_val, jac=jac_val,
                           lam_var=self.lam_var, lam_jac=self.lam_jac,
                           wd_g=self.pen.wd_g, l2g=l2g)

            oos = self._compute_oos(self._last_theta_all)
            if oos is not None: metrics['oos'] = oos
            return total, metrics

        return _c


# =========================================================
# FiLM joint closure (with gated Jacobian on g and/or variance on mods)
# =========================================================
class FiLMJoint:
    def __init__(self,
                 X: tf.Tensor, Y: tf.Tensor,
                 f_model: tf.keras.Model,
                 g_model: tf.keras.Model,
                 hidden_widths: Sequence[int],
                 pen: PenaltyConfig,
                 val_pair: Optional[Tuple[tf.Tensor, tf.Tensor, None]] = None):
        self.X, self.Y = X, Y
        self.f = f_model
        self.g = g_model
        self.hidden_widths = list(hidden_widths)
        self.pen = pen
        self._auto_done = False
        self.lam_var = 0.0
        self.lam_jac = 0.0
        self.val_pair = val_pair

    def _mods(self, X: tf.Tensor) -> tf.Tensor:
        return self.g(X, training=True)

    def _oos(self) -> Optional[float]:
        if self.val_pair is None: return None
        Xv, Yv, _ = self.val_pair
        mods_v = self.g(Xv, training=False)
        yv_hat = film_forward_from_vars(Xv, self.f, mods_v, self.hidden_widths)
        mse_oos = tf.reduce_mean(tf.square(yv_hat - Yv))
        return float(tf.sqrt(mse_oos).numpy())

    def closure(self) -> Callable[[], Tuple[tf.Tensor, Dict[str, Any]]]:
        def _c():
            mods = self._mods(self.X)  # [N, 2*sum(hidden)]
            y_hat = film_forward_from_vars(self.X, self.f, mods, self.hidden_widths)
            mse = tf.reduce_mean(tf.square(y_hat - self.Y))

            if not self._auto_done:
                mse0 = float(mse.numpy())
                if self.pen.ratio_var > 0.0:
                    var0 = float(variance_penalty(mods).numpy())
                    self.lam_var = (self.pen.ratio_var * mse0) / (var0 + 1e-12)
                if self.pen.ratio_jac > 0.0:
                    cols = list(self.pen.jac_cols)  # indexes in X
                    if self.pen.jac_mode == "hutch":
                        jac0 = float(jacobian_frob_penalty_hutch_subset(self.g, self.X, cols,
                                                                        probes=self.pen.jac_probes).numpy())
                    else:
                        jac0 = float(jacobian_dir_penalty_rfd_subset(self.g, self.X, cols, probes=self.pen.jac_probes,
                                                                     eps=self.pen.rfd_eps).numpy())
                    self.lam_jac = (self.pen.ratio_jac * mse0) / (jac0 + 1e-12)
                self._auto_done = True

            total = mse
            var_val = tf.constant(0.0, dtype=mse.dtype)
            jac_val = tf.constant(0.0, dtype=mse.dtype)
            if self.pen.ratio_var > 0.0 and self.lam_var > 0.0:
                var_val = variance_penalty(mods);
                total += self.lam_var * var_val
            if self.pen.ratio_jac > 0.0 and self.lam_jac > 0.0:
                cols = list(self.pen.jac_cols)
                if self.pen.jac_mode == "hutch":
                    jac_val = jacobian_frob_penalty_hutch_subset(self.g, self.X, cols, probes=self.pen.jac_probes)
                else:
                    jac_val = jacobian_dir_penalty_rfd_subset(self.g, self.X, cols, probes=self.pen.jac_probes,
                                                              eps=self.pen.rfd_eps)
                total += self.lam_jac * jac_val

            l2g = tf.constant(0.0, dtype=mse.dtype)
            if self.pen.wd_g > 0.0:
                l2g = l2_on_model(self.g)
                total += self.pen.wd_g * l2g

            rmse = tf.sqrt(mse)
            metrics = dict(mse=mse, rmse=rmse, var=var_val, jac=jac_val,
                           lam_var=self.lam_var, lam_jac=self.lam_jac,
                           wd_g=self.pen.wd_g, l2g=l2g)
            oos = self._oos()
            if oos is not None: metrics['oos'] = oos
            return total, metrics

        return _c


# =========================================================
# Plotting
# =========================================================
def plot_convergence_theta(hist_a, hist_b):
    def unpack(curve): return [r['t'] for r in curve], [r['loss'] for r in curve]

    ta, la = unpack(hist_a);
    tb, lb = unpack(hist_b)
    plt.figure();
    plt.plot(ta, la, label='A: L-BFGS');
    plt.plot(tb, lb, label='B: L-BFGS + Adam diag')
    plt.xlabel('Wall time (s)');
    plt.ylabel('√MSE');
    plt.yscale('log');
    plt.legend();
    plt.grid(True, which='both', ls=':');
    plt.title('θ convergence')


def plot_convergence_omega(hist, title='Ω convergence'):
    t = [r['t'] for r in hist]
    rmse = [r.get('rmse', float('nan')) for r in hist]
    oos = [r.get('oos', float('nan')) for r in hist]
    use_jac = any(np.isfinite(r.get('jac', float('nan'))) for r in hist)
    pen = [r.get('jac', float('nan')) for r in hist] if use_jac else [r.get('var', float('nan')) for r in hist]
    yright = 'Jacobian proxy' if use_jac else 'Variance'
    fig, ax1 = plt.subplots()
    ax1.plot(t, rmse, label='√MSE (train)')
    if np.isfinite(np.array(oos)).any(): ax1.plot(t, oos, '--', label='√MSE (OOS)')
    ax1.set_xlabel('Wall time (s)');
    ax1.set_ylabel('√MSE');
    ax1.set_yscale('log');
    ax1.grid(True, which='both', ls=':')
    ax2 = ax1.twinx();
    ax2.plot(t, pen, alpha=0.65, label=yright);
    ax2.set_ylabel(yright)
    lines1, labels1 = ax1.get_legend_handles_labels();
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best');
    plt.title(title)


# =========================================================
# Main
# =========================================================
if __name__ == "__main__":
    set_seed(1)

    # --- data knobs
    USE_S = True  # set False to drop s = sqrt(1 - t) from inputs entirely
    D_in = 3 if USE_S else 2

    Ntr, Nval = 4096, 512
    A_PARAM, B_PARAM = 0.0, 1.0
    ITER = 200

    # --- architectures
    arch = ArchConfig(
        widths_std_f=(16, 16, 16),
        widths_omega_f=(16, 16, 16),  # must define the same P as used by hyper-net f-specs
        widths_omega_g=(16, 16),
        widths_film_f=(16, 16, 16),
        widths_film_g=(16, 16),
    )

    # --- penalties
    pen_omega = PenaltyConfig(
        ratio_jac=0.0,  # >0 enables Jacobian; 0 disables and skips compute/grad
        ratio_var=0.0,  # keep 0 for kNN flavor; set >0 to stabilize Δθ variance
        jac_cols=(0, 1),  # penalize only (t,x); do NOT include target T column
        jac_probes=2,
        jac_mode="rfd",  # "rfd" is faster than "hutch"
        rfd_eps=1e-3,
        wd_g=0.0  # L2(g-weights); 0 disables
    )

    pen_film = PenaltyConfig(
        ratio_jac=0.0,  # set >0 to regularize g wrt (t,x)
        ratio_var=0.0,  # variance on mods
        jac_cols=(0, 1),
        jac_probes=2,
        jac_mode="rfd",
        rfd_eps=1e-3,
        wd_g=0.0
    )

    # --- inference / neighbor usage
    infer_cfg = InferenceConfig(
        use_fp_refine=True,  # fixed-point inference for hyper-net
        fp_max_iter=20,
        fp_relax=1.0,
        fp_tol=1e-6,
        use_nn_seed_for_fp=True,  # seed FP with neighbor’s T; False -> seed with f(x; θ*)
        neighbor_borrow_infer=True,  # only used if use_fp_refine=False
        neighbor_borrow_train=False,  # TRAIN: feed T_nn(i) instead of T_i to g
        nn_metric="x_scaled",
        clip_lo=1e-6, clip_hi=1.0 - 1e-6
    )

    Xtr, Ytr = make_dataset(Ntr, seed=1, a=A_PARAM, b=B_PARAM, use_s=USE_S)

    # =====================================================
    # θ-direct training (A, B)
    # =====================================================
    model_a = build_mlp(D_in, arch.widths_std_f, out_dim=1, hidden_activation='tanh', out_activation='sigmoid')
    model_b = build_mlp(D_in, arch.widths_std_f, out_dim=1, hidden_activation='tanh', out_activation='sigmoid')
    model_b.set_weights([w.copy() for w in model_a.get_weights()])

    Xval, Yval = make_dataset(Nval, seed=2, a=A_PARAM, b=B_PARAM, use_s=USE_S)


    @dataclasses.dataclass
    class LB(LBFGSConfig):
        pass


    cfg_a = LB(max_iters=ITER, mem=20, adam_diagonal=False)
    cfg_b = LB(max_iters=ITER, mem=20, adam_diagonal=True)

    print("\n=== Mode A: L-BFGS (θ) ===")
    runner_a = LBFGSRunner(model_a.trainable_variables, cfg_a, mse_closure(model_a, Xtr, Ytr), seed=123)
    hist_a = runner_a.run_autodiff()

    print("\n=== Mode B: L-BFGS + Adam-diag (θ) ===")
    runner_b = LBFGSRunner(model_b.trainable_variables, cfg_b, mse_closure(model_b, Xtr, Ytr), seed=123)
    hist_b = runner_b.run_autodiff()


    # =====================================================
    # θ* anchor for Ω-Fullθ must match widths_omega_f
    # =====================================================
    f_anchor = build_mlp(D_in, arch.widths_omega_f, out_dim=1, hidden_activation='tanh', out_activation='sigmoid')
    if tuple(arch.widths_omega_f) == tuple(arch.widths_std_f):
        # strong init: copy from better baseline model
        f_anchor.set_weights([w.copy() for w in model_b.get_weights()])
    else:
        # quick pre-train θ* (optional but recommended)
        cfg_anchor = LB(max_iters=60, mem=20, adam_diagonal=True)
        runner_anchor = LBFGSRunner(f_anchor.trainable_variables, cfg_anchor, mse_closure(f_anchor, Xtr, Ytr), seed=123)
        _ = runner_anchor.run_autodiff(printer_name="[anchor f] ")

    theta_star_vec = pack_variables(f_anchor.trainable_variables)

    # =====================================================
    # Ω-Fullθ: Δθ around θ* (gated penalties + NN controls)
    # =====================================================
    specs_f = layer_specs(input_dim=D_in, widths=arch.widths_omega_f, out_dim=1)
    P = theta_size_from_specs(specs_f)
    g_in_dim = D_in + 1  # g input = [X, T]
    g_hnet = build_mlp(g_in_dim, arch.widths_omega_g, out_dim=P, hidden_activation='tanh', out_activation=None)

    # Precompute neighbors for TRAIN (exclude self) and for VAL (val→train)
    nn_train_idx = None
    if infer_cfg.neighbor_borrow_train:
        nn_train_idx = knn_argmin_excl_self(Xtr.numpy(), metric=infer_cfg.nn_metric)
    nn_val2train = knn_argmin(Xtr.numpy(), Xval.numpy(), metric=infer_cfg.nn_metric)

    hnet = HyperNetFullTheta(
        Xtr, Ytr, g_hnet, specs_f, theta_star_vec, pen_omega,
        val_pair=(Xval, Yval, nn_val2train),
        infer_cfg=infer_cfg,
        nn_train_idx=nn_train_idx,
    )

    cfg_o = LB(max_iters=ITER, mem=20, adam_diagonal=True)
    print("\n=== Ω-Fullθ: Δθ + gated [Jac/Var] ===")
    runner_o = LBFGSRunner(g_hnet.trainable_variables, cfg_o, hnet.closure(), seed=123)
    hist_o = runner_o.run_autodiff(printer_name="")

    # =====================================================
    # Ω-FiLM: joint θ+ω with gated penalties
    # =====================================================
    f_film = build_mlp(D_in, arch.widths_film_f, out_dim=1, hidden_activation='tanh', out_activation='sigmoid')
    mods_dim = 2 * sum(arch.widths_film_f)  # per hidden layer: (γ, β)
    g_film = build_mlp(D_in, arch.widths_film_g, out_dim=mods_dim, hidden_activation='tanh', out_activation=None)

    film = FiLMJoint(Xtr, Ytr, f_film, g_film, hidden_widths=arch.widths_film_f, pen=pen_film,
                     val_pair=(Xval, Yval, None))
    cfg_f = LB(max_iters=ITER, mem=20, adam_diagonal=True)
    print("\n=== Ω-FiLM: joint θ+ω with gated [Jac/Var] ===")
    runner_f = LBFGSRunner(f_film.trainable_variables + g_film.trainable_variables, cfg_f, film.closure(), seed=123)
    hist_f = runner_f.run_autodiff(printer_name="")

    # =====================================================
    # Results & plots
    # =====================================================
    print("\nFinal results (θ-trained):")
    print(f"Mode A: √MSE={hist_a[-1]['loss']:.6f}, wall_time={hist_a[-1]['t']:.3f}s")
    print(f"Mode B: √MSE={hist_b[-1]['loss']:.6f}, wall_time={hist_b[-1]['t']:.3f}s")

    print("\nFinal results (Ω-trained) — plain √MSE on train:")
    print(f"Ω-Fullθ:  √MSE={hist_o[-1]['rmse']:.6f},  OOS={hist_o[-1].get('oos', None)}")
    print(f"Ω-FiLM:   √MSE={hist_f[-1]['rmse']:.6f},  OOS={hist_f[-1].get('oos', None)}")

    # Convergence charts
    plot_convergence_theta(hist_a, hist_b)
    plot_convergence_omega(hist_o, title='Ω-Fullθ (Δθ) convergence (FP OOS as configured)')
    plot_convergence_omega(hist_f, title='Ω-FiLM convergence')
    plt.show()
