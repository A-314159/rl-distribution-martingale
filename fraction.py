# piecewise_rational_normal.py
# Nonlinear Mixture-of-Experts (soft-EM) for
#   f(x,t) = N( (x - a t - b t^2) / sqrt(1 - t) )
# on (-4,4)×(0,1), with rational P/Q experts and a learned nonlinear gate.
#
# No prior knowledge of the split is used.
# Dependencies: numpy (matplotlib optional for plotting)

from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Optional

# ============================ Normal CDF =============================

SQRT2 = math.sqrt(2.0)
# Vectorized math.erf (Python 3.7-friendly)
_ERF = np.vectorize(math.erf, otypes=[np.float64])

def Phi(z: np.ndarray) -> np.ndarray:
    """Standard normal CDF (hybrid not needed; erf is sufficient here)."""
    z = np.asarray(z, dtype=np.float64)
    return 0.5 * (1.0 + _ERF(z / SQRT2))

def target_f(x: np.ndarray, t: np.ndarray, a: float, b: float,
             t_floor: float = 1e-15, t_ceil: float = 1.0 - 1e-15) -> np.ndarray:
    """f(x,t)=N((x - a t - b t^2)/sqrt(1-t)), vectorized, safe near t→1-."""
    x = np.asarray(x, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    t = np.clip(t, t_floor, t_ceil)
    s = np.sqrt(1.0 - t)
    z = (x - a * t - b * t * t) / s
    return Phi(z)

# ===================== Chebyshev tensor rational =====================

def chebvander_1d(xhat: np.ndarray, deg: int) -> np.ndarray:
    """Chebyshev Vandermonde (first kind) via recurrence; shape (..., deg+1)."""
    xhat = np.asarray(xhat, dtype=np.float64)
    V = np.zeros(xhat.shape + (deg + 1,), dtype=np.float64)
    V[..., 0] = 1.0
    if deg >= 1:
        V[..., 1] = xhat
        for n in range(2, deg + 1):
            V[..., n] = 2 * xhat * V[..., n - 1] - V[..., n - 2]
    return V

def map_to_box(x: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Map x∈[-4,4] -> xhat∈[-1,1],  t∈[0,1] -> that∈[-1,1]."""
    return (x / 4.0, 2.0 * t - 1.0)

def tensor_features_cheb(x: np.ndarray, t: np.ndarray, dx: int, dt: int) -> np.ndarray:
    """Chebyshev tensor features T_i(x̂)T_j(t̂), flattened with i-major/j-minor order."""
    xhat, that = map_to_box(x, t)
    Vx = chebvander_1d(xhat, dx)  # (..., dx+1)
    Vt = chebvander_1d(that, dt)  # (..., dt+1)
    feats = []
    for i in range(dx + 1):
        for j in range(dt + 1):
            feats.append(Vx[..., i] * Vt[..., j])
    return np.stack(feats, axis=-1)  # (..., (dx+1)(dt+1))

# ====================== Utilities for logits/probs ====================

def _logit_clip(y: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    y = np.clip(y, eps, 1.0 - eps)
    return np.log(y / (1.0 - y))

def _sigmoid_safe(z: np.ndarray) -> np.ndarray:
    """Stable sigmoid."""
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)
    m = (z >= 0)
    out[m]  = 1.0 / (1.0 + np.exp(-z[m]))
    ez = np.exp(z[~m])
    out[~m] = ez / (1.0 + ez)
    return out

# ============================ Experts (P/Q) ===========================

@dataclass
class ChebRational2D:
    dx_num: int
    dt_num: int
    dx_den: int
    dt_den: int
    a_num: np.ndarray
    b_den: np.ndarray
    g_clip: float = 10.0  # clamp on raw logit for extra robustness

    @staticmethod
    def fit(x, t, y,
            dx_num, dt_num, dx_den, dt_den,
            ridge=1e-4, weights: Optional[np.ndarray] = None,
            sk_iters: int = 8, q_floor: float = 1e-2, g_clip: float = 10.0) -> "ChebRational2D":
        """
        Fit a rational P/Q to the *logit* of y with ridge and SK reweighting.
        - Linearized LS: P - g*(Q - 1) ≈ g
        - SK: weights ~ 1/max(|Q_prev|, q_floor) to push poles away
        """
        x = np.asarray(x); t = np.asarray(t); y = np.asarray(y)
        g = _logit_clip(y)  # transform (0,1) -> R

        Phi_num = tensor_features_cheb(x, t, dx_num, dt_num)               # (N, nnum)
        Phi_den_full = tensor_features_cheb(x, t, dx_den, dt_den)          # (N, nden_full)
        mask_den = np.ones(Phi_den_full.shape[-1], dtype=bool); mask_den[0] = False
        Phi_den = Phi_den_full[..., mask_den]                               # (N, nden)

        # Initial LS
        A = np.concatenate([Phi_num, -g[..., None] * Phi_den], axis=-1)
        b = g.copy()
        if weights is not None:
            W = np.sqrt(np.asarray(weights, float))[..., None]
            A = W * A; b = (W[..., 0] * b)
        ATA = A.T @ A + ridge * np.eye(A.shape[1])
        ATb = A.T @ b
        theta = np.linalg.solve(ATA, ATb)
        a = theta[:Phi_num.shape[-1]]
        bcoef = np.zeros(Phi_den_full.shape[-1], float)
        bcoef[0] = 1.0; bcoef[mask_den] = theta[Phi_num.shape[-1]:]

        # SK iterations
        for _ in range(sk_iters):
            Qprev = Phi_den_full @ bcoef
            w = 1.0 / np.maximum(np.abs(Qprev), q_floor)
            W = np.sqrt(w)[..., None]
            A = np.concatenate([W * Phi_num,  -(W[...,0] * g)[...,None] * Phi_den], axis=-1)
            b = (W[...,0] * g)
            ATA = A.T @ A + ridge * np.eye(A.shape[1])
            ATb = A.T @ b
            theta = np.linalg.solve(ATA, ATb)
            a = theta[:Phi_num.shape[-1]]
            bcoef = np.zeros(Phi_den_full.shape[-1], float)
            bcoef[0] = 1.0; bcoef[mask_den] = theta[Phi_num.shape[-1]:]

        return ChebRational2D(dx_num, dt_num, dx_den, dt_den, a, bcoef, g_clip=g_clip)

    def eval_raw(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        P = (tensor_features_cheb(x, t, self.dx_num, self.dt_num) @ self.a_num)
        Q = (tensor_features_cheb(x, t, self.dx_den, self.dt_den) @ self.b_den)
        return P / Q

    def eval(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Prediction in (0,1): sigmoid(P/Q), with raw-logit clamp for stability."""
        g = np.clip(self.eval_raw(x, t), -self.g_clip, self.g_clip)
        return _sigmoid_safe(g)

# =============================== Gate ================================

def poly_features(x: np.ndarray, t: np.ndarray, deg: int) -> np.ndarray:
    """Polynomial features on (x̂, t̂): 1, x̂^i, t̂^j, and cross terms up to 'deg'."""
    xhat, that = map_to_box(x, t)
    feats = [np.ones_like(xhat)]
    for i in range(1, deg + 1):
        feats.append(xhat ** i)
    for j in range(1, deg + 1):
        feats.append(that ** j)
    for i in range(1, deg + 1):
        for j in range(1, deg + 1):
            feats.append((xhat ** i) * (that ** j))
    return np.stack(feats, axis=-1)  # (..., F)

@dataclass
class LogisticGate:
    W: np.ndarray  # (F, K)
    deg: int

    @staticmethod
    def fit_soft(x: np.ndarray, t: np.ndarray, R: np.ndarray,
                 deg: int = 3, l2: float = 1e-3, iters: int = 600, lr: float = 0.2) -> "LogisticGate":
        """
        Train a softmax gate on *soft* responsibilities R (N×K), minimizing
        cross-entropy: -∑_n ∑_k R_{nk} log P_{nk}.
        """
        x = np.asarray(x); t = np.asarray(t); R = np.asarray(R, dtype=np.float64)
        N, K = R.shape
        Phi = poly_features(x, t, deg)  # (N, F)
        F = Phi.shape[1]
        W = np.zeros((F, K), dtype=np.float64)

        for _ in range(iters):
            Z = Phi @ W
            Z -= Z.max(axis=1, keepdims=True)
            P = np.exp(Z); P /= P.sum(axis=1, keepdims=True)  # (N, K)
            G = Phi.T @ (P - R) / N + l2 * W
            W -= lr * G
        return LogisticGate(W, deg)

    def proba(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        Phi = poly_features(x, t, self.deg)
        Z = Phi @ self.W
        Z -= Z.max(axis=-1, keepdims=True)
        P = np.exp(Z); P /= P.sum(axis=-1, keepdims=True)
        return P

    def predict(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        return self.proba(x, t).argmax(axis=-1)

# =================== Mixture-of-Experts model =======================

@dataclass
class PiecewiseRational:
    experts: List[ChebRational2D]
    gate: LogisticGate
    K: int

    def predict(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Hard gate (nonlinear boundary): pick argmax expert."""
        x = np.asarray(x); t = np.asarray(t)
        k = self.gate.predict(x, t)
        out = np.zeros_like(x, dtype=np.float64)
        for idx in range(self.K):
            mask = (k == idx)
            if np.any(mask):
                out[mask] = self.experts[idx].eval(x[mask], t[mask])
        return out

    def predict_soft(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Soft mixture: ∑_k p_k(x,t) * expert_k(x,t)."""
        x = np.asarray(x); t = np.asarray(t)
        Pk = self.gate.proba(x, t)  # (..., K)
        vals = np.zeros_like(x, dtype=np.float64)
        for idx in range(self.K):
            vals += Pk[..., idx] * self.experts[idx].eval(x, t)
        return vals

# ====================== Soft-EM (no split prior) =====================

def huber_loss(x: np.ndarray, delta: float = 0.02) -> np.ndarray:
    a = np.abs(x)
    return np.where(a <= delta, 0.5 * a * a, delta * (a - 0.5 * delta))

def update_responsibilities(x, t, y, experts: List[ChebRational2D], gate: LogisticGate,
                            tau: float = 0.03, huber_delta: float = 0.02,
                            eps: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    """r_{nk} ∝ gate_prob * exp( -Huber( y - y_k ) / τ ), row-normalized."""
    preds = np.stack([exp.eval(x, t) for exp in experts], axis=1)  # (N, K)
    loss  = huber_loss(preds - y[:, None], delta=huber_delta)      # (N, K)
    G     = gate.proba(x, t)                                       # (N, K)
    r     = G * np.exp(-loss / tau)
    r     = np.maximum(r, eps)
    r    /= r.sum(axis=1, keepdims=True)
    return r, preds

def kmeans2(xy: np.ndarray, K: int, iters: int = 20, seed: int = 0) -> np.ndarray:
    """Tiny K-means in 2D for initialization."""
    rng = np.random.default_rng(seed)
    N = xy.shape[0]
    centers = xy[rng.choice(N, size=K, replace=False)].copy()
    labels = np.zeros(N, dtype=int)
    for _ in range(iters):
        d2 = ((xy[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = d2.argmin(axis=1)
        for k in range(K):
            m = labels == k
            if m.any():
                centers[k] = xy[m].mean(axis=0)
            else:
                centers[k] = xy[rng.integers(0, N)]
    return labels

def fit_piecewise_moe(a: float, b: float,
                      K: int = 2,
                      N_train: int = 25000,
                      N_val: int = 30000,
                      dx_num: int = 6, dt_num: int = 6,
                      dx_den: int = 5, dt_den: int = 2,   # slightly lower t-degree in Q
                      ridge: float = 1e-4,
                      gate_deg: int = 3,
                      em_iters: int = 10,
                      tau_start: float = 0.05, tau_end: float = 0.01,
                      huber_delta: float = 0.02,
                      seed: int = 0) -> Tuple[PiecewiseRational, dict]:
    """
    Fully unsupervised soft-EM MoE (no f=0.5 prior).
    Returns (model, validation metrics dict).
    """
    rng = np.random.default_rng(seed)
    # training/validation samples
    x_tr = rng.uniform(-4.0, 4.0, N_train)
    t_tr = rng.uniform(0.0, 1.0, N_train)
    y_tr = target_f(x_tr, t_tr, a=a, b=b)

    x_va = rng.uniform(-4.0, 4.0, N_val)
    t_va = rng.uniform(0.0, 1.0, N_val)
    y_va = target_f(x_va, t_va, a=a, b=b)

    # init responsibilities via K-means
    labels0 = kmeans2(np.stack([x_tr, t_tr], axis=1), K=K, seed=seed)
    R = np.zeros((N_train, K), dtype=np.float64)
    for k in range(K):
        m = (labels0 == k)
        if not np.any(m):
            m = rng.choice(N_train, size=max(1, N_train // K), replace=False)
        R[m, k] = 1.0
    R /= R.sum(axis=1, keepdims=True)

    # init gate and experts
    gate = LogisticGate.fit_soft(x_tr, t_tr, R, deg=gate_deg, l2=1e-3, iters=600, lr=0.2)
    experts: List[ChebRational2D] = [None] * K  # type: ignore
    for k in range(K):
        experts[k] = ChebRational2D.fit(
            x_tr, t_tr, y_tr,
            dx_num, dt_num, dx_den, dt_den,
            ridge=ridge, weights=R[:, k],
            sk_iters=8, q_floor=1e-2, g_clip=10.0
        )

    # EM loop with annealed temperature
    for it in range(em_iters):
        tau = tau_start + (tau_end - tau_start) * (it / max(1, em_iters - 1))
        # E-step
        R, _ = update_responsibilities(x_tr, t_tr, y_tr, experts, gate,
                                       tau=tau, huber_delta=huber_delta)
        # M-step
        for k in range(K):
            experts[k] = ChebRational2D.fit(
                x_tr, t_tr, y_tr,
                dx_num, dt_num, dx_den, dt_den,
                ridge=ridge, weights=R[:, k],
                sk_iters=8, q_floor=1e-2, g_clip=10.0
            )
        gate = LogisticGate.fit_soft(x_tr, t_tr, R, deg=gate_deg, l2=1e-3, iters=800, lr=0.15)

    # package model
    model = PiecewiseRational(experts=experts, gate=gate, K=K)

    # validation metrics
    yh_h = model.predict(x_va, t_va)
    yh_s = model.predict_soft(x_va, t_va)
    e_h = np.abs(yh_h - y_va); e_s = np.abs(yh_s - y_va)
    metrics = {
        "val_RMSE_hard": float(np.sqrt(np.mean((yh_h - y_va) ** 2))),
        "val_max_abs_hard": float(e_h.max()),
        "val_med_abs_hard": float(np.median(e_h)),
        "val_RMSE_soft": float(np.sqrt(np.mean((yh_s - y_va) ** 2))),
        "val_max_abs_soft": float(e_s.max()),
        "val_med_abs_soft": float(np.median(e_s)),
    }
    return model, metrics

# ============================== Plots ================================

def plot_3d_pred_and_error(model: PiecewiseRational, a: float, b: float,
                           Nx: int = 80, Nt: int = 80, soft: bool = True) -> None:
    """3D surfaces: (1) predicted value, (2) absolute error."""
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    except Exception as e:
        print("Matplotlib not available:", e)
        return

    xs = np.linspace(-4.0, 4.0, Nx)
    ts = np.linspace(1e-6, 1.0 - 1e-6, Nt)
    X, T = np.meshgrid(xs, ts, indexing="xy")

    Y_pred = model.predict_soft(X, T) if soft else model.predict(X, T)
    Y_true = target_f(X, T, a=a, b=b)
    Y_err  = np.abs(Y_pred - Y_true)

    fig1 = plt.figure(figsize=(9, 6))
    ax1 = fig1.add_subplot(111, projection="3d")
    s1 = ax1.plot_surface(X, T, Y_pred, rstride=1, cstride=1, linewidth=0, antialiased=True, cmap="viridis")
    ax1.set_xlabel("x"); ax1.set_ylabel("t"); ax1.set_zlabel("predicted f")
    ax1.set_title("Predicted value ({})".format("soft mixture" if soft else "hard gate"))
    fig1.colorbar(s1, shrink=0.7, aspect=12)

    fig2 = plt.figure(figsize=(9, 6))
    ax2 = fig2.add_subplot(111, projection="3d")
    s2 = ax2.plot_surface(X, T, Y_err, rstride=1, cstride=1, linewidth=0, antialiased=True, cmap="magma")
    ax2.set_xlabel("x"); ax2.set_ylabel("t"); ax2.set_zlabel("|error|")
    ax2.set_title("Absolute error vs true f")
    fig2.colorbar(s2, shrink=0.7, aspect=12)

    plt.show()

def plot_gate_vs_zt0(model: PiecewiseRational, a: float, b: float, Nx=500, Nt=500):
    """
    Plot the learned gate boundary (p1 == p0) vs the curve x = a t + b t^2 (z=0).
    This is only for *visualization*; the trainer does not use this prior.
    """
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print("Matplotlib not available:", e)
        return

    if getattr(model, "K", None) != 2:
        raise ValueError("This helper expects K=2.")

    xs = np.linspace(-4.0, 4.0, Nx)
    ts = np.linspace(1e-6, 1.0 - 1e-6, Nt)
    X, T = np.meshgrid(xs, ts, indexing="xy")

    P = model.gate.proba(X, T)              # (...,2)
    score = P[..., 1] - P[..., 0]           # zero-level set

    fig, ax = plt.subplots(figsize=(7, 5))
    cs = ax.contour(X, T, score, levels=[0.0], colors="k", linewidths=2)
    if cs.collections:
        cs.collections[0].set_label("learned boundary (p1 = p0)")
    ax.plot(a * ts + b * ts**2, ts, "r--", lw=2, label=r"$x = a\,t + b\,t^2$ (z=0)")

    ax.set_xlim(-4, 4); ax.set_ylim(0, 1)
    ax.set_xlabel("x"); ax.set_ylabel("t")
    ax.set_title("Learned gate vs theoretical z=0 curve (for sanity check)")
    ax.legend(loc="best")
    fig.tight_layout()
    plt.show()

# ============================== Demo ================================

def _demo():
    a, b = 0.7, -0.3
    model, metrics = fit_piecewise_moe(
        a=a, b=b,
        K=2,
        N_train=25000, N_val=40000,
        dx_num=6, dt_num=6, dx_den=5, dt_den=2,
        ridge=1e-4,
        gate_deg=3,
        em_iters=10,
        tau_start=0.06, tau_end=0.02,
        huber_delta=0.02,
        seed=42
    )
    print("Validation metrics (hard gate):")
    print(f"  RMSE   = {metrics['val_RMSE_hard']:.4e}")
    print(f"  max|e| = {metrics['val_max_abs_hard']:.4e}")
    print(f"  med|e| = {metrics['val_med_abs_hard']:.4e}")
    print("Validation metrics (soft mixture):")
    print(f"  RMSE   = {metrics['val_RMSE_soft']:.4e}")
    print(f"  max|e| = {metrics['val_max_abs_soft']:.4e}")
    print(f"  med|e| = {metrics['val_med_abs_soft']:.4e}")

    # Visualizations (optional)
    plot_3d_pred_and_error(model, a=a, b=b, Nx=80, Nt=80, soft=True)
    plot_gate_vs_zt0(model, a=a, b=b)

_demo()
