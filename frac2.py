# moe_expQ_normal.py
# Mixture-of-Experts for f(x,t)=N((x - a t - b t^2)/sqrt(1-t)) on (-4,4)x(0,1)
# with *positive* denominator: Q(x,t)=exp(R(x,t)). No prior knowledge of the split.
#
# - 2 experts by default (K=2), learned with soft-EM (no f=0.5 hint)
# - each expert fits the *logit* target g=logit(y) with Gauss-Newton / LM on
#     g_hat(x,t) = P(x,t) * exp(-R(x,t))        [since Q=exp(R)]
# - final prediction y_hat = sigmoid(g_hat) ∈ (0,1)
# - includes 3D surfaces for prediction and error, plus boundary plot (optional)
#
# Dependencies: numpy (matplotlib optional)

from __future__ import annotations
import math, numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict

# ============================ Target: normal CDF ============================

SQRT2 = math.sqrt(2.0)
_ERF = np.vectorize(math.erf, otypes=[np.float64])

def _poly_fit_weighted_x_of_t(ts, xs_target, w, deg=3):
    """
    Weighted LS fit of x ≈ sum_{k=0}^deg c_k t^k.
    ts, xs_target, w are 1D arrays of same length.
    Returns coeffs c (length deg+1), in increasing powers.
    """
    ts = np.asarray(ts, float); xs_target = np.asarray(xs_target, float); w = np.asarray(w, float)
    V = np.vander(ts, N=deg+1, increasing=True)                # [1, t, t^2, ...]
    # Solve (V^T W V) c = V^T W x
    WV = (w[:, None] * V)
    A = V.T @ WV
    b = V.T @ (w * xs_target)
    return np.linalg.solve(A + 1e-12*np.eye(A.shape[0]), b)

def fit_moe_expQ_gradgate(a: float, b: float,
                          K: int = 2,
                          N_train: int = 6000,        # fewer samples are fine
                          N_val: int = 12000,
                          dxP: int = 7, dtP: int = 6, # your better setting
                          dxR: int = 3, dtR: int = 2,
                          ridgeP: float = 1e-4, ridgeR: float = 1e-4,
                          em_iters: int = 10,
                          gate_deg: int = 3,
                          boundary_deg: int = 3,
                          tau_start: float = 0.06, tau_end: float = 0.02,
                          huber_delta: float = 0.02,
                          seed: int = 0) -> Tuple[MoEModel, dict]:
    """
    MoE with Q=exp(R) experts. Gate is initialized from a *data-driven boundary*
    extracted as the ridge of |∂f/∂x| — no f=0.5 prior.
    """
    rng = np.random.default_rng(seed)
    # Samples (uniform; generic)
    x_tr = rng.uniform(-4.0, 4.0, N_train)
    t_tr = rng.uniform(0.0, 1.0, N_train)
    y_tr = target_f(x_tr, t_tr, a=a, b=b)

    x_va = rng.uniform(-4.0, 4.0, N_val)
    t_va = rng.uniform(0.0, 1.0, N_val)
    y_va = target_f(x_va, t_va, a=a, b=b)

    # ----- Gate init from gradient ridge of f -----
    coeffs, ts_ridge, x_peak, w_ridge = estimate_boundary_poly_from_f(
        a, b, deg=boundary_deg, Nx=300, Nt=300
    )
    gate = init_gate_from_boundary_poly(x_tr, t_tr, coeffs, gate_deg=gate_deg, sharpness=0.15)

    # Initial soft responsibilities from gate
    R = gate.proba(x_tr, t_tr)

    # Experts init
    experts: List[ExpQRational2D] = [None] * K  # type: ignore
    for k in range(K):
        experts[k] = ExpQRational2D.fit(
            x_tr, t_tr, y_tr,
            dxP, dtP, dxR, dtR,
            weights=R[:, k],
            ridgeP=ridgeP, ridgeR=ridgeR,
            iters=25, lam=1e-3, g_clip=12.0
        )

    # EM loop
    for it in range(em_iters):
        tau = tau_start + (tau_end - tau_start) * (it / max(1, em_iters - 1))
        R, _ = update_responsibilities(x_tr, t_tr, y_tr, experts, gate,
                                       tau=tau, huber_delta=huber_delta)
        for k in range(K):
            experts[k] = ExpQRational2D.fit(
                x_tr, t_tr, y_tr,
                dxP, dtP, dxR, dtR,
                weights=R[:, k],
                ridgeP=ridgeP, ridgeR=ridgeR,
                iters=25, lam=1e-3, g_clip=12.0
            )
        gate = LogisticGate.fit_soft(x_tr, t_tr, R, deg=gate_deg, l2=1e-3, iters=800, lr=0.15)

    model = MoEModel(experts=experts, gate=gate, K=K)

    # Validation metrics
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
        "boundary_coeffs": [float(c) for c in coeffs],
    }
    return model, metrics

def estimate_boundary_poly_from_f(a, b, *,
                                  deg=3, Nx=300, Nt=300,
                                  x_min=-4.0, x_max=4.0,
                                  t_min=0.0, t_max=1.0):
    """
    Find a boundary curve x_b(t) by locating, for each t-slice, the x where |∂f/∂x|
    is maximal. Fit a degree-`deg` polynomial to those ridge points (weighted by the max slope).
    Uses ONLY samples of f (gradient via finite differences on the grid).
    Returns coeffs c (length deg+1) for x_b(t) = sum c_k t^k, and the (ts, x_peak, w) used.
    """
    xs = np.linspace(x_min, x_max, Nx)
    ts = np.linspace(t_min + 1e-6, t_max - 1e-6, Nt)
    # Shapes: (Nt, Nx)
    X, T = np.meshgrid(xs, ts, indexing="xy")
    F = target_f(X, T, a=a, b=b)                   # evaluate once
    # gradients: order = (d/dt, d/dx)
    dFdt, dFdx = np.gradient(F, ts, xs)
    G = np.abs(dFdx)                                # slope along x
    # For each t row, take the x of the largest slope
    i_max = G.argmax(axis=1)
    x_peak = xs[i_max]
    w = G[np.arange(Nt), i_max]                     # weights = slope magnitude
    # Fit x_b(t) as weighted polynomial of t
    coeffs = _poly_fit_weighted_x_of_t(ts, x_peak, w, deg=deg)
    return coeffs, ts, x_peak, w

def boundary_poly_eval(coeffs, t):
    t = np.asarray(t, float)
    V = np.vander(t, N=len(coeffs), increasing=True)
    return V @ coeffs

def init_gate_from_boundary_poly(x, t, coeffs, *, gate_deg=3, sharpness=0.15):
    """
    Initialize a softmax gate from the signed distance d = x - x_b(t).
    Produces soft targets R via p1 = sigmoid(d/sharpness), p0=1-p1, then trains the gate.
    """
    xb = boundary_poly_eval(coeffs, t)
    d = x - xb
    p1 = _sigmoid_safe(d / sharpness)
    R = np.stack([1.0 - p1, p1], axis=1)            # (N,2), rows sum to 1
    return LogisticGate.fit_soft(x, t, R, deg=gate_deg, l2=1e-3, iters=800, lr=0.2)

def Phi(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    return 0.5 * (1.0 + _ERF(z / SQRT2))

def target_f(x: np.ndarray, t: np.ndarray, a: float, b: float,
             t_floor: float = 1e-15, t_ceil: float = 1.0 - 1e-15) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    t = np.clip(t, t_floor, t_ceil)
    s = np.sqrt(1.0 - t)
    z = (x - a * t - b * t * t) / s
    return Phi(z)

# ============================== Chebyshev basis =============================

def map_to_box(x: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # [-4,4]x[0,1] -> [-1,1]^2
    return (np.asarray(x, float) / 4.0, 2.0 * np.asarray(t, float) - 1.0)

def chebvander_1d(xhat: np.ndarray, deg: int) -> np.ndarray:
    xhat = np.asarray(xhat, dtype=float)
    V = np.zeros(xhat.shape + (deg + 1,), float)
    V[..., 0] = 1.0
    if deg >= 1:
        V[..., 1] = xhat
        for n in range(2, deg + 1):
            V[..., n] = 2 * xhat * V[..., n - 1] - V[..., n - 2]
    return V

def cheb_tensor(x: np.ndarray, t: np.ndarray, dx: int, dt: int) -> np.ndarray:
    xh, th = map_to_box(x, t)
    Vx = chebvander_1d(xh, dx)
    Vt = chebvander_1d(th, dt)
    feats = []
    for i in range(dx + 1):
        for j in range(dt + 1):
            feats.append(Vx[..., i] * Vt[..., j])
    return np.stack(feats, axis=-1)  # (..., (dx+1)*(dt+1))

# ============================ logits / sigmoid ==============================

def _logit_clip(y: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    y = np.asarray(y, float)
    y = np.clip(y, eps, 1.0 - eps)
    return np.log(y / (1.0 - y))

def _sigmoid_safe(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, float)
    out = np.empty_like(z)
    m = (z >= 0)
    out[m]  = 1.0 / (1.0 + np.exp(-z[m]))
    ez = np.exp(z[~m])
    out[~m] = ez / (1.0 + ez)
    return out

# ========================= Expert with Q = exp(R) ==========================

@dataclass
class ExpQRational2D:
    # g_hat(x,t) = P(x,t) * exp(-R(x,t)),  with P and R Chebyshev tensors
    dxP: int; dtP: int
    dxR: int; dtR: int
    aP: np.ndarray     # numerator coeffs (size nP)
    bR: np.ndarray     # R coeffs (size nR)
    g_clip: float = 12.0   # clamp raw logit for robustness

    @staticmethod
    def fit(x: np.ndarray, t: np.ndarray, y: np.ndarray,
            dxP: int, dtP: int, dxR: int, dtR: int,
            weights: Optional[np.ndarray] = None,
            ridgeP: float = 1e-4, ridgeR: float = 1e-4,
            iters: int = 15, lam: float = 1e-3, g_clip: float = 12.0) -> "ExpQRational2D":
        """
        Gauss-Newton / Levenberg-Marquardt fit on g = logit(y):
            minimize sum w * ( g - P * exp(-R) )^2 + ridge
        where P = ΦP aP, R = ΦR bR.
        """
        x = np.asarray(x, float); t = np.asarray(t, float)
        g = _logit_clip(np.asarray(y, float))     # N
        PhiP = cheb_tensor(x, t, dxP, dtP)       # (N, nP)
        PhiR = cheb_tensor(x, t, dxR, dtR)       # (N, nR)
        N, nP = PhiP.shape; nR = PhiR.shape[1]

        # init: R=0, solve P ~ g (ridge LS)
        if weights is not None:
            sw = np.sqrt(np.asarray(weights, float))
            A0 = sw[:, None] * PhiP
            b0 = sw * g
        else:
            sw = None
            A0 = PhiP
            b0 = g
        aP = np.linalg.solve(A0.T @ A0 + ridgeP * np.eye(nP), A0.T @ b0)
        bR = np.zeros(nR, float)

        I_P = ridgeP * np.eye(nP)
        I_R = ridgeR * np.eye(nR)

        for _ in range(iters):
            R = PhiR @ bR               # (N,)
            E = np.exp(-R)              # exp(-R)
            P = PhiP @ aP               # (N,)
            f = P * E                   # (N,)  = g_hat

            r = g - f                   # residual
            if sw is not None:
                r = sw * r

            # Jacobian blocks
            Ja = E[:, None] * PhiP                         # d f / d aP
            Jb = -(P * E)[:, None] * PhiR                  # d f / d bR
            if sw is not None:
                Ja = sw[:, None] * Ja
                Jb = sw[:, None] * Jb

            # ---- Build full (nP+nR)×(nP+nR) normal matrix correctly ----
            J = np.concatenate([Ja, Jb], axis=1)           # (N, nP+nR)
            ATA = J.T @ J + lam * np.eye(nP + nR)
            ATb = J.T @ r
            # block ridge (keeps P/R scales sane)
            ATA[:nP, :nP] += I_P
            ATA[nP:, nP:] += I_R

            delta = np.linalg.solve(ATA, ATb)
            da = delta[:nP]; db = delta[nP:]

            # simple LM line-search
            a_trial = aP + da; b_trial = bR + db
            Rtrial = PhiR @ b_trial; Etrial = np.exp(-Rtrial)
            ftrial = (PhiP @ a_trial) * Etrial
            rtrial = (g - ftrial)
            if sw is not None: rtrial = sw * rtrial

            loss_old = float(np.dot(r, r))
            loss_new = float(np.dot(rtrial, rtrial))
            if loss_new > loss_old:
                lam *= 10.0
            else:
                lam = max(lam / 2.0, 1e-6)
                aP, bR = a_trial, b_trial
                if np.linalg.norm(delta) < 1e-6 * (1.0 + np.linalg.norm(np.r_[aP, bR])):
                    break

        return ExpQRational2D(dxP, dtP, dxR, dtR, aP, bR, g_clip=g_clip)

    def eval_raw(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        P = cheb_tensor(x, t, self.dxP, self.dtP) @ self.aP
        R = cheb_tensor(x, t, self.dxR, self.dtR) @ self.bR
        g = P * np.exp(-R)
        return np.clip(g, -self.g_clip, self.g_clip)

    def eval(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        return _sigmoid_safe(self.eval_raw(x, t))


# ============================== Gate (softmax) ==============================

def poly_features(x: np.ndarray, t: np.ndarray, deg: int) -> np.ndarray:
    xh, th = map_to_box(x, t)
    feats = [np.ones_like(xh)]
    for i in range(1, deg + 1): feats.append(xh ** i)
    for j in range(1, deg + 1): feats.append(th ** j)
    for i in range(1, deg + 1):
        for j in range(1, deg + 1):
            feats.append((xh ** i) * (th ** j))
    return np.stack(feats, axis=-1)

@dataclass
class LogisticGate:
    W: np.ndarray  # (F, K)
    deg: int

    @staticmethod
    def fit_soft(x: np.ndarray, t: np.ndarray, R: np.ndarray,
                 deg: int = 3, l2: float = 1e-3, iters: int = 600, lr: float = 0.2) -> "LogisticGate":
        x = np.asarray(x); t = np.asarray(t); R = np.asarray(R, float)
        N, K = R.shape
        Phi = poly_features(x, t, deg)  # (N, F)
        F = Phi.shape[1]
        W = np.zeros((F, K), float)
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

# ============================ Mixture-of-Experts ============================

@dataclass
class MoEModel:
    experts: List[ExpQRational2D]
    gate: LogisticGate
    K: int

    def predict(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        x = np.asarray(x); t = np.asarray(t)
        idx = self.gate.predict(x, t)
        out = np.zeros_like(x, float)
        for k in range(self.K):
            m = (idx == k)
            if np.any(m):
                out[m] = self.experts[k].eval(x[m], t[m])
        return out

    def predict_soft(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        x = np.asarray(x); t = np.asarray(t)
        Pk = self.gate.proba(x, t)  # (..., K)
        out = np.zeros_like(x, float)
        for k in range(self.K):
            out += Pk[..., k] * self.experts[k].eval(x, t)
        return out

def huber_loss(x: np.ndarray, delta: float = 0.02) -> np.ndarray:
    a = np.abs(x)
    return np.where(a <= delta, 0.5*a*a, delta*(a - 0.5*delta))

def update_responsibilities(x, t, y, experts: List[ExpQRational2D], gate: LogisticGate,
                            tau: float = 0.03, huber_delta: float = 0.02, eps: float = 1e-6):
    preds = np.stack([e.eval(x, t) for e in experts], axis=1)  # (N, K)
    loss  = huber_loss(preds - y[:, None], delta=huber_delta)  # (N, K)
    G     = gate.proba(x, t)                                   # (N, K)
    R     = np.maximum(G * np.exp(-loss / tau), eps)
    R    /= R.sum(axis=1, keepdims=True)
    return R, preds

def kmeans2(xy: np.ndarray, K: int, iters: int = 20, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    N = xy.shape[0]
    centers = xy[rng.choice(N, size=K, replace=False)].copy()
    labels = np.zeros(N, dtype=int)
    for _ in range(iters):
        d2 = ((xy[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = d2.argmin(axis=1)
        for k in range(K):
            m = labels == k
            centers[k] = xy[m].mean(axis=0) if np.any(m) else xy[rng.integers(0, N)]
    return labels

def fit_moe_expQ(a: float, b: float,
                 K: int = 2,
                 N_train: int = 8000,
                 N_val: int = 16000,
                 dxP: int = 6, dtP: int = 6,
                 dxR: int = 3, dtR: int = 2,
                 ridgeP: float = 1e-4, ridgeR: float = 1e-4,
                 em_iters: int = 25,
                 gate_deg: int = 3,
                 tau_start: float = 0.06, tau_end: float = 0.02,
                 huber_delta: float = 0.02,
                 seed: int = 0) -> Tuple[MoEModel, Dict]:
    """Train a K-expert MoE with Q=exp(R) experts (no split prior)."""
    rng = np.random.default_rng(seed)
    x_tr = rng.uniform(-4.0, 4.0, N_train)
    t_tr = rng.uniform(0.0, 1.0, N_train)
    y_tr = target_f(x_tr, t_tr, a=a, b=b)

    x_va = rng.uniform(-4.0, 4.0, N_val)
    t_va = rng.uniform(0.0, 1.0, N_val)
    y_va = target_f(x_va, t_va, a=a, b=b)

    # init responsibilities via k-means
    labels0 = kmeans2(np.stack([x_tr, t_tr], axis=1), K=K, seed=seed)
    R = np.zeros((N_train, K), float)
    for k in range(K):
        m = (labels0 == k); R[m, k] = 1.0
    R /= R.sum(axis=1, keepdims=True)

    gate = LogisticGate.fit_soft(x_tr, t_tr, R, deg=gate_deg, l2=1e-3, iters=600, lr=0.2)

    experts: List[ExpQRational2D] = [None] * K  # type: ignore
    for k in range(K):
        experts[k] = ExpQRational2D.fit(
            x_tr, t_tr, y_tr,
            dxP, dtP, dxR, dtR,
            weights=R[:, k],
            ridgeP=ridgeP, ridgeR=ridgeR,
            iters=15, lam=1e-3, g_clip=12.0
        )

    for it in range(em_iters):
        tau = tau_start + (tau_end - tau_start) * (it / max(1, em_iters - 1))
        R, _ = update_responsibilities(x_tr, t_tr, y_tr, experts, gate, tau=tau, huber_delta=huber_delta)
        for k in range(K):
            experts[k] = ExpQRational2D.fit(
                x_tr, t_tr, y_tr,
                dxP, dtP, dxR, dtR,
                weights=R[:, k],
                ridgeP=ridgeP, ridgeR=ridgeR,
                iters=15, lam=1e-3, g_clip=12.0
            )
        gate = LogisticGate.fit_soft(x_tr, t_tr, R, deg=gate_deg, l2=1e-3, iters=800, lr=0.15)

    model = MoEModel(experts=experts, gate=gate, K=K)

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

# ================================== Plots ==================================

def plot_3d_pred_and_error(model: MoEModel, a: float, b: float,
                           Nx: int = 90, Nt: int = 90, soft: bool = True) -> None:
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    except Exception as e:
        print("Matplotlib not available:", e); return

    xs = np.linspace(-4.0, 4.0, Nx)
    ts = np.linspace(1e-6, 1.0 - 1e-6, Nt)
    X, T = np.meshgrid(xs, ts, indexing="xy")

    Yp = model.predict_soft(X, T) if soft else model.predict(X, T)
    Yt = target_f(X, T, a=a, b=b)
    Ye = np.abs(Yp - Yt)

    fig1 = plt.figure(figsize=(9, 6))
    ax1 = fig1.add_subplot(111, projection="3d")
    s1 = ax1.plot_surface(X, T, Yp, rstride=1, cstride=1, linewidth=0, antialiased=True, cmap="viridis")
    ax1.set_xlabel("x"); ax1.set_ylabel("t"); ax1.set_zlabel("predicted f")
    ax1.set_title("Predicted value ({})".format("soft mixture" if soft else "hard gate"))
    fig1.colorbar(s1, shrink=0.7, aspect=12)

    fig2 = plt.figure(figsize=(9, 6))
    ax2 = fig2.add_subplot(111, projection="3d")
    s2 = ax2.plot_surface(X, T, Ye, rstride=1, cstride=1, linewidth=0, antialiased=True, cmap="magma")
    ax2.set_xlabel("x"); ax2.set_ylabel("t"); ax2.set_zlabel("|error|")
    ax2.set_title("Absolute error vs true f")
    fig2.colorbar(s2, shrink=0.7, aspect=12)
    plt.show()

def plot_gate_vs_curve(model: MoEModel, a: float, b: float, Nx=500, Nt=500):
    """Just for sanity check: learned boundary vs x=a t + b t^2 (not used in training)."""
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print("Matplotlib not available:", e); return
    xs = np.linspace(-4.0, 4.0, Nx)
    ts = np.linspace(1e-6, 1.0 - 1e-6, Nt)
    X, T = np.meshgrid(xs, ts, indexing="xy")
    P = model.gate.proba(X, T)
    score = P[..., 1] - P[..., 0]
    fig, ax = plt.subplots(figsize=(7,5))
    cs = ax.contour(X, T, score, levels=[0.0], colors="k", linewidths=2)
    if cs.collections:
        cs.collections[0].set_label("learned boundary (p1 = p0)")
    ax.plot(a*ts + b*ts**2, ts, "r--", lw=2, label=r"$x=a t + b t^2$")
    ax.set_xlim(-4,4); ax.set_ylim(0,1)
    ax.set_xlabel("x"); ax.set_ylabel("t")
    ax.set_title("Learned boundary vs reference curve (for visualization only)")
    ax.legend(loc="best"); fig.tight_layout(); plt.show()

# ================================== Demo ===================================

def _demo():
    a, b = 0.7, -0.3
    model, metrics = fit_moe_expQ(
        a=a, b=b,
        K=2,
        N_train=8000, N_val=16000,
        dxP=7, dtP=6,
        dxR=3, dtR=2,
        ridgeP=1e-4, ridgeR=1e-4,
        em_iters=10, gate_deg=4,
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

    # 3D plots
    plot_3d_pred_and_error(model, a=a, b=b, Nx=90, Nt=90, soft=True)
    # boundary check (optional)
    plot_gate_vs_curve(model, a=a, b=b)


def _demo2():
    a, b = 0.7, -0.3
    model, metrics = fit_moe_expQ_gradgate(
        a=a, b=b,
        K=2,
        N_train=6000, N_val=12000,
        dxP=7, dtP=6, dxR=3, dtR=2,
        gate_deg=3, boundary_deg=3,
        em_iters=10, seed=42
    )
    print("Validation metrics (hard gate):")
    print(f"  RMSE   = {metrics['val_RMSE_hard']:.4e}")
    print(f"  max|e| = {metrics['val_max_abs_hard']:.4e}")
    print(f"  med|e| = {metrics['val_med_abs_hard']:.4e}")
    print("Validation metrics (soft mixture):")
    print(f"  RMSE   = {metrics['val_RMSE_soft']:.4e}")
    print(f"  max|e| = {metrics['val_max_abs_soft']:.4e}")
    print(f"  med|e| = {metrics['val_med_abs_soft']:.4e}")
    # your existing plots still work:
    plot_3d_pred_and_error(model, a=a, b=b, Nx=400, Nt=400, soft=True)
    plot_gate_vs_curve(model, a=a, b=b)

_demo2()
