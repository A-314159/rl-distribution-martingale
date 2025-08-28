import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
from core.bs import bs_call_price, bs_delta


def mc_cdf(u, t_idx: int, x0: float, y_grid: np.ndarray, paths: int):
    h = u.h
    steps = u.P - t_idx
    if steps <= 0: return (y_grid > 0).astype(float)
    rng = np.random.default_rng(12345 + t_idx)
    S0 = np.exp(x0)
    ys = np.zeros(paths)
    for p in range(paths):
        S = S0
        y_cum = 0.0
        for k in range(steps):
            tau = u.T - (t_idx + k) * h
            q = - float(bs_delta(S, u.K, u.sigma, tau).numpy())
            z = rng.standard_normal()
            S_next = S * math.exp((-0.5 * u.sigma ** 2) * h + u.sigma * math.sqrt(h) * z)
            payoff = max(S_next - u.K, 0.0) if (t_idx + k + 1) == u.P else 0.0
            y_cum += -q * (S_next - S) - payoff
            S = S_next
        ys[p] = y_cum
    ys.sort()
    return np.searchsorted(ys, y_grid, side="right") / float(paths)


def make_and_save_chart(model, u, cfg, out_pdf: Path):
    plt.figure(figsize=(7, 5))
    for (t_idx, x_val) in (cfg.eval_pairs or []):
        tau = u.T - t_idx * u.h
        mu = - bs_call_price(np.exp(x_val), u.K, u.sigma, tau).numpy()
        y_half = 0.02
        y_grid = np.linspace(mu - 2 * y_half, mu + 2 * y_half, 201)
        sqrt_tau = np.sqrt(max(tau, 0.0))
        feats = np.stack([np.full_like(y_grid, t_idx),
                          np.full_like(y_grid, x_val),
                          y_grid,
                          np.full_like(y_grid, sqrt_tau)], axis=1)
        feats = feats.astype(
            np.float16 if is_mixed() else (np.float64 if tf.keras.backend.floatx() == "float64" else np.float32))
        predictions = model(tf.convert_to_tensor(feats, dtype=model.input.dtype), training=False).numpy().squeeze(-1)
        mc = mc_cdf(u, t_idx, x_val, y_grid, cfg.mc_paths)
        plt.plot(y_grid, predictions, label=f"Model t={t_idx} x={x_val:.2f}")
        plt.plot(y_grid, mc, "--", label=f"MC t={t_idx} x={x_val:.2f}")
    plt.xlabel("y")
    plt.ylabel("CDF F(t,x,y)")
    plt.title("F critic vs Monte Carlo")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_pdf)
    plt.close()
