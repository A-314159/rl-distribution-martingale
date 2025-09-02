# --------------------------------------------------------------------------
# 'eager' version w/out graph and @tf_function
# c.f. explanations in https://chatgpt.com/share/68b47ba4-69e0-800a-891a-56aba89e89fd
# --------------------------------------------------------------------------
"""
TensorFlow L-BFGS(D) optimizer with:
  - Two line searches: Nonmonotone Armijo (Grippo-Lampariello-Lucidi) and Hager–Zhang strong Wolfe
  - Adaptive initial scaling (Barzilai–Borwein style) or direction-matching scaling
  - Pair quality metrics and pruning; optional simple aggregation
  - Optional Powell damping (approximate, inverse-form friendly)
  - Rich per-iteration metrics useful for contextual bandits
  - GPU-friendly: keeps S,Y and two-loop on the same device as x/g

    Author: ChatGPT

"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Any
import tensorflow as tf
import time, math
from utilities.tensorflow_config import tf_compile

"""
If a method is decorated by @tf_function with the intent to run it as graph, 
applying tf.config.run_functions_eagerly(True) forces to run the method eagerly.

If a method is not decorated by @tf_function, it will run eagerly whether or not  we apply
tf.config.run_functions_eagerly(False)

Methods that compute tensors but use python-style controls (e.g. if x>0) 
can not work as graph because the condition is only traced at decoration time, and not executed at runtime.
The condition wont depend on the actual value during execution.
Therefore, it is important to not decorate them with @tf_function.
Python-style controls break automatic differentiation and should only be used when there is no need for it.

Here, the cost of setting LBFGS direction and of line search is small 
compared with the computation of the function f and its gradient, 
therefore it is better to not decorate them as graph (and de facto run them eagerly), 
so as to keep controls (on curvature or else) in plain python-style: easier debug and better readability. 

However, this means that there are a lot of synch between CPU and GPU 
even if the loss function is computed entirely on GPU with graph. 
This can slow down training on GPU by a factor of 10 to 100.

Conclusion: for debugging: use this version. for training: use the graph version of lbfgs_graph.py

"""


def dot(a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
    return tf.tensordot(a, b, axes=1)


def norm(a: tf.Tensor) -> tf.Tensor:
    return tf.sqrt(tf.maximum(tf.constant(0., a.dtype), dot(a, a)))


@dataclass
class LineSearchResult:
    alpha: float
    f: float
    g: tf.Tensor
    evals: int
    backtracks: int
    success: bool
    reason: str = ""


class LineSearchBase:
    def __init__(self, dtype=tf.float64):
        self.dtype = dtype

    def search(self, x, f, g, d, loss_and_grad, alpha0) -> LineSearchResult:
        raise NotImplementedError


class NonmonotoneArmijo(LineSearchBase):
    def __init__(self, c1=1e-4, window=5, backtrack=0.5, max_evals=6, dtype=tf.float64):
        super().__init__(dtype)
        self.c1 = float(c1)
        self.window = int(window)
        self.backtrack = float(backtrack)
        self.max_evals = int(max_evals)
        self.f_hist: List[float] = []

    def set_history(self, f_hist: List[float]):
        self.f_hist = f_hist

    def search(self, x, f, g, d, loss_and_grad, alpha0) -> LineSearchResult:
        gTd = float(dot(g, d).numpy())
        if gTd >= 0.0:
            return LineSearchResult(0.0, float(f.numpy()), g, 0, 0, False, "Non-descent direction")
        alpha = float(alpha0)
        backtracks, evals = 0, 0
        f_ref = max(self.f_hist[-self.window:]) if self.f_hist else float(f.numpy())
        while True:
            x_try = x + alpha * d
            f_try, g_try = loss_and_grad(x_try)
            f_try_val = float(f_try.numpy())
            evals += 1
            if f_try_val <= f_ref + self.c1 * alpha * gTd:
                return LineSearchResult(alpha, f_try_val, g_try, evals, backtracks, True, "accepted")
            alpha *= self.backtrack
            backtracks += 1
            if evals >= self.max_evals:
                return LineSearchResult(alpha, f_try_val, g_try, evals, backtracks, True, "eval_cap")


class HagerZhang(LineSearchBase):
    def __init__(self, c1=1e-4, c2=0.9, amax=1.0, max_evals=20, dtype=tf.float64):
        super().__init__(dtype)
        self.c1, self.c2, self.amax, self.max_evals = float(c1), float(c2), float(amax), int(max_evals)

    def _phi(self, x, d, a, loss_and_grad):
        x_try = x + a * d
        f_try, g_try = loss_and_grad(x_try)
        return float(f_try.numpy()), g_try

    def search(self, x, f, g, d, loss_and_grad, alpha0) -> LineSearchResult:
        f0, g0, g0Td = float(f.numpy()), g, float(dot(g, d).numpy())
        if g0Td >= 0.0:
            return LineSearchResult(0.0, f0, g, 0, 0, False, "Non-descent direction")
        a0, a1 = 0.0, min(self.amax, float(alpha0))
        f_a0, g_a0 = f0, g0
        evals, backtracks = 0, 0
        f_a1, g_a1 = self._phi(x, d, a1, loss_and_grad);
        evals += 1
        while True:
            if (f_a1 > f0 + self.c1 * a1 * g0Td) or (evals > 1 and f_a1 >= f_a0):
                alpha, f_final, g_final, evals_zoom, bts = self._zoom(x, d, a0, a1, f_a0, f_a1,
                                                                      g0Td, f0, loss_and_grad)
                return LineSearchResult(alpha, f_final, g_final, evals + evals_zoom, backtracks + bts, True, "zoom")
            g_a1Td = float(dot(g_a1, d).numpy())
            if abs(g_a1Td) <= -self.c2 * g0Td:
                return LineSearchResult(a1, f_a1, g_a1, evals, backtracks, True, "accepted")
            if g_a1Td >= 0:
                alpha, f_final, g_final, evals_zoom, bts = self._zoom(x, d, a1, a0, f_a1, f_a0,
                                                                      g0Td, f0, loss_and_grad)
                return LineSearchResult(alpha, f_final, g_final, evals + evals_zoom, backtracks + bts, True, "zoom")
            a2 = min(self.amax, 2.0 * a1)
            a0, f_a0, a1 = a1, f_a1, a2
            f_a1, g_a1 = self._phi(x, d, a1, loss_and_grad);
            evals += 1
            if evals >= self.max_evals:
                return LineSearchResult(a1, f_a1, g_a1, evals, backtracks, True, "eval_cap")

    def _zoom(self, x, d, alo, ahi, flo, fhi, g0Td, f0, loss_and_grad):
        evals, backtracks = 0, 0
        while True:
            aj = 0.5 * (alo + ahi)
            f_aj, g_aj = self._phi(x, d, aj, loss_and_grad);
            evals += 1
            if (f_aj > f0 + self.c1 * aj * g0Td) or (f_aj >= flo):
                ahi, fhi = aj, f_aj
            else:
                g_ajTd = float(dot(g_aj, d).numpy())
                if abs(g_ajTd) <= -self.c2 * g0Td:
                    return aj, f_aj, g_aj, evals, backtracks
                if g_ajTd * (ahi - alo) >= 0: ahi, fhi = alo, flo
                alo, flo = aj, f_aj
            backtracks += 1
            if evals >= self.max_evals:
                return aj, f_aj, g_aj, evals, backtracks


@dataclass
class LBFGSConfig:
    m: int = 20
    line_search: str = 'nonmonotone_armijo'
    armijo_c1: float = 1e-4
    armijo_window: int = 5
    backtrack_factor: float = 0.5
    max_evals_per_iter: int = 6
    powell_damping: bool = True
    pair_quality_min_cos: float = 1e-8
    prune_by_quality: bool = True
    proximity_filter: bool = False
    proximity_c: float = 10.0
    proximity_window: int = 20
    init_scaling: str = 'bb'
    init_gamma: float = 1.0
    aggregate: bool = False
    aggregate_cos: float = 0.995
    verbose: bool = False


class LBFGS:
    def __init__(self, **kwargs):
        self.cfg = LBFGSConfig(**kwargs)
        if self.cfg.line_search == 'nonmonotone_armijo':
            self.ls = NonmonotoneArmijo(c1=self.cfg.armijo_c1, window=self.cfg.armijo_window,
                                        backtrack=self.cfg.backtrack_factor, max_evals=self.cfg.max_evals_per_iter)
        elif self.cfg.line_search == 'hager_zhang':
            self.ls = HagerZhang(c1=self.cfg.armijo_c1, c2=0.9, amax=1.0,
                                 max_evals=max(10, self.cfg.max_evals_per_iter * 2))
        else:
            raise ValueError("line_search must be 'nonmonotone_armijo' or 'hager_zhang'")
        self.S, self.Y, self.x_hist, self.f_hist = [], [], [], []
        self.alpha_prev = 1.0

    def two_loop(self, g: tf.Tensor, gamma: float) -> tf.Tensor:
        S, Y = self.S, self.Y
        q = tf.identity(g)
        alphas, rhos = [], []
        for i in range(len(S) - 1, -1, -1):
            si, yi = S[i], Y[i]
            rhoi = 1.0 / float(dot(yi, si).numpy())
            ai = rhoi * float(dot(si, q).numpy())
            alphas.append(ai)
            rhos.append(rhoi)
            q = q - ai * yi
        r = gamma * q
        alphas, rhos = alphas[::-1], rhos[::-1]
        for i in range(len(S)):
            si, yi = S[i], Y[i]
            bi = rhos[i] * float(dot(yi, r).numpy())
            r = r + si * (alphas[i] - bi)
        return r

    def initial_gamma(self, g: tf.Tensor, d_prev: Optional[tf.Tensor]) -> float:
        if self.cfg.init_scaling == 'constant' or len(self.S) == 0:
            return float(self.cfg.init_gamma)
        if self.cfg.init_scaling == 'bb':  # standard Barzilai–Borwein formula
            s, y = self.S[-1], self.Y[-1]
            yTy = float(dot(y, y).numpy())
            if yTy <= 0: return float(self.cfg.init_gamma)
            return max(1e-12, min(1e12, float(dot(s, y).numpy()) / yTy))
        if self.cfg.init_scaling == 'direction_match' and d_prev is not None:
            gTd, g2 = float(dot(g, d_prev).numpy()), float(dot(g, g).numpy())
            if g2 <= 0: return float(self.cfg.init_gamma)
            return max(1e-12, min(1e12, -gTd / g2))
        return float(self.cfg.init_gamma)

    def powell_damp(self, s, y, gamma):
        sTy = float(dot(s, y).numpy())
        sBs = float((1.0 / max(1e-12, gamma)) * dot(s, s).numpy())
        if sTy >= 0.2 * sBs: return y
        theta = 0.8 * sBs / max(1e-12, (sBs - sTy))
        Bs = (1.0 / max(1e-12, gamma)) * s
        return theta * y + (1 - theta) * Bs

    @staticmethod
    def pair_quality(s, y):
        sTy, denom = float(dot(s, y).numpy()), float(norm(s).numpy()) * float(norm(y).numpy())
        if denom == 0: return 0.0
        return (sTy / denom) ** 2

    def proximity_ok(self, xk, xi, s_rms):
        if not self.cfg.proximity_filter or s_rms <= 0: return True
        dist = float(norm(xk - xi).numpy())
        return dist <= self.cfg.proximity_c * s_rms

    def maybe_aggregate(self):
        if not self.cfg.aggregate or len(self.S) < 2: return
        i, j = 0, 1
        s1, s2 = self.S[i], self.S[j]
        c = float(dot(s1, s2).numpy()) / (max(1e-12, float(norm(s1).numpy()) * float(norm(s2).numpy())))
        if c >= self.cfg.aggregate_cos and float(dot(self.Y[i], self.S[i]).numpy()) > 0 and float(
                dot(self.Y[j], self.S[j]).numpy()) > 0:
            self.S[i], self.Y[i] = s1 + s2, self.Y[i] + self.Y[j]
            del self.S[j]
            del self.Y[j]
            if self.cfg.verbose: print(f"[aggregate] merged two oldest pairs; cos={c:.6f}")

    def minimize(self, loss_and_grad, x: tf.Variable, max_iters=1000, tol_grad=1e-6, target_loss=None, callback=None):
        if isinstance(self.ls, NonmonotoneArmijo): self.ls.set_history(self.f_hist)
        self.S.clear()
        self.Y.clear()
        self.x_hist.clear()
        self.f_hist.clear()
        self.alpha_prev = 1.0
        f, g = loss_and_grad(x)
        f_val = float(f.numpy())
        g_norm = float(norm(g).numpy())
        self.f_hist.append(f_val)
        self.x_hist.append(tf.identity(x))
        history = {k: [] for k in ['f', 'g_norm', 'alpha', 'evals', 'backtracks', 'cos_dir', 'm', 'sTy', 'pair_quality',
                                   'accepted_pairs', 'skipped_pairs']}
        total_evals = 1
        d_prev = None
        for k in range(max_iters):
            if (target_loss is not None and f_val <= target_loss) or g_norm <= tol_grad: break
            gamma = self.initial_gamma(g, d_prev)
            d = -self.two_loop(g, gamma)
            cos_dir = 0.0
            if d_prev is not None:
                denom = max(1e-12, float(norm(d_prev).numpy()) * float(norm(d).numpy()))
                cos_dir = float(dot(d_prev, d).numpy()) / denom
            alpha0 = min(1.0, 2.0 * self.alpha_prev)
            ls_res = self.ls.search(x, f, g, d, loss_and_grad, alpha0)
            total_evals += ls_res.evals
            alpha, backtracks = ls_res.alpha, ls_res.backtracks
            x_new = x + alpha * d
            f_new, g_new = ls_res.f, ls_res.g
            s, y = x_new - x, g_new - g
            sTy = float(dot(s, y).numpy())
            s_norm = float(norm(s).numpy())
            y_norm = float(norm(y).numpy())
            accepted_pair, quality = False, 0.0
            eps_cos = self.cfg.pair_quality_min_cos
            if s_norm > 0 and y_norm > 0 and sTy > eps_cos * s_norm * y_norm:
                y_used = self.powell_damp(s, y, gamma) if self.cfg.powell_damping else y
                if float(dot(s, y_used).numpy()) > 0.0:
                    rms_window = min(len(self.S), self.cfg.proximity_window)
                    s_rms = math.sqrt(sum(float(norm(self.S[t]).numpy()) ** 2 for t in range(len(self.S) - rms_window,
                                                                                             len(self.S))) / rms_window) if rms_window > 0 else s_norm
                    if self.proximity_ok(x_new, self.x_hist[-1], s_rms):
                        self.S.append(tf.identity(s))
                        self.Y.append(tf.identity(y_used))
                        self.x_hist.append(tf.identity(x_new))
                        accepted_pair = True
                        quality = self.pair_quality(s, y_used)
                        if self.cfg.aggregate: self.maybe_aggregate()
                        while len(self.S) > self.cfg.m:
                            if self.cfg.prune_by_quality and len(self.S) > 1:
                                qualities = [self.pair_quality(self.S[i], self.Y[i]) for i in range(len(self.S))]
                                idx = min(range(len(self.S)), key=lambda i: qualities[i])
                                del self.S[idx]
                                del self.Y[idx]
                                del self.x_hist[idx + 1]
                            else:
                                del self.S[0]
                                del self.Y[0]
                                del self.x_hist[1]
            f_val, g, x = float(f_new), g_new, tf.Variable(x_new)
            g_norm = float(norm(g).numpy())
            self.f_hist.append(f_val)
            self.alpha_prev = alpha
            d_prev = tf.identity(d)
            history['f'].append(f_val)
            history['g_norm'].append(g_norm)
            history['alpha'].append(alpha)
            history['evals'].append(ls_res.evals)
            history['backtracks'].append(backtracks)
            history['cos_dir'].append(cos_dir)
            history['m'].append(len(self.S))
            history['sTy'].append(sTy)
            history['pair_quality'].append(quality);
            history['accepted_pairs'].append(1 if accepted_pair else 0)
            history['skipped_pairs'].append(0 if accepted_pair else 1)
            if self.cfg.verbose and (k % 10 == 0 or k < 5):
                print(
                    f"[iter {k:5d}] f={f_val:.6e} |g|={g_norm:.3e} alpha={alpha:.2e} evals={ls_res.evals} back={backtracks} m={len(self.S)} cos(d,d_prev)={cos_dir:.3f}")
        return {'x': x, 'f': f_val, 'g_norm': g_norm, 'iters': len(history['f']), 'total_evals': total_evals,
                'history': history}
