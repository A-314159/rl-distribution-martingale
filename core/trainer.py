import time, math, json, os
from dataclasses import dataclass, asdict
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
import numpy as np
from typing import Tuple

from core.sampling import family, expand_family, SamplerConfig, cast_data_low
from core.critics import distribution_critic
from core.actors import Actor
from core.universe import UniverseBS
from core.train_config import TrainConfig

from optimizers.gnlm import GaussNewtonLM
from optimizers.optimizer import Optimizer

from utilities.tensorflow_config import tf_compile, LOW, HIGH, SENSITIVE_CALC
from utilities.decorators import requires_grad, no_grad_ok
from utilities.misc import HotKeys, to_csv, jsonable
import matplotlib.pyplot as plt
from itertools import product


class DistributionTrainer:
    def __init__(self, universe: UniverseBS, sampler_cfg: SamplerConfig, train_cfg: TrainConfig, actor: Actor):
        self.universe, self.sampler_cfg, self.train_cfg = universe, sampler_cfg, train_cfg
        self.data, self.model, self.actor = None, None, actor

        # private
        self._beta = 0
        self._batch = None

    def build(self, reload_model=False):

        parent = family(self.sampler_cfg, self.universe)
        self.data = expand_family(parent, self.universe)
        # warning: below no fully test yet as we ran on FP64
        cast_data_low(self.data, no_cast_keys=['t', 'sqrt_tau', 'dS', 'pv_kids', 'Y_hint'])

        # Model + adapt norm
        self.model = distribution_critic(4, self.train_cfg.hidden, self.train_cfg.activation)
        feats_null = tf.stack([self.data["t"], self.data["x"], self.data["y"], self.data["sqrt_tau"]], axis=1)
        self.model.get_layer("norm").adapt(feats_null)

        out_dir = Path(self.train_cfg.model_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        with open(out_dir / "hyperparams.json", "w") as f:
            #json.dump({"universe": jsonable(self.universe), "actor": jsonable(self.actor),
            #           "sampler": asdict(self.sampler_cfg), "train": jsonable(self.train_cfg)}, f, indent=2)
            json.dump({
                "universe": self.universe.get_config(),
                "actor": self.actor.get_config(),
                "sampler": asdict(self.sampler_cfg),
                "train": self.train_cfg.get_config()
            }, f, indent=2)
        return self.model

    def make_errors_function(self, data, detailed=False):
        universe = self.universe

        @tf_compile
        def residuals() -> tf.Tensor:
            r, _, _ = details()
            return r

        @tf_compile
        def gather():
            keys = ['t', 'sqrt_tau', 'x', 'x_lo', 'x_hi', 'y', 'y_lo', 'y_hi', 't_kids', 'sqrt_tau_kids',
                    'x_kids', 'w_kids', 'terminal', 'dS', 'pv_kids', 'Y_hint']
            values = []
            for k in keys:
                val = data[k] if self._batch is None else tf.gather(data[k], self._batch)
                values.append(val)
            return tuple(values)

        @tf_compile
        @requires_grad
        def details() -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
            t, sqrt_tau, x, x_lo, x_hi, y, y_lo, y_hi, t_c, sqrt_tau_c, x_c, w_c, term, dS, pv_c, Y_hint = gather()

            nb_kids = tf.shape(x_c)[1]
            t_c = tf.repeat(t_c, repeats=nb_kids, axis=1)
            sqrt_tau_c = tf.repeat(sqrt_tau_c, repeats=nb_kids, axis=1)

            # This should be _done in FP32 at least
            q = self.actor(t, sqrt_tau, universe=universe)
            l_prime = -q[:, None] * dS - tf.where(term, pv_c, 0)
            y_c = y[:, None] - l_prime

            # Compute model in LOW precision and cast output to HIGH
            parent = tf.cast(tf.stack([t, x, y, sqrt_tau], axis=1), LOW)
            child = tf.cast(tf.stack([t_c, x_c, y_c, sqrt_tau_c], axis=-1), LOW)
            child_flat = tf.reshape(child, (-1, 4))
            fused = tf.concat([parent, child_flat], axis=0)

            sensitive_type = SENSITIVE_CALC if self.train_cfg.cast_64 else HIGH
            F = tf.cast(tf.squeeze(self.model(fused, training=True), -1), sensitive_type)

            # Split F at parent and children
            F, F_c = F[:tf.shape(parent)[0]], tf.reshape(F[tf.shape(parent)[0]:], tf.shape(x_c))

            # Boundaries for degeneracy are in LOW, distributions are in HIGH

            # Degeneracy of F versus y
            one, zero = tf.ones_like(F_c), tf.ones_like(F_c)
            F_c = tf.where(y_c <= y_lo[:, None], zero, tf.where(y_c >= y_hi[:, None], one, F_c))

            # Degeneracy of F versus x
            F_degenerate_x = tf.where(y_c > -tf.cast(pv_c, LOW), one, zero)
            F_c = tf.where(tf.logical_or(x_c >= x_hi[:, None], x_c <= x_lo[:, None]), F_degenerate_x, F_c)

            # Degeneracy of F at T
            F_degenerate_T = tf.where(y_c >= 0, one, zero)
            F_c = tf.where(term, F_degenerate_T, F_c)

            # Blend with hint
            Y_bellman = tf.reduce_sum(tf.cast(w_c, sensitive_type) * F_c, axis=1)
            Y = self._beta * Y_bellman + (1.0 - self._beta) * Y_hint
            r = F - Y
            return r, F, Y

        return details if detailed else residuals

    def train(self):
        out_dir = Path(self.train_cfg.model_dir)
        csv_path = out_dir / self.train_cfg.log_csv
        to_csv(csv_path, "w", ["epoch", "elapsed_sec", "rmse_loss", "beta_blending"])
        model, universe, cfg, data = self.model, self.universe, self.train_cfg, self.data

        opt = Optimizer(cfg, model, self.make_errors_function(data))

        full_batch = opt.require_full_batch or cfg.full_batch
        epoch, t0, hot, N = 0, time.time(), HotKeys(), int(data["t"].shape[0])
        while epoch < cfg.max_epochs:

            # self._beta = tf.cast(min(epoch / cfg.anneal_beta_period, 1.0), HIGH)
            self._beta = 0.0
            losses = []
            if full_batch:
                self._batch = None
                r = opt.step()
                losses.append(r['f'].numpy())
            else:
                perm = tf.random.shuffle(tf.range(N))
                for start in range(0, N, cfg.batch_size):
                    self._batch = perm[start:min(start + cfg.batch_size, N)]
                    losses.append(opt.step())

            if hot.c: self.chart(t=np.array([0, universe.T - universe.h]), show_chart=cfg.show_chart)

            rmse, elapsed = math.sqrt(sum(losses) / max(1, len(losses))), time.time() - t0
            print(f"Epoch {epoch:04d}  rmse={rmse:.6e}  beta={self._beta:.3f}  elapsed={elapsed:.1f}s")
            to_csv(csv_path, "a", [epoch, f"{elapsed:.3f}", f"{rmse:.8e}", f"{self._beta:.6f}"])

            if epoch % 500 == 0: model.save(out_dir / "model.keras")
            if hot.q or time.time() - t0 > cfg.max_time_sec or rmse < cfg.loss_tol_sqrt:
                print(f"Termination either by user, tolerance, or max time")
                break
            epoch += 1

        model.save(out_dir / "model.keras")
        return model

    def chart(self, t, x=None, show_chart=True):
        N, cushion_y = 10001, 0.05
        if x is None:
            x = np.array([-0.3, -0.1, 0, 0.1, 0.2, 0.3])
        y = np.linspace(-max(math.exp(x.max()) - 1, 0) - cushion_y, -max(math.exp(x.min()) - 1, 0) + cushion_y, N)
        dy = (y.max() - y.min()) / (N - 1)

        tp = tf.keras.backend.floatx()
        _t, _x, _y = np.meshgrid(np.array(t), x, y, indexing='ij')
        _t = tf.cast(tf.convert_to_tensor(_t.flatten()), tp)
        _x = tf.cast(tf.convert_to_tensor(_x.flatten()), tp)
        _y = tf.cast(tf.convert_to_tensor(_y.flatten()), tp)

        mums = family(self.sampler_cfg, self.universe, txy={'t': _t, 'x': _x, 'y': _y})
        families = expand_family(mums, self.universe)
        self._batch = None
        res_fn = self.make_errors_function(families, detailed=True)
        e, F, Y = res_fn()
        e = e.numpy().reshape(t.size, x.size, y.size)
        F = F.numpy().reshape(t.size, x.size, y.size)
        T = Y.numpy().reshape(t.size, x.size, y.size)
        mu = families['y_mu'].numpy().reshape(t.size, x.size, y.size)

        f = (F[:, :, 2:] - F[:, :, :-2]) / dy

        lw = [1, 0.5]
        ls = ['-', '-']
        fig, ax = plt.subplots(2, 2, figsize=(12.8, 9.6))
        fig.suptitle(' ')

        col = np.array([[0, 0], [0, 0]], dtype=object)
        r_e, r_f, r_F = np.array([np.min(e), np.max(e)]), np.array([np.min(f), np.max(f)]), np.array([0, 1])
        rg = np.array([[r_F, r_f], [r_F, r_e]])
        _y = [[y, y[1:-1]], [y, y]]
        data = [[F, f], [T, e]]
        for r, c in product(range(2), repeat=2):

            for j in range(len(x)):
                for i in range(len(t)):
                    #if x[j] < x_min_sampling[i, j, 0] or x[j] > x_max_sampling[i, j, 0]: continue

                    mu_x = np.array([mu[i, j, 0], mu[i, j, 0]])
                    d = data[r][c][i, j, :]
                    if i == 0:
                        line, = ax[r, c].plot(_y[r][c], d, label='x=%.1f' % (x[j]), linewidth=lw[i],
                                              linestyle=ls[i])
                        col[r, c] = line.get_color()
                    else:
                        ax[r, c].plot(_y[r][c], d, color=col[r, c], linewidth=lw[i], linestyle=ls[i])
                    ax[r, c].plot(mu_x, rg[r, c], color=col[r, c], linestyle='--', linewidth=lw[i])

        titles = np.array([
            [r'Distribution at $t=%.2f$, $t=%0.2f$' % (t[0], t[1]), r'Density at $t=%.2f$, $t=%0.2f$' % (t[0], t[1])],
            [r'Target at $t=%.2f$, $t=%0.2f$' % (t[0], t[1]), r'Error on $F$ at $t=%.2f$, $t=%0.2f$' % (t[0], t[1])]])
        y_labels = np.array([[r'$F(y)$', r'$f(y)$'], [r'$T(y)$', r'$F(y)-T(y)$']])
        for r, c in product(range(2), repeat=2):
            ax[r, c].set_title(titles[r, c])
            ax[r, c].set(xlabel=r'$y$', ylabel=y_labels[r, c])
            ax[r, c].grid(True, color='silver', linestyle='--', linewidth=0.5)

        lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
        lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
        unique = [(h, l) for i, (h, l) in enumerate(zip(lines, labels)) if l not in labels[:i]]
        fig.legend(*zip(*unique), ncol=len(unique), loc='upper center', fontsize='x-small')
        fig.tight_layout()

        if not show_chart:
            out_dir = Path(self.train_cfg.model_dir)
            chart_path = out_dir / self.train_cfg.chart_pdf
            if os.path.isfile(chart_path): os.remove(chart_path)
            plt.savefig(chart_path, format='pdf', dpi=600)
            plt.close()
        else:
            plt.show()
