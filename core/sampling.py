from dataclasses import dataclass
import tensorflow as tf
from core.bs import bs_call_price
from utilities.misc import cast_all, cdf
from utilities.tensorflow_config import tf_compile
from core.universe import UniverseBS


@dataclass
class SamplerConfig:
    N: int = 60000
    x0: float = 0.0
    a: float = 0.3
    b: float = 4.0  # multiple of deviation for x
    c: float = 1 / 52  # cushion for residual maturity
    r0: float = 0.02
    r1: float = 0.002


@tf_compile
def family(cfg: SamplerConfig, u, **kwargs) -> dict:
    """
    return a dictionary with random parent samples if kwargs is not used
    :param cfg:
    :param u:
    :param kwargs: if not empty,
    :return:
    """
    N = cfg.N
    rng = tf.random
    tp = tf.keras.backend.floatx()
    # todo: put the casting in the universe and configuration init instead
    (T, h, K, sigma, x0, a, b, c, r0, r1) = (
        cast_all(u.T, u.h, u.K, u.sigma, cfg.x0, cfg.a, cfg.b, cfg.c, cfg.r0, cfg.r1, dtype=tp))

    # -----------------------------
    # draw time
    # -----------------------------
    if 't' in kwargs:
        t = kwargs['t']
        k = tf.cast(t, tf.int32)
    else:
        k = rng.uniform(shape=(N,), minval=0, maxval=u.P, dtype=tf.int32)
    k = tf.cast(k, tp)
    t = k * h
    tau = T - t
    sqrt_tau = tf.sqrt(tau)

    # -----------------------------
    # draw x=log S
    # -----------------------------
    # degeneracy bounds
    center, band = tf.math.log(K) + 0.5 * (sigma ** 2) * tau, b * sigma * tf.sqrt(tau + c)
    x_hi, x_lo = center + band, center - band

    # bounds for expanding tree
    dlin = sigma * tf.sqrt(h) * k

    # sampling bounds
    x_hi_s = tf.minimum(x0 + a + dlin, x_hi)
    x_lo_s = tf.maximum(x0 - a - dlin, x_lo)

    if 'x' in kwargs:
        x = kwargs['x']
    else:
        u = rng.uniform(shape=(N,), minval=0., maxval=1., dtype=tp)
        x = x_lo_s + u * (x_hi_s - x_lo_s)

    # -----------------------------
    # draw y
    # -----------------------------
    y_mu = -bs_call_price(x, sigma * sqrt_tau, K)
    y_half = r0 + t / T * (r1 - r0)
    y_hi = y_mu + y_half
    y_lo = y_mu - y_half

    if 'y' in kwargs:
        y = kwargs['y']
    else:
        u = rng.uniform(shape=(N,), minval=0., maxval=1., dtype=tp)
        y = y_lo + u * (y_hi - y_lo)

    # -----------------------------
    # prepare hint for the distribution
    # -----------------------------
    Y_hint = cdf((y - y_mu) / y_half)
    return dict(
        t=t, sqrt_tau=sqrt_tau,
        x=x, x_hi=x_hi, x_lo=x_lo,
        y=y, y_lo=y_lo, y_hi=y_hi, y_mu=y_mu,
        Y_hint=Y_hint
    )


@tf_compile
def expand_family(parents: dict, universe: UniverseBS) -> dict:
    t = parents["t"]
    x = parents["x"]
    sigma, T, h, P, K = universe.sigma, universe.T, universe.h, universe.P, universe.K

    x_kids, w_kids = universe.children(x, sigma * tf.math.sqrt(h))
    t_kids = tf.expand_dims(t + h, axis=-1)

    tau_kids = T - t_kids
    sqrt_tau_kids = tf.sqrt(tf.maximum(tau_kids, 0))
    terminal = tf.abs(t_kids - T) < 1e-6

    dS = tf.exp(x_kids) - tf.exp(x)[:, None]
    pv_kids = bs_call_price(x_kids, sigma * sqrt_tau_kids, K)

    parents.update(dict(
        x_kids=x_kids, w_kids=w_kids, t_kids=t_kids, terminal=terminal,
        dS=dS, pv_kids=pv_kids, tau_kids=tau_kids, sqrt_tau_kids=sqrt_tau_kids
    ))
    return parents
