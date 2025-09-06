from dataclasses import dataclass
import tensorflow as tf
from core.bs import bs_call_price
from utilities.misc import cast_all, cdf
from utilities.tensorflow_config import tf_compile, LOW, HIGH, SENSITIVE_CALC
from core.universe import UniverseBS


@dataclass
class SamplerConfig:
    N: int = 60000
    x0: float = 0.0
    a: float = 0.3  # range around [x0: x0-a, x0+a]
    b: float = 4.0  # multiple of deviation for x
    c: float = 1 / 52  # cushion for residual maturity
    r0: float = 0.02
    r1: float = 0.002


@tf_compile
def family(cfg: SamplerConfig, u, txy=None) -> dict:
    """
    return a dictionary with random parent samples if kwargs is not used
    :param cfg:
    :param u:
    :param kwargs: if not empty,
    :return:
    """
    N = cfg.N
    rng = tf.random
    dtype = HIGH
    # todo: put the casting in the configuration init instead
    h, T, sigma, K = u.h, u.T, u.sigma, u.K
    (x0, a, b, c, r0, r1) = cast_all(cfg.x0, cfg.a, cfg.b, cfg.c, cfg.r0, cfg.r1, dtype=dtype)

    usable_dict = isinstance(txy, dict)
    if usable_dict:
        usable_dict = isinstance(txy, dict) and 't' in txy and 'x' in txy and 'y' in txy
        if not usable_dict:
            raise Exception("txy is not a dictionary in family or does not include all the keys 't', 'x', 'y'")

    # -----------------------------
    # draw time
    # -----------------------------
    if usable_dict:
        t = txy['t']
        k = tf.cast(t/h, tf.int32)
    else:
        k = rng.uniform(shape=(N,), minval=0, maxval=u.P, dtype=tf.int32)
    k = tf.cast(k, dtype)
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

    if usable_dict:
        x = txy['x']
    else:
        u = rng.uniform(shape=(N,), minval=0., maxval=1., dtype=dtype)
        x = x_lo_s + u * (x_hi_s - x_lo_s)

    # -----------------------------
    # draw y
    # -----------------------------
    y_mu = -bs_call_price(x, sigma * sqrt_tau, K)
    y_half = r0 + t / T * (r1 - r0)
    y_hi = y_mu + y_half
    y_lo = y_mu - y_half

    if usable_dict:
        y = txy['y']
    else:
        u = rng.uniform(shape=(N,), minval=0., maxval=1., dtype=dtype)
        y = y_lo + u * (y_hi - y_lo)

    # -----------------------------
    # prepare hint for the distribution
    # -----------------------------
    Y_hint = cdf(tf.cast((y - y_mu) / y_half, SENSITIVE_CALC))

    # -----------------------------
    # To save memory:
    # a) store only the variables that are needed or that would take really too much time to recompute at each epoch.
    # b) cast to LOW precision variables that will not require HIGH precision.
    # -----------------------------
    return dict(
        t=t, sqrt_tau=sqrt_tau, y_mu=tf.cast(y_mu,LOW),
        x=x, x_hi=tf.cast(x_hi, LOW), x_lo=tf.cast(x_lo, LOW),
        y=y, y_lo=tf.cast(y_lo, LOW), y_hi=tf.cast(y_hi, LOW),
        Y_hint=Y_hint)


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
        x_kids=tf.cast(x_kids, LOW), w_kids=w_kids, t_kids=tf.cast(t_kids, LOW), terminal=terminal,
        dS=dS, pv_kids=pv_kids, sqrt_tau_kids=tf.cast(sqrt_tau_kids, LOW)
    ))
    return parents


def cast_data_low(data_dictionary, no_cast_keys):
    for k, v in data_dictionary.items():
        if k not in no_cast_keys:
            if v.dtype in (HIGH, SENSITIVE_CALC): data_dictionary[k] = tf.cast(v, LOW)
