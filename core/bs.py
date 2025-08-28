import tensorflow as tf
from utilities.tensorflow_config import tf_compile
from utilities.misc import cdf
# =========================
# Black–Scholes
# =========================

eps = 1e-12


@tf_compile
def bs_call_price(x, d, K):
    """

    :param x: log S/K
    :param d: sigma sqrt(tau)
    :return: Black-Scholes price normalized by K
    """
    d = tf.sqrt(tf.maximum(d, eps))
    d1 = (x - tf.math.log(K)) / d + 0.5 * d
    price = tf.exp(x) * cdf(d1) - K * cdf(d1 - d)
    return price


@tf_compile
def bs_delta(x, d, K):
    d = tf.sqrt(tf.maximum(d, eps))
    d1 = (x - tf.math.log(K)) / d + 0.5 * d
    return cdf(d1)
