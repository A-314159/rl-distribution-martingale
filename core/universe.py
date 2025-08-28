from dataclasses import dataclass
import tensorflow as tf
from utilities.tensorflow_config import tf_compile
from utilities.misc import set_attributes, cast_all


# ---------------------------------------------
# At target, universe class should include anything related to the pseudo-Markovian state, such as:
# a) time, multi-dimension variable of a stochastic process that represent the state
# b) in finance: the description of the portfolio to be hedged (and the frequency of actions)
# c) a model of evolution of the universe
# ---------------------------------------------

class Universe:
    def children(self, *args) -> tf.Tensor:
        raise NotImplementedError


class UniverseBS(Universe):
    def __init__(self, sigma: float = 0.3, T: float = 1, K: float = 1, P: int = 60, **kwargs):
        self.sigma, self.T, self.K, self.P = sigma, T, K, P
        if kwargs is not None: set_attributes(self, kwargs)
        self.h: float = self.T / self.P

        tp = tf.keras.backend.floatx()
        self.sigma, self.T, self.h, self.K = cast_all(self.sigma, self.T, self.h, self.K, dtype=tp)

    @tf_compile
    def children(*args):
        x, d = args[1], args[2]  # caution: arg[0] is the calling object
        x_mu = x - 0.5 * d ** 2
        x_children = tf.stack([x_mu + d, x_mu - d], axis=1)
        probs = tf.ones_like(x_children) * 0.5
        return x_children, probs
