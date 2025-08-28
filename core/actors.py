import tensorflow as tf
from core.universe import UniverseBS
from core.bs import bs_delta  # your dtype-safe helper
from utilities.tensorflow_config import tf_compile
from utilities.misc import cast_all


class Actor:
    def __call__(self, *arg, universe: UniverseBS) -> tf.Tensor:
        raise NotImplementedError


@tf_compile
class BSDeltaHedge(Actor):

    @tf_compile
    def __call__(self, *data, universe: UniverseBS) -> tf.Tensor:
        st = tf.keras.backend.floatx()
        (sigma, K) = cast_all(universe.sigma, universe.K, dtype=st)
        x, d = data[0], data[1] * sigma
        return -bs_delta(x, d, K)
