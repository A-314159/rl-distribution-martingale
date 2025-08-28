from abc import ABC

import tensorflow as tf
from core.universe import UniverseBS
from core.bs import bs_delta  # your dtype-safe helper
from utilities.tensorflow_config import tf_compile
from utilities.misc import cast_all
import abc


class Actor(abc.ABC):
    def __call__(self, *arg, universe: UniverseBS) -> tf.Tensor:
        raise NotImplementedError

    @abc.abstractmethod
    def get_config(self) -> dict: ...

@tf_compile
class BSDeltaHedge(Actor, ABC):

    @tf_compile
    def __call__(self, *data, universe: UniverseBS) -> tf.Tensor:
        st = tf.keras.backend.floatx()
        (sigma, K) = cast_all(universe.sigma, universe.K, dtype=st)
        x, d = data[0], data[1] * sigma
        return -bs_delta(x, d, K)

    def get_config(self) -> dict:
        return {"name": "BSDeltaHedge"}