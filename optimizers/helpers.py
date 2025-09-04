import tensorflow as tf
from utilities.tensorflow_config import tf_compile


# ======================= small math helpers ==================================

@tf_compile
def _dot(a, b):
    return tf.tensordot(a, b, axes=1)


@tf_compile
def _norm(a):
    return tf.sqrt(tf.maximum(0.0, _dot(a, a)))
