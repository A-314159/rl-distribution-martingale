import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def distribution_critic(input_dim=4, hidden=(32, 32), activation="tanh"):
    dtype = tf.keras.backend.floatx()
    inputs = keras.Input(shape=(input_dim,), dtype=dtype, name="features")
    norm = layers.Normalization(axis=-1, name="norm")
    x = norm(inputs)
    for i, h in enumerate(hidden):
        x = layers.Dense(h, activation=activation, name=f"dense_{i}")(x)
    out = layers.Dense(1, activation="sigmoid", name="cdf")(x)
    return keras.Model(inputs, out, name="F_critic")
