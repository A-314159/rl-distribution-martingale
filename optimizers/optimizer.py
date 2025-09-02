import tensorflow as tf
import numpy as np
from tensorflow import keras
from optimizers.gnlm import GaussNewtonLM
from optimizers.tfp_lbfgs import LBFGS
from optimizers.lbfgsm import LBFGSM
from utilities.tensorflow_config import tf_compile, LOW, HIGH, SENSITIVE_CALC


def function_factory(model, f):
    shapes = tf.shape_n(model.trainable_variables)
    n_tensors = len(shapes)
    count = 0
    idx = []  # stitch indices
    part = []  # partition indices
    for i, shape in enumerate(shapes):
        n = np.product(shape)
        idx.append(tf.reshape(tf.range(count, count + n, dtype=tf.int32), shape))
        part.extend([i] * n)
        count += n
    part = tf.constant(part)

    @tf_compile
    def to_model(x):
        tensors = tf.dynamic_partition(x, part, n_tensors)
        for i, (shape, param) in enumerate(zip(shapes, tensors)):
            model.trainable_variables[i].assign(tf.reshape(param, shape))

    @tf_compile
    def f(x):
        with tf.GradientTape() as tape:
            to_model(x)
            l = f()
        g_list = tape.gradient(l, model.trainable_variables)
        g = tf.dynamic_stitch(idx, g_list)
        return g, l

    f.idx, f.part, f.shapes, f.to_model = idx, part, shapes, to_model
    return f


class Optimizer:
    def __init__(self, cfg, function, model):
        name = cfg.optimizer.lower()
        self.model = model
        self.function = function
        self.require_full_batch = False

        run = self.gd  # gd as in gradient descent
        if name == "adam":
            opt = keras.optimizers.Adam(cfg.lr)
        elif name == "sgd":
            opt = keras.optimizers.SGD(cfg.lr, momentum=0.9, nesterov=True)
        elif name == "rmsprop":
            opt = keras.optimizers.RMSprop(cfg.lr)
        elif name in ("gn", "lm"):
            opt = GaussNewtonLM(damping=cfg.gn_damping, max_iters=cfg.gn_iters, verbose=cfg.gn_verbose)
            run = self.gn  # gn as in Gauss-Newton
        elif name == "lbfgs":
            opt = LBFGS(max_iters=cfg.lbfgs_iters, tol=cfg.lbfgs_tol, verbose=cfg.lbfgs_verbose)
            run = self.gn  # reuse the same closure-based path
            self.require_full_batch = True
        elif name == "lbfgsm":
            func = function_factory(model, self._loss)

            opt = LBFGSM(f=func, x_list=model.trainable_variables)
        else:
            raise Exception('Optimizer %s is not implemented' % name)

        self.opt = opt
        self.kind = name
        self.run = run

    @tf_compile
    def _loss(self):
        r = self.function()
        return tf.cast(0.5 * tf.reduce_mean(tf.square(r)), HIGH)

    @tf_compile
    def gd(self):
        with tf.GradientTape() as tape:
            loss = self._loss()
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        return float(loss.numpy())

    @tf_compile
    def gn(self):
        try:
            self.opt.minimize(self.function, self.model.trainable_variables)
        except AttributeError:
            raise Exception('Wrong usage of %s minimizer' % self.kind)
        return self._loss().numpy()
