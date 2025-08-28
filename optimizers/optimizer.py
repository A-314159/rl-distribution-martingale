import tensorflow as tf
from tensorflow import keras
from optimizers.gnlm import GaussNewtonLM
from utilities.tensorflow_config import tf_compile


class Optimizer:
    def __init__(self, cfg, model, residuals_fn_builder):
        name = cfg.optimizer.lower()
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
        else:
            raise Exception('Optimizer %s is not implemented' % name)
        self.model = model
        self.opt = opt
        self.function_builder = residuals_fn_builder
        self.run = run

    @tf_compile
    def _loss(self, idx_batch, lam):
        r = self.function_builder(lam)(idx_batch)
        return 0.5 * tf.reduce_mean(tf.square(r))

    @tf_compile
    def gd(self, idx_batch, lam):
        with tf.GradientTape() as tape:
            loss = self._loss(idx_batch, lam)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        return float(loss.numpy())

    @tf_compile
    def gn(self, idx_batch, lam):
        rfn = self.function_builder(lam)
        try:
            self.gn.minimize(rfn, self.model.trainable_variables)
        except AttributeError:
            if hasattr(self.gn, "step"):
                self.gn.run(rfn, self.model.trainable_variables)
            else:
                raise
        r = rfn(idx_batch)
        return float(0.5 * tf.reduce_mean(tf.square(r)).numpy())
