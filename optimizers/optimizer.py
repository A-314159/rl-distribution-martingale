import tensorflow as tf
from tensorflow import keras
from optimizers.gnlm import GaussNewtonLM
from optimizers.lbfgs import LBFGS
from utilities.tensorflow_config import tf_compile


class Optimizer:
    def __init__(self, cfg, model, error_function):
        name = cfg.optimizer.lower()
        self.model = model
        self.error_function = error_function
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
        elif name == 'lbfgs':
            opt = LBFGS(max_iters=cfg.lbfgs_iters, tol=cfg.lbfgs_tol, verbose=cfg.lbfgs_verbose)
            run = self.gn  # reuse the same closure-based path
            self.require_full_batch = True
        else:
            raise Exception('Optimizer %s is not implemented' % name)

        self.opt = opt
        self.kind = name
        self.run = run

    @tf_compile
    def _loss(self, lam, idx_batch):
        r = self.error_function(lam, idx_batch)
        return 0.5 * tf.reduce_mean(tf.square(r))

    @tf_compile
    def gd(self, lam, idx_batch=None):
        with tf.GradientTape() as tape:
            loss = self._loss(lam, idx_batch)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        return float(loss.numpy())

    @tf_compile
    def gn(self, lam, idx_batch=None):
        def error_closure():
            return self.error_function(lam, idx_batch)
        try:
            self.opt.minimize(error_closure, self.model.trainable_variables)
        except AttributeError:
            raise Exception('Wrong usage of %s minimizer'%self.kind)
        r = error_closure()
        return float(0.5 * tf.reduce_mean(tf.square(r)).numpy())
