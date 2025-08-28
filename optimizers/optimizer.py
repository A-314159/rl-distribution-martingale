import tensorflow as tf
from tensorflow import keras
from optimizers.gnlm import GaussNewtonLM
from utilities.tensorflow_config import tf_compile


class Optimizer:
    def __init__(self, cfg, model, error_function):
        name = cfg.optimizer.lower()
        run = self.gd  # gd as in gradient descent
        self.require_full_batch = False
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
            self.require_full_batch = True
            # todo: plug tfp's optimizer later
            opt = None
            run = None
        else:
            raise Exception('Optimizer %s is not implemented' % name)
        self.model = model
        self.opt = opt
        self.error_function = error_function
        self.run = run

    @tf_compile
    def _loss(self, lam, idx_batch):
        r = self.error_function(lam, idx_batch)
        return 0.5 * tf.reduce_mean(tf.square(r))

    @tf_compile
    def gd(self, lam, idx_batch):
        with tf.GradientTape() as tape:
            loss = self._loss(lam, idx_batch)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        return float(loss.numpy())

    @tf_compile
    def gn(self, lam, idx_batch):
        def error_closure():
            return self.error_function(lam, idx_batch)
        try:
            self.opt.minimize(error_closure, self.model.trainable_variables)
        except AttributeError:
            if hasattr(self.gn, "run"):
                self.opt.run(error_closure, self.model.trainable_variables)
            else:
                raise
        r = error_closure()
        return float(0.5 * tf.reduce_mean(tf.square(r)).numpy())
