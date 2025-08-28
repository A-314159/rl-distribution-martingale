from abc import ABC

import tensorflow as tf
from utilities.tensorflow_config import tf_compile


@tf_compile
def pack_vars(vars_list):
    return tf.concat([tf.reshape(v, [-1]) for v in vars_list], axis=0)


@tf_compile
def unpack_like(vec, vars_like):
    parts, offset = [], 0
    for v in vars_like:
        size = tf.size(v)
        part = tf.reshape(vec[offset: offset + size], tf.shape(v))
        parts.append(part)
        offset += size
    return parts


@tf_compile
def apply_delta(vars_list, delta_flat):
    for v, d in zip(vars_list, unpack_like(delta_flat, vars_list)):
        v.assign_add(d)


# Your residuals function (batched). Return a *flat* tensor.
@tf_compile
def residuals(model, x, y):
    r = model(x) - y  # [B, out]
    return tf.reshape(r, [-1])  # [B*out]


# =========================================================
# 2) GN/LM Optimizer class
#    - Works like a TF optimizer, but `minimize(...)` expects
#      a residuals_fn that returns the flattened residual vector.
# =========================================================
class GaussNewtonLM(tf.keras.optimizers.Optimizer, ABC):
    def __init__(self, lam=1e-2, cg_tol=1e-6, cg_iters=50, name="GNLM", **kwargs):
        super().__init__(name, **kwargs)
        self.lam = tf.constant(lam, dtype=tf.float32)
        self.cg_tol = tf.constant(cg_tol, dtype=tf.float32)
        self.cg_iters = tf.constant(cg_iters, dtype=tf.int32)

    def get_config(self):
        cfg = super().get_config()
        cfg.update(dict(lam=float(self.lam.numpy()),
                        cg_tol=float(self.cg_tol.numpy()),
                        cg_iters=int(self.cg_iters.numpy())))
        return cfg

    # ---- CG solver (matrix-free): (A) v = J^T J v + lam v
    @tf_compile
    def _cg(self, matvec, b):
        x = tf.zeros_like(b)
        r = b - matvec(x)
        p = tf.identity(r)
        rs_old = tf.tensordot(r, r, 1)
        for _ in tf.range(self.cg_iters):
            Ap = matvec(p)
            alpha = rs_old / (tf.tensordot(p, Ap, 1) + 1e-30)
            x = x + alpha * p
            r = r - alpha * Ap
            rs_new = tf.tensordot(r, r, 1)
            if tf.sqrt(rs_new) < self.cg_tol:
                break
            beta = rs_new / (rs_old + 1e-30)
            p = r + beta * p
            rs_old = rs_new
        return x

    # ---- One GN/LM step wrapped as "minimize"
    # residuals_fn() must return the *flattened residual vector* r (shape [B*out])
    @tf_compile
    def minimize(self, residuals_fn, var_list):
        # g = J^T r (gradient of mean 0.5||r||^2)
        with tf.GradientTape() as tape:
            r = residuals_fn()
            loss = 0.5 * tf.reduce_mean(tf.square(r))
        g_list = tape.gradient(loss, var_list)
        g_flat = pack_vars([tf.zeros_like(v) if g is None else g for v, g in zip(var_list, g_list)])
        b = -g_flat

        # matvec: v -> J^T J v + lam v  (no explicit Jacobian)
        def matvec(v_flat):
            v_list = unpack_like(v_flat, var_list)
            # Forward JVP: u = J v
            with tf.autodiff.ForwardAccumulator(primals=var_list, tangents=v_list) as acc:
                res = residuals_fn()
            u = acc.jvp(res)
            # Reverse VJP: J^T u
            with tf.GradientTape() as tape2:
                r2 = residuals_fn()
                inner = tf.reduce_sum(r2 * tf.stop_gradient(u)) / tf.cast(tf.size(r2), tf.float32)
            jtju_list = tape2.gradient(inner, var_list)
            jtju_flat = pack_vars([tf.zeros_like(v) if g is None else g for v, g in zip(var_list, jtju_list)])
            return jtju_flat + self.lam * v_flat

        # Solve for delta with CG and apply
        delta = self._cg(matvec, b)
        apply_delta(var_list, delta)
        return loss, tf.norm(g_flat), tf.norm(delta)


@tf_compile
def standard_step(model, x, y, opt):
    with tf.GradientTape() as tape:
        r = residuals(model, x, y)
        loss = 0.5 * tf.reduce_mean(tf.square(r))
    grads = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(grads, model.trainable_variables))
    return loss
