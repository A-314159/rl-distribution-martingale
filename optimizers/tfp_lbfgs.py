import tensorflow as tf
from utilities.tensorflow_config import tf_compile
from utilities.misc import cast_all
from typing import Tuple

try:
    import tensorflow_probability as tfp
except Exception as e:
    tfp = None


def _to_scalar_loss(residuals: tf.Tensor) -> tf.Tensor:
    # 0.5 * mean(||r||^2) — same objective you’re already using
    return 0.5 * tf.reduce_mean(tf.square(residuals))


class LBFGS:
    """
    Full-batch L-BFGS driver using TensorFlow Probability.
    Usage:
        opt = LBFGS(max_iters=50, tol=1e-6, verbose=False)
        loss = opt.run(error_closure, model.trainable_variables)

    Where:
        - error_closure(): returns the residual vector (Tensor) using CURRENT model weights.
          (No arguments; capture anything you need via closure: lambda: error_fn(lam, full_idx))
        - variables: list of tf.Variable (e.g., model.trainable_variables)

    Notes:
        * Deterministic, full-batch only (no mini-batch).
        * Runs on GPU if your model/tensors are on GPU.
        * Mixed precision is fine (variables typically float32; compute can be float16).
    """

    def __init__(self, max_iters: int = 50, tol: float = 1e-6, verbose: bool = False):
        if tfp is None:
            raise RuntimeError(
                "tensorflow_probability is required for LBFGS. "
                "Install a TF/TFP-matching version, e.g. TF 2.16 ↔ TFP 0.24.*"
            )
        self.max_iters = int(max_iters)
        self.tol = float(tol)
        self.verbose = bool(verbose)

    @tf_compile
    def cast(self):
        tp = tf.keras.backend.floatx()
        self.tol, cast_all(self.tol, dtype=tp)

    # ---- packing helpers (flatten/unflatten weights) ----

    @staticmethod
    @tf_compile
    def _pack(variables) -> tf.Tensor:
        tp = tf.keras.backend.floatx()
        if not variables:
            return tf.zeros([0], dtype=tp)
        flats = [tf.reshape(v, [-1]) for v in variables]
        return tf.concat(flats, axis=0)

    @staticmethod
    @tf_compile
    def _unpack_assign(theta_vec: tf.Tensor, variables):
        """Assign flat theta back into variable shapes."""
        offset = 0
        for v in variables:
            n = int(tf.size(v))
            slice_vec = theta_vec[offset: offset + n]
            v.assign(tf.reshape(slice_vec, v.shape))
            offset += n

    # ---- TF loss/grad on the CURRENT theta (weights set before call) ----

    @tf_compile
    def _loss_and_grad(self, f, variables) -> Tuple[tf.Tensor, tf.Tensor]:
        with tf.GradientTape() as tape:
            # Tape watches variables automatically
            residuals = f()
            loss = _to_scalar_loss(residuals)
        grads = tape.gradient(loss, variables)
        # Replace None grads (e.g., frozen vars) with zeros to keep shapes consistent
        flat_grads = []
        for v, g in zip(variables, grads):
            if g is None:
                g = tf.zeros_like(v)
            flat_grads.append(tf.reshape(g, [-1]))
        grad_vec = tf.concat(flat_grads, axis=0) if flat_grads else tf.zeros([0], dtype=loss.dtype)
        return loss, grad_vec

    # ---- public API ----

    def minimize(self, error_closure, variables) -> float:
        """
        Perform one L-BFGS solve (up to max_iters) and return the final scalar loss.
        """
        # Initial parameter vector
        theta0 = self._pack(variables)

        # We’ll repeatedly (re)assign theta before evaluating loss/grad
        def value_and_grad(theta_np):
            # Convert numpy -> tf, assign, then compute loss/grad in TF
            theta = tf.convert_to_tensor(theta_np, dtype=theta0.dtype)
            # assign into variables
            self._unpack_assign(theta, variables)
            # compute loss/grad with current weights
            loss, grad = self._loss_and_grad(error_closure, variables)
            # TFP expects float64 numpy arrays for robustness
            return loss.numpy().astype("float64"), grad.numpy().astype("float64")

        results = tfp.optimizer.lbfgs_minimize(
            value_and_gradients_function=value_and_grad,
            initial_position=theta0.numpy().astype("float64"),
            max_iterations=self.max_iters,
            tolerance=self.tol,
        )

        theta_star = tf.convert_to_tensor(results.position, dtype=theta0.dtype)
        self._unpack_assign(theta_star, variables)

        if self.verbose:
            iters = int(results.num_iterations.numpy())
            conv = bool(results.converged.numpy())
            objv = float(results.objective_value)
            print(f"[LBFGS] iters={iters} converged={conv} loss={objv:.6e}")

        # final loss for logging (with the final assigned weights)
        final_loss, _ = self._loss_and_grad(error_closure, variables)
        return float(final_loss.numpy())

