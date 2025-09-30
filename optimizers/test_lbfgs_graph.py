import utilities.tensorflow_config as cfg

SEED = 1
cfg.configure(
    mode="eager",                 # or "eager" while debugging
    precision="float32",
    deterministic_ops=True,       # reproducible GPU kernels (no effect on CPU but fine)
    deterministic_shuffling=True, # if any tf.data is used elsewhere
    seed=SEED,                    # <<< match the ref
    use_gpu=False                 # optional: force CPU to remove device noise
)

import tensorflow as tf, numpy as np, random
tf.keras.utils.set_random_seed(SEED)
from utilities.tensorflow_config import tf_compile

# === your helpers from the demo file (same MLP + dataset) ===
from optimizers.ES_LBFGS import build_mlp, make_dataset  # <-- rename to your file name

# === the graph-friendly L-BFGS class from the first file ===
from optimizers.lbfgs_graph import LBFGS_GRAPH  # <-- rename to your first file name

# --- data & model (same setup as your script) ---
N = 32768 // 8
X, Y = make_dataset(N, seed=SEED, a=0.0, b=1.0)  # <<< same as ref

# If build_mlp(seed=...) exists, pass the same seed; otherwise the global seed above is enough.
model = build_mlp()  # or: build_mlp(seed=SEED)
var_list = model.trainable_variables

print("model var dtype:", var_list[0].dtype)
print("X dtype:", X.dtype)
print("Y dtype:", Y.dtype)

# First 2 rows of data
print("X[:2] =", X[:2].numpy())
print("Y[:2] =", Y[:2].numpy())

# A tiny weight fingerprint (sum and first 5 scalars)
w0 = tf.concat([tf.reshape(v, [-1]) for v in model.weights], axis=0)
print("w-sum =", float(tf.reduce_sum(w0)))
print("w[:5] =", w0[:5].numpy())


# --- loss+grad closure in the signature expected by LBFGS_GRAPH ---
#     • When need_gradient=True: return (loss, grads)
#     • When need_gradient=False: return (loss, None)  (the class is fine with None)
def loss_and_grad(vars_list, need_gradient: tf.Tensor):
    def with_grad():
        with tf.GradientTape() as tape:
            y_hat = model(X, training=False)
            loss = tf.reduce_mean(tf.square(y_hat - Y))
        grads = tape.gradient(loss, vars_list)
        # Replace any None grads (just in case) with zeros_like(var)
        grads = [tf.zeros_like(v) if g is None else g for g, v in zip(grads, vars_list)]
        return loss, grads

    def no_grad():
        y_hat = model(X, training=False)
        loss = tf.reduce_mean(tf.square(y_hat - Y))
        zeros = [tf.zeros_like(v) for v in vars_list]
        return loss, zeros

    # Ensure branch predicate is a boolean tensor
    need_gradient = tf.convert_to_tensor(need_gradient, dtype=tf.bool)
    return tf.cond(need_gradient, with_grad, no_grad)


# --- instantiate the optimizer (tweak knobs as you like) ---
opt = LBFGS_GRAPH(
    loss_and_grad=loss_and_grad,
    x0=var_list,  # list-of-variables mode (the class will pack/assign internally)
    memory=20,  # L-BFGS history size
    line_search="armijo",  # or "hager_zhang" if you’ve wired that path
    y_sign_mode="normal",  # or "auto"
    memory_update="fifo",  # or "quality_prune"
    armijo_c1=1e-4,
    armijo_window=1,
    backtrack_factor=0.5,
    max_evals_per_iter=20,
    wolfe_c2=0.9,
    powell_damping=True,
    init_scaling="bb",  # "bb" | "direction_match" | "constant"
    init_gamma=1.0,
    eps_curv=1e-12,
    dtype=tf.float32,  # model dtype
    opt_dtype=tf.float32,  # accumulation dtype (higher precision for optimizer math)
    debug_checks=False,
    # --- your Armijo engine extras (if exposed in your file) ---
    # armijo_win=5, c2=0.9, strong_wolfe=False, nonmonotone=True, etc.
    stall_reset_K=2,  # reset on repeated α≈0 (if present in your version)
    debug_print=True
)


# --- one compiled call that advances K iterations and returns rich history ---
@tf.function(jit_compile=False)
def lbfgs_step(K: int):
    return opt.step(tf.convert_to_tensor(K, tf.int32))

f0 = tf.cast(opt.f, tf.float64)                   # MSE stored by the optimizer
rmse0 = tf.sqrt(tf.maximum(f0, tf.constant(0.0, tf.float64)))
print(f"Iter 0: loss={rmse0.numpy():.6f}")

# --- run a few chunks until converged (or a hard limit) ---
max_outer = 100
iters_per_call = 1
for outer in range(max_outer):
    state = lbfgs_step(iters_per_call)  # dict with current state + stacked per-iter history
    f = float(state["f"].numpy())
    gnorm = float(state["g_norm"].numpy())
    # print(f"[outer {outer:02d}] f={f:.6e}  ||g||={gnorm:.3e}  mem={int(state['mem_size'].numpy())}")
    if gnorm < 1e-6:
        break

# --- final RMSE (matches the metric printed in your logs) ---
# (If your loss is MSE)
final_rmse = (state["f"].numpy()) ** 0.5
print("final √mse:", float(final_rmse))
