import tensorflow as tf
from utilities.tensorflow_config import tf_compile, HIGH, SENSITIVE_CALC
from optimizers.helpers import _dot, _norm
from optimizers.armijo import armijo_engine
from optimizers.hager_zhang import hager_zhang


# pylint: disable=shadowed-name
# noinspection PyShadowingNames

# -----------------------------------------------------------------
# Example of loss and gradient function: it must return 2 variables if when no gradient is required
# -----------------------------------------------------------------

@tf_compile
def loss_and_grad_example(x, need_gradient: tf.Tensor):
    def with_grad():
        with tf.GradientTape() as tape:
            tape.watch(x)  # ensure x is watched if it’s a Tensor
            f = ...  # compute your loss from x (pure TF ops)
        g = tape.gradient(f, x)
        return f, g

    def no_grad():
        f = ...  # SAME loss as above
        return f, None

    return tf.cond(need_gradient, with_grad, no_grad)


# ============== utils: pack / assign for list-of-variables ===================

@tf_compile
def pack(var_list):
    """Concatenate all variables into a 1-D tensor. Graph-friendly."""
    flats = [tf.reshape(v, [-1]) for v in var_list]
    return tf.concat(flats, axis=0)


def _sizes_shapes_static(var_list):
    """Compute static sizes and shapes once (Python)."""
    sizes = [int(tf.size(v)) for v in var_list]
    shapes = [v.shape for v in var_list]
    return sizes, shapes


def make_assign_fn(var_list):
    """
    Returns assign_from_flat(x_flat) that writes into var_list.
    Uses a Python for-loop; AutoGraph unrolls to device-side assigns.
    """
    sizes, shapes = _sizes_shapes_static(var_list)

    @tf_compile
    def assign_from_flat(x_flat):
        parts = tf.split(x_flat, sizes, axis=0)
        for v, p, shp in zip(var_list, parts, shapes):
            v.assign(tf.reshape(p, shp))
        return tf.constant(0, tf.int32)

    return assign_from_flat


# ======================== Debug helper =======================================
@tf_compile
def add_flag(mask, cond, bit):
    add = tf.where(cond, tf.zeros_like(mask),
                   tf.bitwise.left_shift(tf.ones_like(mask), bit))
    return mask | add


@tf_compile
def _probe_pre_ls_flagged(flag, f_m, g_m, d_o):
    def run():
        one = tf.constant(1, tf.int32)
        mask = tf.constant(0, tf.int32)

        f_ok = tf.reduce_all(tf.math.is_finite(f_m))
        g_ok = tf.reduce_all(tf.math.is_finite(g_m))
        d_ok = tf.reduce_all(tf.math.is_finite(d_o))
        gTd = _dot(tf.cast(g_m, d_o.dtype), d_o)
        gTd_ok = tf.reduce_all(tf.math.is_finite(gTd))

        mask |= tf.where(f_ok, 0, one)
        mask = add_flag(mask, g_ok, 1)
        mask = add_flag(mask, d_ok, 2)
        mask = add_flag(mask, gTd_ok, 3)
        #mask |= tf.where(g_ok, 0, one << 1)
        #mask |= tf.where(d_ok, 0, one << 2)
        #mask |= tf.where(gTd_ok, 0, one << 3)

        return mask, tf.cast(gTd, d_o.dtype)

    def no_run():
        return tf.constant(0, tf.int32), tf.zeros([], d_o.dtype)

    return tf.cond(flag, run, no_run)


@tf_compile
def _probe_post_ls_flagged(flag, f_new_m, g_new_m, dtype_o):
    def run():
        one = tf.constant(1, tf.int32)
        mask = tf.constant(0, tf.int32)
        f_ok = tf.reduce_all(tf.math.is_finite(f_new_m))
        g_ok = tf.reduce_all(tf.math.is_finite(g_new_m))
        mask |= tf.where(f_ok, 0, one)
        mask = add_flag(mask, g_ok, 1)
        #mask |= tf.where(g_ok, 0, one << 1)
        return mask

    def no_run():
        return tf.constant(0, tf.int32)

    return tf.cond(flag, run, no_run)


@tf_compile
def _probe_curvature_flagged(flag, s_m, y_m, sTy_o, q_new_o):
    def run():
        one = tf.constant(1, tf.int32)
        mask = tf.constant(0, tf.int32)
        s_ok = tf.reduce_all(tf.math.is_finite(s_m))
        y_ok = tf.reduce_all(tf.math.is_finite(y_m))
        sTy_ok = tf.reduce_all(tf.math.is_finite(sTy_o))
        q_ok = tf.reduce_all(tf.math.is_finite(q_new_o))
        mask |= tf.where(s_ok, 0, one)
        mask = add_flag(mask, y_ok, 1)
        mask = add_flag(mask, sTy_ok, 2)
        mask = add_flag(mask, q_ok, 3)
        #mask |= tf.where(y_ok, 0, one << 1)
        #mask |= tf.where(sTy_ok, 0, one << 2)
        #mask |= tf.where(q_ok, 0, one << 3)
        return mask

    def no_run():
        return tf.constant(0, tf.int32)

    return tf.cond(flag, run, no_run)


# noinspection SpellCheckingInspection
@tf_compile
def _debug_assert_list(flag: tf.Tensor, tensors):
    """If `flag` is True, assert all tensors in `tensors` are finite. Returns a dummy int32."""

    def do():
        for t in tensors:
            tf.debugging.assert_all_finite(t, "non-finite detected")
        return tf.constant(0, tf.int32)

    return tf.cond(flag, do, lambda: tf.constant(0, tf.int32))


# ======================== L-BFGS primitives (OPT dtype kernels) ==============
# noinspection PyShadowingNames
# pylint: disable=shadowed-name
@tf_compile
def _two_loop_opt(S_m, Y_m, mem_size, m_cap, gamma_o, g_m, eps_rho_o):
    """
    Two-loop in OPT dtype; returns (direction, used_rho_cap_flag).
    """
    g_o = tf.cast(g_m, gamma_o.dtype)
    ms = tf.cast(tf.convert_to_tensor(mem_size), tf.int32)
    S = tf.cast(S_m[:ms], g_o.dtype)
    Y = tf.cast(Y_m[:ms], g_o.dtype)

    def _empty():
        return -g_o, tf.constant(False)

    def _nonempty():
        alpha_TA = tf.TensorArray(g_o.dtype, size=ms, clear_after_read=False)
        rho_TA = tf.TensorArray(g_o.dtype, size=ms, clear_after_read=False)
        used_cap = tf.constant(False)

        def bwd_cond(i, _q, _aTA, _rTA, _used):
            return i >= 0

        def bwd_body(i, q, aTA, rTA, used):
            si = S[i]
            yi = Y[i]
            sTy = _dot(yi, si)
            denom = tf.maximum(sTy, eps_rho_o)
            used = tf.logical_or(used, sTy <= eps_rho_o)  # flag if clamped
            rho = 1.0 / denom
            a = rho * _dot(si, q)
            q = q - a * yi
            return i - 1, q, aTA.write(i, a), rTA.write(i, rho), used

        _, q, alpha_TA, rho_TA, used_cap = tf.while_loop(
            bwd_cond, bwd_body,
            loop_vars=(ms - 1, g_o, alpha_TA, rho_TA, used_cap),
            maximum_iterations=ms, parallel_iterations=1
        )

        r = gamma_o * q

        def fwd_cond(i, _r):
            return i < ms

        def fwd_body(i, r):
            si = S[i]
            yi = Y[i]
            rho = rho_TA.read(i)
            a = alpha_TA.read(i)
            b = rho * _dot(yi, r)
            return i + 1, r + si * (a - b)

        _, r = tf.while_loop(
            fwd_cond, fwd_body, loop_vars=(0, r),
            maximum_iterations=ms, parallel_iterations=1
        )
        return r, used_cap

    return tf.cond(ms > 0, _nonempty, _empty)


@tf_compile
def _initial_gamma_opt(mode_code, init_gamma_o, S_m, Y_m, mem_size, g_m, d_prev_m, gam_lo_o, gam_hi_o, eps_div_o):
    S = tf.cast(S_m, init_gamma_o.dtype)
    Y = tf.cast(Y_m, init_gamma_o.dtype)
    g = tf.cast(g_m, init_gamma_o.dtype)
    d_prev = tf.cast(d_prev_m, init_gamma_o.dtype)
    gam_lo = tf.cast(gam_lo_o, init_gamma_o.dtype)
    gam_hi = tf.cast(gam_hi_o, init_gamma_o.dtype)
    eps_div = tf.cast(eps_div_o, init_gamma_o.dtype)

    def const():
        return tf.clip_by_value(init_gamma_o, gam_lo, gam_hi)

    def barzilai_borwein():  # classical choice of gamma
        idx = mem_size - 1
        sL, yL = S[idx], Y[idx]
        yTy = _dot(yL, yL)
        gam = _dot(sL, yL) / tf.maximum(yTy, eps_div)
        gam = tf.clip_by_value(gam, gam_lo, gam_hi)
        gam = tf.where(tf.math.is_finite(gam), gam, tf.ones_like(gam))
        return gam

    # noinspection PyShadowingNames
    # pylint: disable=shadowed-name
    # noinspection SpellCheckingInspection
    def direction_based():  # to be tested: use the last direction to set gamma
        gTd = _dot(g, d_prev)
        g2 = _dot(g, g)
        gam = -gTd / tf.maximum(g2, eps_div)
        gam = tf.clip_by_value(gam, gam_lo, gam_hi)
        gam = tf.where(tf.math.is_finite(gam), gam, tf.ones_like(gam))
        return gam

    return tf.case(
        pred_fn_pairs=[
            (tf.equal(mode_code, 0), const),
            (tf.logical_and(tf.equal(mode_code, 1), mem_size > 0), barzilai_borwein),
            (tf.logical_and(tf.equal(mode_code, 2), _norm(d_prev) > 0.0), direction_based),
        ],
        default=const, exclusive=False
    )


@tf_compile
def _powell_damp_ref_opt(s_m, y_m, eps_curv_o, powell_c_o):
    # dtypes
    do = eps_curv_o.dtype          # optimizer dtype
    dm = y_m.dtype                 # model dtype

    # cast once
    s_o = tf.cast(s_m, do)
    y_o = tf.cast(y_m, do)

    # basic scalars
    sTs = _dot(s_o, s_o)
    yTy = _dot(y_o, y_o)
    sy  = _dot(s_o, y_o)

    # numerics guard (match ref scale ~1e-16)
    epsd = tf.maximum(eps_curv_o, tf.cast(1e-16, do))

    # --- ref: flip y if negative curvature ---
    flipped_b   = sy < 0.0
    y_flipped_o = tf.where(flipped_b, -y_o, y_o)
    sy_abs      = tf.abs(sy)  # |s^T y| (unchanged by flip)

    # --- ref: tiny floor on |s^T y| ---
    fl     = sTs * yTy * tf.cast(1e-9, do)
    tiny_b = sy_abs < fl
    a      = fl / tf.maximum(yTy, epsd)         # scale factor for y if tiny
    y_eff_o = tf.where(tiny_b, tf.cast(a, do) * y_flipped_o, y_flipped_o)

    # effective |s^T y| and y^T y after tiny-floor
    yTy_eff     = tf.where(tiny_b, (a * a) * yTy, yTy)
    sy_eff_abs  = tf.where(tiny_b, fl, sy_abs)

    # --- ref: s^T B s with H0 = gamma I, B0 = I/gamma, gamma = |s^T y| / (y^T y) ---
    gamma = sy_eff_abs / tf.maximum(yTy_eff, epsd)
    sBs   = (1.0 / tf.maximum(gamma, epsd)) * sTs

    # Powell criterion: c = powell_delta (ref uses 0.2)
    c    = tf.cast(powell_c_o, do)
    num  = (1.0 - c) * sBs
    den  = tf.maximum(sBs - sy_eff_abs, epsd)
    theta = tf.where(sy_eff_abs >= c * sBs, tf.ones((), do), num / den)

    # y_bar = theta * y_eff + (1 - theta) * (1/gamma) * s
    y_bar_o  = theta * y_eff_o + (1.0 - theta) * ((1.0 / tf.maximum(gamma, epsd)) * s_o)
    y_bar_m  = tf.cast(y_bar_o, dm)
    sTy_used = _dot(s_o, y_bar_o)  # optimizer dtype

    # keep EXACTLY the same 3 returns your code expects
    return y_bar_m, sTy_used, theta, flipped_b


# ======================== =======================

@tf_compile
def _append_fifo_with_q(S, Y, Q, mem_size, s_new, y_new, q_new, m):
    def not_full():
        idx = tf.cast(mem_size, tf.int32)
        S2 = tf.tensor_scatter_nd_update(S, [[idx]], [s_new])
        Y2 = tf.tensor_scatter_nd_update(Y, [[idx]], [y_new])
        Q2 = tf.tensor_scatter_nd_update(Q, [[idx]], [q_new])
        return S2, Y2, Q2, mem_size + 1

    def full():
        S2 = tf.concat([S[1:], tf.expand_dims(s_new, 0)], axis=0)
        Y2 = tf.concat([Y[1:], tf.expand_dims(y_new, 0)], axis=0)
        Q2 = tf.concat([Q[1:], tf.expand_dims(q_new, 0)], axis=0)
        return S2, Y2, Q2, mem_size

    return tf.cond(mem_size < m, not_full, full)


@tf_compile
def _append_quality_prune_with_q(S, Y, Q, mem_size, s_new, y_new, q_new, m):
    def not_full():
        idx = tf.cast(mem_size, tf.int32)
        return (tf.tensor_scatter_nd_update(S, [[idx]], [s_new]),
                tf.tensor_scatter_nd_update(Y, [[idx]], [y_new]),
                tf.tensor_scatter_nd_update(Q, [[idx]], [q_new]),
                mem_size + 1)

    def full():
        worst = tf.argmin(Q[:mem_size], output_type=tf.int32)
        return (tf.tensor_scatter_nd_update(S, [[worst]], [s_new]),
                tf.tensor_scatter_nd_update(Y, [[worst]], [y_new]),
                tf.tensor_scatter_nd_update(Q, [[worst]], [q_new]),
                mem_size)

    return tf.cond(mem_size < m, not_full, full)


# ---------------- compile checks ----------------

def _is_tf_function(fn) -> bool:
    """
    Heuristic: 'tf.function' objects have a 'get_concrete_function' attribute
    (and a 'python_function' backref). Raw Python callables won't.
    Works for functions decorated by your @tf_compile in non-eager mode.
    """
    return callable(getattr(fn, "get_concrete_function", None))


def _assert_compilation_mode(fn, fn_name="loss_and_grad"):
    if tf.executing_eagerly():
        # Eager/debug: both compiled and raw functions work.
        # Optional hint if compiled (some people prefer raw for debugging).
        if _is_tf_function(fn):
            tf.print("[info]", fn_name, "is compiled but running in eager mode; "
                                        "that’s fine, just note step-by-step Python debugging may be less convenient.")
    else:
        # Graph mode: prefer compiled to reduce overhead/retracing.
        if not _is_tf_function(fn):
            tf.print("[warn]", fn_name, "is not a tf.function while running in graph mode; "
                                        "it will still work (AutoGraph will inline it), but consider decorating with "
                                        "@tf_compile"
                                        "to reduce retracing and enable XLA if desired.")


# ======================= The L-BFGS stepper ==================================
# noinspection PyShadowingNames
# pylint: disable=shadowed-name
class LBFGS_GRAPH:
    """
    Initialize once. Call step(K) inside your own @tf.function training loop to advance K iterations.
    Supports:
      - line_search: "armijo" or "hager_zhang"
      - memory_update: "fifo" or "quality_prune"
    """

    def __init__(self,
                 loss_and_grad, x0,
                 memory=20,
                 line_search="armijo",
                 y_sign_mode="normal",
                 memory_update="fifo",
                 armijo_c1=1e-4, armijo_window=5, backtrack_factor=0.5,
                 max_evals_per_iter=20, wolfe_c2=0.9,
                 powell_damping=True,
                 powell_mode="ref",  # "ref" or "classic"
                 powell_delta=0.2,  # δ in (0,1)
                 init_scaling="bb",  # "bb" for classical BB γ; "direction_match" uses last direction; "constant"
                 init_gamma=1.0,
                 eps_curv=1e-12,
                 dtype=tf.float32,  # model dtype (default fp32)
                 opt_dtype=None,  # accumulation dtype (None → choose by model dtype)
                 debug_checks=False,  # runtime finite checks
                 # ---- Armijo engine knobs (defaults preserve current behavior) ----
                 armijo_use_cubic=False,  # False → geometric; True → safeguarded cubic
                 armijo_use_wolfe=False,  # False → pure Armijo; True → light curvature check
                 armijo_step_max=0.0,  # 0.0 → no trust-region cap on ||d||
                 stall_reset_K=2,
                 debug_print: bool = False):

        self.y_sign_mode = tf.constant({"normal": 0, "auto": 1}[y_sign_mode], tf.int32)

        # dtypes
        self.dtype_model = dtype
        self.dtype_opt = dtype if opt_dtype is None else opt_dtype

        self.powell = tf.constant(1 if powell_damping else 0, tf.int32)
        self.powell_mode = tf.constant(0 if powell_mode == "classic" else 1, tf.int32)  # 1="ref"
        self.powell_delta = tf.constant(powell_delta, self.dtype_opt)

        # Scalars used in algebra live in OPT dtype
        self.line_search = line_search
        self.memory_update = memory_update
        self.c1 = tf.constant(armijo_c1, self.dtype_opt)
        self.window = tf.constant(int(armijo_window), tf.int32)
        self.bt = tf.constant(backtrack_factor, self.dtype_opt)
        self.max_evals = tf.constant(int(max_evals_per_iter), tf.int32)
        self.c2 = tf.constant(wolfe_c2, self.dtype_opt)
        self.powell = tf.constant(1 if powell_damping else 0, tf.int32)
        self.gamma_mode = tf.constant({"constant": 0, "bb": 1, "direction_match": 2}[init_scaling], tf.int32)
        self.gamma_init = tf.constant(init_gamma, self.dtype_opt)
        self.eps_curv = tf.constant(eps_curv, self.dtype_opt)
        self.debug_checks = tf.constant(bool(debug_checks), tf.bool)

        # Armijo policy (opt dtype)
        self.armijo_use_cubic = tf.constant(bool(armijo_use_cubic), tf.bool)
        self.armijo_use_wolfe = tf.constant(bool(armijo_use_wolfe), tf.bool)
        self.armijo_max_norm_d = tf.cast(armijo_step_max, self.dtype_opt)

        # gamma clamps chosen by OPT dtype
        if self.dtype_opt == tf.float64:
            lo, hi = 1e-6, 1e6
        elif self.dtype_opt == tf.float32:
            lo, hi = 1e-4, 1e4
        else:
            lo, hi = 1e-2, 1e2
        self.gam_lo = tf.constant(lo, self.dtype_opt)
        self.gam_hi = tf.constant(hi, self.dtype_opt)

        # Dtype-aware small constants (based on OPT dtype)
        if self.dtype_opt == tf.float64:
            eps_div = 1e-32
            eps_q = 1e-16
            tol_alpha = 1e-12
            alpha_floor = 1e-8
            eps_rho = 1e-24
        elif self.dtype_opt == tf.float32:
            eps_div = 1e-32
            eps_q = 1e-8
            tol_alpha = 1e-8
            alpha_floor = 1e-6
            eps_rho = 1e-12
        else:  # fp16/bf16 accumulation (uncommon)
            eps_div = 1e-7
            eps_q = 1e-4
            tol_alpha = 1e-4
            alpha_floor = 1e-3
            eps_rho = 1e-6

        self.eps_div = tf.constant(eps_div, self.dtype_opt)
        self.eps_q = tf.constant(eps_q, self.dtype_opt)
        self.tol_alpha = tf.constant(tol_alpha, self.dtype_opt)
        self.alpha_floor = tf.constant(alpha_floor, self.dtype_opt)
        self.eps_rho = tf.constant(eps_rho, self.dtype_opt)

        if callable(loss_and_grad):
            _assert_compilation_mode(loss_and_grad, "loss_and_grad")
        else:
            raise TypeError("loss_and_grad must be callable")

        # Support tensor x0 OR list-of-variables x0
        if isinstance(x0, list):
            # list-of-variables mode — variables remain in MODEL dtype
            self.var_list = x0
            self.assign_from_flat = make_assign_fn(self.var_list)
            x_flat_m = tf.cast(pack(self.var_list), self.dtype_model)

            @tf_compile
            def _loss_and_grad(x, need_gradient: tf.Tensor):
                _ = self.assign_from_flat(x)

                def with_grad():
                    f, g_list = loss_and_grad(self.var_list, tf.constant(True))
                    g_list = [gi if gi is not None else tf.zeros_like(vi)
                              for gi, vi in zip(g_list, self.var_list)]
                    return tf.cast(f, self.dtype_model), tf.cast(pack(g_list), self.dtype_model)

                def no_grad():
                    f, g_list = loss_and_grad(self.var_list, tf.constant(False))
                    g_flat = tf.zeros_like(x)
                    return tf.cast(f, self.dtype_model), tf.cast(g_flat, self.dtype_model)

                return tf.cond(need_gradient, with_grad, no_grad)

            self.loss_and_grad = _loss_and_grad
            self.x = tf.Variable(x_flat_m, dtype=self.dtype_model, trainable=False)
        else:
            if callable(loss_and_grad):
                try:
                    _ = loss_and_grad(tf.convert_to_tensor(x0, self.dtype_model), tf.constant(True))
                    self.loss_and_grad = loss_and_grad
                except TypeError:

                    @tf_compile
                    def _wrap(x, need_gradient: tf.Tensor):
                        f, g = loss_and_grad(x)

                        def with_grad():
                            return tf.cast(f, self.dtype_model), tf.cast(g, self.dtype_model)

                        def no_grad():
                            return tf.cast(f, self.dtype_model), tf.zeros_like(x)

                        return tf.cond(need_gradient, with_grad, no_grad)

                    self.loss_and_grad = _wrap
            else:
                raise TypeError("loss_and_grad must be callable")

            self.x = tf.Variable(tf.cast(tf.identity(x0), self.dtype_model), dtype=self.dtype_model, trainable=False)

        tf_true = tf.constant(True)
        tf_false = tf.constant(False)

        # Check gradients path
        fT_m, gT_m = self.loss_and_grad(self.x, tf_true)
        tf.debugging.assert_rank_at_least(fT_m, 0)
        tf.debugging.assert_equal(tf.shape(gT_m)[0], tf.shape(self.x)[0], message="grad size mismatch")
        tf.debugging.assert_all_finite(fT_m, "loss not finite at init")
        tf.debugging.assert_all_finite(gT_m, "grad not finite at init")

        # Check loss-only path returns a loss Tensor
        fF_m, gF_m = self.loss_and_grad(self.x, tf_false)
        tf.debugging.assert_rank_at_least(fF_m, 0)

        # Evaluate initial f,g (MODEL dtype)
        f0_m, g0_m = self.loss_and_grad(self.x, tf.constant(True))
        self.f = tf.Variable(tf.cast(f0_m, self.dtype_model), trainable=False)
        self.g = tf.Variable(tf.cast(g0_m, self.dtype_model), trainable=False)
        self.g_norm = tf.Variable(_norm(self.g), trainable=False)

        # Memory (MODEL dtype storage for bandwidth)
        n = tf.shape(self.x)[0]
        self.m = tf.constant(int(memory), tf.int32)
        self.S = tf.Variable(tf.zeros([self.m, n], dtype=self.dtype_model), trainable=False)
        self.Y = tf.Variable(tf.zeros([self.m, n], dtype=self.dtype_model), trainable=False)
        self.mem_size = tf.Variable(0, dtype=tf.int32, trainable=False)
        self.Q = tf.Variable(tf.fill([self.m], tf.constant(0., self.dtype_model)), trainable=False)

        # Bookkeeping
        self.alpha_prev = tf.Variable(tf.cast(1.0, self.dtype_opt), dtype=self.dtype_opt, trainable=False)  # opt dtype
        self.d_prev = tf.Variable(tf.zeros_like(self.x), trainable=False)  # model dtype

        self.f_hist = tf.Variable(tf.fill([self.window], tf.cast(self.f, self.dtype_model)), trainable=False)
        self.f_hist_size = tf.Variable(1, dtype=tf.int32, trainable=False)

        # Debug counters
        self.rho_cap_count = tf.Variable(0, dtype=tf.int32, trainable=False)
        self.descent_repair_count = tf.Variable(0, dtype=tf.int32, trainable=False)

        # --- α=0 stall guard knobs/state ---
        self.stall_reset_K = tf.constant(int(stall_reset_K), tf.int32)
        self.zero_alpha_streak = tf.Variable(0, dtype=tf.int32, trainable=False)

        # --- NEW: lifetime reset counters ---
        self.alpha_zero_resets = tf.Variable(0, dtype=tf.int32, trainable=False)
        self.nonfinite_resets = tf.Variable(0, dtype=tf.int32, trainable=False)

        self.debug_print = bool(debug_print)
        self.debug_tf = tf.constant(self.debug_print)

        # tiny eps in opt dtype (one place)
        self.eps_o = tf.cast(1e-30 if self.dtype_opt == tf.float64 else 1e-20, self.dtype_opt)
        # inside LBFGS_GRAPH.__init__(...) after dtype fields are set
        self.eps_o = tf.cast(1e-30 if self.dtype_opt == tf.float64 else 1e-20, self.dtype_opt)
        self.eps_m = tf.cast(1e-30 if self.dtype_model == tf.float64 else 1e-20, self.dtype_model)

        # eager-only / print-only state
        self._prev_d = None  # will be a tf.Variable shaped like d
        self._have_prev_d = tf.Variable(False, trainable=False)
        self._iter_counter = tf.Variable(0, trainable=False, dtype=tf.int32)

    def _dbg(self, *vals):
        # print only if debug_print is True; uses tf.print under graph, print under eager
        if self.debug_print:
            if tf.executing_eagerly():
                # pure Python print reads tensors' .numpy()
                txt = []
                for v in vals:
                    try:
                        txt.append(float(v.numpy()) if hasattr(v, "numpy") else v)
                    except Exception:
                        txt.append(v)
                print(*txt)
            else:
                tf.print(*vals)

    @tf_compile
    def step(self, iters: tf.Tensor):
        """
        Advance 'iters' iterations. Returns current state and per-iteration history for this call.
        """
        max_iters = tf.cast(iters, tf.int32)

        f_TA = tf.TensorArray(self.dtype_model, size=max_iters, clear_after_read=False)
        gnorm_TA = tf.TensorArray(self.dtype_model, size=max_iters, clear_after_read=False)
        alpha_TA = tf.TensorArray(self.dtype_opt, size=max_iters, clear_after_read=False)
        evals_TA = tf.TensorArray(tf.int32, size=max_iters, clear_after_read=False)
        msize_TA = tf.TensorArray(tf.int32, size=max_iters, clear_after_read=False)
        qual_TA = tf.TensorArray(self.dtype_model, size=max_iters, clear_after_read=False)
        # diagnostics
        damped_TA = tf.TensorArray(tf.int32, size=max_iters, clear_after_read=False)
        flipped_TA = tf.TensorArray(tf.int32, size=max_iters, clear_after_read=False)
        angle_TA = tf.TensorArray(self.dtype_opt, size=max_iters, clear_after_read=False)
        # NEW: per-iteration reset flags
        stallreset_TA = tf.TensorArray(tf.int32, size=max_iters, clear_after_read=False)
        nfreset_TA = tf.TensorArray(tf.int32, size=max_iters, clear_after_read=False)

        def cond(i, *_):
            return i < max_iters

        def body(i, fTA, gTA, aTA, eTA, mTA, qTA, dampTA, flipTA, angTA, stallTA, nfTA):
            # ----------------------------------------------------------------------
            # Set gamma either by Barzilai_Borwein or last-direction-based and clamp
            # ----------------------------------------------------------------------
            gamma_o = _initial_gamma_opt(self.gamma_mode, self.gamma_init,
                                         self.S, self.Y, self.mem_size, self.g, self.d_prev,
                                         self.gam_lo, self.gam_hi, self.eps_div)

            # ----------------------------------------------------------------------
            # Set the direction by two-loop L-BFGS using existing pairs (s,y) and clap rho is needed
            # ----------------------------------------------------------------------
            d_o, used_rho_cap = _two_loop_opt(self.S, self.Y,
                                              self.mem_size, self.m, gamma_o, self.g, self.eps_rho)
            self.rho_cap_count.assign_add(tf.cast(used_rho_cap, tf.int32))

            # ----------------------------------------------------------------------
            # Fall back to steepest descent if d is non-finite
            # ----------------------------------------------------------------------
            all_finite_d = tf.reduce_all(tf.math.is_finite(d_o))
            d_o = tf.cond(all_finite_d, lambda: d_o, lambda: -tf.cast(self.g, self.dtype_opt))

            # ----------------------------------------------------------------------
            # Repair direction if g^T d >= -tau
            # ----------------------------------------------------------------------
            g_o = tf.cast(self.g, self.dtype_opt)
            gTd_o = _dot(g_o, d_o)
            g2_o = _dot(g_o, g_o)
            tau_o = tf.cast(10.0, self.dtype_opt) * self.eps_q

            def _make_descent_direction_old():
                eps_proj = self.eps_q
                coef = (gTd_o / g2_o) + eps_proj
                d_new = d_o - coef * g_o
                return tf.where(g2_o > self.eps_q, d_new, -g_o)

            def _make_descent_direction():
                kappa = tf.cast(0.1, self.dtype_opt)  # e.g., 0.1 * ||g||^2
                return d_o - ((gTd_o + kappa * g2_o) / g2_o) * g_o

            repair_direction = gTd_o >= -tau_o
            d_o = tf.cond(repair_direction, _make_descent_direction, lambda: d_o)
            self.descent_repair_count.assign_add(tf.cast(repair_direction, tf.int32))
            _ = _debug_assert_list(self.debug_checks, [d_o])

            # ----------------------------------------------------------------------
            # Check for inf/nan only if debug_checks is True
            # ----------------------------------------------------------------------
            pre_mask, gTd_dbg = _probe_pre_ls_flagged(self.debug_checks, self.f, self.g, d_o)
            tf.cond(self.debug_checks, lambda: tf.print("[dbg pre-LS] mask=", pre_mask, " g·d=", gTd_dbg), lambda: 0)

            # ----------------------------------------------------------------------
            # Perform the line search
            # with warm start for geometric search (not for cubic search)
            # ----------------------------------------------------------------------
            alpha0_o = tf.maximum(self.alpha_floor,
                                  tf.minimum(tf.cast(1.0, self.dtype_opt), 2.0 * self.alpha_prev))

            # line search
            if self.line_search == "armijo":
                alpha_o, f_new_m, g_new_m, evals, backs, armijo_ok = armijo_engine(
                    self.loss_and_grad, self.x, self.f, self.g, d_o,
                    self.f_hist, self.f_hist_size,
                    alpha0_o, self.c1, self.max_evals,
                    self.window, self.armijo_use_cubic,
                    self.bt, self.tol_alpha,  # tolx
                    self.armijo_max_norm_d,
                    self.armijo_use_wolfe, self.c2
                    #self.eps_div                         # dtype-aware epsilon for alamin guard
                )
                evals_or_backs = backs
            else:
                alpha_o, f_new_m, g_new_m, evals = hager_zhang(
                    self.loss_and_grad, self.x, self.f, self.g, d_o,
                    alpha0_o, self.c1, self.c2, self.max_evals, self.tol_alpha,
                    self.eps_div  # (may be ignored inside HZ)
                )
                armijo_ok = False  # not used
                evals_or_backs = evals

            # ----------------------------------------------------------------------
            # Track α==0 stalls
            # ----------------------------------------------------------------------
            is_zero_alpha = tf.equal(alpha_o, tf.cast(0.0, self.dtype_opt))
            self.zero_alpha_streak.assign(
                tf.where(is_zero_alpha, self.zero_alpha_streak + 1, tf.constant(0, tf.int32))
            )

            def _on_stall():
                # reset memory
                self.S.assign(tf.zeros_like(self.S))
                self.Y.assign(tf.zeros_like(self.Y))
                self.Q.assign(tf.zeros_like(self.Q))
                self.mem_size.assign(0)
                # conservative next warm-start & bias -g
                self.alpha_prev.assign(self.alpha_floor)
                self.d_prev.assign(-tf.cast(self.g.read_value(), self.dtype_model))
                # counters
                self.zero_alpha_streak.assign(0)
                self.alpha_zero_resets.assign_add(1)
                # force α=0 this iter & emit flag
                return tf.cast(0.0, self.dtype_opt), tf.constant(1, tf.int32)

            alpha_o, stall_flag = tf.cond(
                self.zero_alpha_streak >= self.stall_reset_K,
                _on_stall,
                lambda: (alpha_o, tf.constant(0, tf.int32))
            )

            # ----------------------------------------------------------------------
            # Check for inf/nan only if debug_checks is True
            # ----------------------------------------------------------------------
            post_mask = _probe_post_ls_flagged(self.debug_checks, f_new_m, g_new_m, d_o.dtype)
            tf.cond(self.debug_checks, lambda: tf.print("[dbg post-LS] mask=", post_mask), lambda: 0)

            # ----------------------------------------------------------------------
            # Bail if new values are non-finite / Nan:
            # Reset memory and keep x constant. This will force to restart with a line search with
            # a steepest descent direction using the last clean gradient
            # ----------------------------------------------------------------------
            bad_new = tf.logical_or(
                tf.logical_not(tf.math.is_finite(f_new_m)),
                tf.logical_not(tf.reduce_all(tf.math.is_finite(g_new_m)))
            )

            def _clear_hist_no_move():
                self.S.assign(tf.zeros_like(self.S))
                self.Y.assign(tf.zeros_like(self.Y))
                self.Q.assign(tf.zeros_like(self.Q))
                self.mem_size.assign(0)
                self.nonfinite_resets.assign_add(1)
                return tf.constant(0.0, self.dtype_opt), self.f.read_value(), self.g.read_value(), tf.constant(1,
                                                                                                               tf.int32)

            alpha_o, f_new_m, g_new_m, nf_flag = tf.cond(
                bad_new,
                _clear_hist_no_move,
                lambda: (alpha_o, f_new_m, g_new_m, tf.constant(0, tf.int32))
            )

            # ----------------------------------------------------------------------
            # check for inf/Nan if debugs_checks=True (works whether eager or graph mode)
            # ----------------------------------------------------------------------
            _ = _debug_assert_list(self.debug_checks, [f_new_m, g_new_m])

            # ----------------------------------------------------------------------
            # Set the new point
            # ----------------------------------------------------------------------
            x_new_m = self.x + tf.cast(alpha_o, self.dtype_model) * tf.cast(d_o, self.dtype_model)
            s_m = x_new_m - self.x
            y_m = g_new_m - self.g

            # precompute norms
            s_s_m = _dot(s_m, s_m)
            s_norm_m = tf.sqrt(tf.maximum(0., s_s_m))

            # ----------------------------------------------------------------------
            # Powell damping if requested
            # ----------------------------------------------------------------------
            y_damped_m, sTy_used_o, theta_o, flipped = _powell_damp_ref_opt(
                s_m, y_m, tf.cast(self.eps_curv, self.dtype_opt), self.powell_delta
            )
            powel_damped_this_iter = theta_o < 1.0


            y_damped_o = tf.cast(y_damped_m, self.dtype_opt)
            # curvature stats
            s_o = tf.cast(s_m, self.dtype_opt)
            sTy_o = _dot(s_o, y_damped_o)
            y_s_o = _dot(y_damped_o, y_damped_o)
            y_norm_o_pre = tf.sqrt(tf.maximum(0., y_s_o))
            thresh_o = self.eps_curv * tf.cast(s_norm_m, self.dtype_opt) * y_norm_o_pre

            # ----------------------------------------------------------------------
            # Flip y to -y if requested if the curvature condition is not met
            # ----------------------------------------------------------------------
            def _use_normal():
                return y_damped_o, sTy_o, y_norm_o_pre

            def _use_auto():
                flip = sTy_o < -thresh_o

                def _flip():
                    y_neg = -y_damped_o
                    return y_neg, -sTy_o, y_norm_o_pre

                def _keep():
                    return y_damped_o, sTy_o, y_norm_o_pre

                return tf.cond(flip, _flip, _keep)

            y_used_o, sTy_used_o, y_norm_o = tf.case(
                pred_fn_pairs=[
                    (tf.equal(self.y_sign_mode, 0), _use_normal),
                    (tf.equal(self.y_sign_mode, 1), _use_auto),
                ],
                default=_use_normal,
                exclusive=False
            )

            # ----------------------------------------------------------------------
            # Compute metrics for diagnostic
            # ----------------------------------------------------------------------
            flag_damped = tf.cast(self.powell > 0, tf.int32)
            flip_bool = tf.logical_and(tf.equal(self.y_sign_mode, 1), tf.less(sTy_o, -thresh_o))
            flag_flipped = tf.cast(flip_bool, tf.int32)

            d_new_o = tf.cast(d_o, self.dtype_opt)
            d_old_o = tf.cast(self.d_prev, self.dtype_opt)
            inner = _dot(d_new_o, d_old_o)
            denom = tf.maximum(_norm(d_new_o) * _norm(d_old_o), self.eps_q)
            cosine = tf.clip_by_value(inner / denom,
                                      tf.cast(-1.0, self.dtype_opt),
                                      tf.cast(1.0, self.dtype_opt))
            angle_pi = tf.acos(cosine) / tf.constant(3.141592653589793, self.dtype_opt)

            # ----------------------------------------------------------------------
            # Update the memory of s and y
            # ----------------------------------------------------------------------
            y_used_m = tf.cast(y_used_o, self.dtype_model)

            # accept pair?
            accept = sTy_used_o > self.eps_curv * tf.cast(s_norm_m, self.dtype_opt) * y_norm_o

            # quality
            q_new_o = sTy_used_o / tf.maximum(tf.cast(s_norm_m, self.dtype_opt) * y_norm_o, self.eps_q)
            q_new_m = tf.cast(q_new_o, self.dtype_model)

            curv_mask = _probe_curvature_flagged(self.debug_checks, s_m, tf.cast(y_used_o, self.dtype_model),
                                                 sTy_used_o, q_new_o)
            tf.cond(self.debug_checks, lambda: tf.print("[dbg curvature] mask=", curv_mask,
                                                        " sTy=", sTy_used_o, " q=", q_new_o), lambda: 0)

            def upd_fifo():
                return _append_fifo_with_q(self.S, self.Y, self.Q, self.mem_size, s_m, y_used_m, q_new_m, self.m)

            def upd_qp():
                return _append_quality_prune_with_q(self.S, self.Y, self.Q, self.mem_size, s_m, y_used_m, q_new_m,
                                                    self.m)

            S2, Y2, Q2, ms2 = tf.cond(accept,
                                      lambda: (upd_fifo() if self.memory_update == "fifo" else upd_qp()),
                                      lambda: (self.S, self.Y, self.Q, self.mem_size))

            if self.debug_print and tf.executing_eagerly():
                dtype_o = self.dtype_opt

                # g·d and norms in optimizer dtype
                g_o = tf.cast(self.g, dtype_o)
                d_o_used = tf.cast(d_o, dtype_o)  # the direction used by LS
                gTd = _dot(g_o, d_o_used)
                gnorm = tf.sqrt(tf.maximum(_dot(g_o, g_o), tf.cast(0.0, dtype_o)))
                dnorm = tf.sqrt(tf.maximum(_dot(d_o_used, d_o_used), tf.cast(0.0, dtype_o)))
                denom = tf.maximum(gnorm * dnorm, tf.cast(1e-30, dtype_o))

                # ----- iter index for printing (starts at 1) -----
                iter_idx = self._iter_counter.assign_add(1)  # returns the incremented value

                # ----- compute angle between previous and current L-BFGS directions -----
                d_curr = tf.cast(d_o, dtype_o)  # the *accepted* search direction used by LS

                # lazily create storage on the first call (shape from current d)
                if self._prev_d is None:
                    self._prev_d = tf.Variable(tf.zeros_like(d_curr), trainable=False)

                d_prev = tf.cast(self._prev_d, dtype_o)

                # safe dot/norms
                ddot = _dot(d_prev, d_curr)
                nprev = tf.sqrt(tf.maximum(_dot(d_prev, d_prev), tf.cast(0.0, dtype_o)))
                ncurr = tf.sqrt(tf.maximum(_dot(d_curr, d_curr), tf.cast(0.0, dtype_o)))
                denom = tf.maximum(nprev * ncurr, tf.cast(1e-30, dtype_o))

                # angle/pi (unitless, matches ref)
                pi = tf.constant(3.141592653589793, dtype_o)
                angle_over_pi = tf.acos(tf.clip_by_value(ddot / denom, -1.0, 1.0)) / pi

                # first printed iteration should show NaN like ref
                angle_over_pi = tf.where(self._have_prev_d,
                                         angle_over_pi,
                                         tf.cast(float('nan'), dtype_o))

                # string with 5 decimals to match ref formatting
                angle_s = tf.strings.as_string(tf.cast(angle_over_pi, tf.float64), precision=5)

                # ----- curvature diagnostics you already have -----
                neg_sTy = (~tf.math.is_finite(sTy_used_o)) | (sTy_used_o <= 0.0)
                pair_ok = tf.math.is_finite(sTy_used_o) & (sTy_used_o > self.eps_curv)

                loss_s = tf.strings.as_string(tf.sqrt(tf.maximum(tf.cast(f_new_m, tf.float64), 0.0)), precision=6)  # "0.012345"
                alpha_s = tf.strings.as_string(tf.cast(alpha_o, tf.float64), scientific=True,
                                               precision=4)  # "1.0000e+00"
                gnorm_s = tf.strings.as_string(tf.cast(gnorm, tf.float64), scientific=True, precision=4)  # "5.8968e-02"

                tf.print(
                    "Iter", tf.cast(iter_idx, tf.int32),
                    ": t=", "n/a", "s, , ",
                    "angle=", angle_s,
                    ", loss=", loss_s,
                    ", d.g>0:", _dot(tf.cast(g_new_m, dtype_o), d_curr) > 0.0,  # keep your original if you prefer
                    ", armijo:", armijo_ok,
                    ", alpha=", alpha_s,
                    ", iter=", tf.cast(evals, tf.int32),
                    ", |g|=", gnorm_s,
                    ", neg_sTy=", flipped,
                    ", powel=", powel_damped_this_iter,
                    ", pair ok:", pair_ok,
                    ", mem=", tf.cast(ms2, tf.int32),
                    sep=""
                )

                self._prev_d.assign(d_curr)
                self._have_prev_d.assign(True)

            # commit state
            self.x.assign(x_new_m)
            self.f.assign(f_new_m)
            self.g.assign(g_new_m)
            self.g_norm.assign(_norm(self.g))
            self.S.assign(S2)
            self.Y.assign(Y2)
            self.Q.assign(Q2)
            self.mem_size.assign(ms2)
            self.alpha_prev.assign(alpha_o)
            self.d_prev.assign(tf.cast(d_o, self.dtype_model))

            # ----------------------------------------------------------------------
            # check for inf/Nan if debugs_checks=True (works whether eager or graph mode)
            # ----------------------------------------------------------------------
            _ = _debug_assert_list(self.debug_checks, [self.x.read_value(), self.f.read_value(), self.g.read_value()])

            # update Armijo history window
            def upd_fhist():
                size = self.f_hist_size

                def not_full():
                    idx = tf.cast(size, tf.int32)
                    self.f_hist.assign(tf.tensor_scatter_nd_update(self.f_hist, [[idx]], [f_new_m]))
                    self.f_hist_size.assign(size + 1)
                    return 0

                def full():
                    self.f_hist.assign(tf.concat([self.f_hist[1:], tf.expand_dims(f_new_m, 0)], axis=0))
                    return 0

                return tf.cond(size < self.window, not_full, full)

            tf.cond(tf.constant(True), upd_fhist, lambda: tf.constant(0))

            return (i + 1,
                    fTA.write(i, f_new_m),
                    gTA.write(i, self.g_norm.read_value()),
                    aTA.write(i, self.alpha_prev.read_value()),
                    eTA.write(i, evals_or_backs),
                    mTA.write(i, self.mem_size.read_value()),
                    qTA.write(i, q_new_m),
                    dampTA.write(i, flag_damped),
                    flipTA.write(i, flag_flipped),
                    angTA.write(i, angle_pi),
                    stallTA.write(i, stall_flag),
                    nfTA.write(i, nf_flag))

        (_, f_TA, gnorm_TA, alpha_TA, evals_TA, msize_TA, qual_TA,
         damped_TA, flipped_TA, angle_TA, stall_TA, nf_TA) = tf.while_loop(
            cond, body,
            loop_vars=(tf.constant(0, tf.int32), f_TA, gnorm_TA, alpha_TA, evals_TA, msize_TA, qual_TA,
                       damped_TA, flipped_TA, angle_TA, stallreset_TA, nfreset_TA),
            maximum_iterations=max_iters, parallel_iterations=1
        )

        history = {
            "f": f_TA.stack(),
            "g_norm": gnorm_TA.stack(),
            "alpha": alpha_TA.stack(),  # opt dtype
            "evals_or_backs": evals_TA.stack(),
            "m": msize_TA.stack(),
            "quality": qual_TA.stack(),  # model dtype
            "damped": damped_TA.stack(),  # int32
            "flipped": flipped_TA.stack(),  # int32
            "angle_pi": angle_TA.stack(),  # opt dtype
            "reset_alpha_stall": stall_TA.stack(),  # int32 flag per iter
            "reset_nonfinite": nf_TA.stack(),  # int32 flag per iter
        }

        return {
            "x": self.x.read_value(),
            "f": self.f.read_value(),
            "g_norm": self.g_norm.read_value(),
            "mem_size": self.mem_size.read_value(),
            "S": self.S[:self.mem_size],
            "Y": self.Y[:self.mem_size],
            "Q": self.Q[:self.mem_size],
            "history": history,
            # lifetime counters exposed (optional)
            "alpha_zero_resets": self.alpha_zero_resets.read_value(),
            "nonfinite_resets": self.nonfinite_resets.read_value(),
        }
