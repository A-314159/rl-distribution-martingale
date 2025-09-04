import tensorflow as tf
from utilities.tensorflow_config import tf_compile, HIGH, SENSITIVE_CALC
from optimizers.helpers import _dot, _norm
from optimizers.line_searches import nonmonotone_armijo, hager_zhang


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

@tf_compile
def _two_loop_opt(S_m, Y_m, mem_size, gamma_o, g_m, eps_div_o):
    """
    Two-loop recursion in accumulation dtype.
    Inputs S_m, Y_m, g_m are model dtype; we cast internally to opt dtype.
    """
    S = tf.cast(S_m, gamma_o.dtype)
    Y = tf.cast(Y_m, gamma_o.dtype)
    g = tf.cast(g_m, gamma_o.dtype)
    eps_div = tf.cast(eps_div_o, gamma_o.dtype)

    alpha_TA = tf.TensorArray(g.dtype, size=mem_size, clear_after_read=False)
    rho_TA = tf.TensorArray(g.dtype, size=mem_size, clear_after_read=False)

    def bwd_cond(i, _q, _aTA, _rTA):
        return i >= 0

    # noinspection PyShadowingNames
    # pylint: disable=shadowed-name
    def bwd_body(i, q, aTA, rTA):
        si = S[i]
        yi = Y[i]
        rho = 1.0 / tf.maximum(_dot(yi, si), eps_div)
        a = rho * _dot(si, q)
        q = q - a * yi
        return i - 1, q, aTA.write(i, a), rTA.write(i, rho)

    _, q, alpha_TA, rho_TA = tf.while_loop(
        bwd_cond, bwd_body,
        loop_vars=(mem_size - 1, g, alpha_TA, rho_TA),
        maximum_iterations=mem_size, parallel_iterations=1
    )

    r = gamma_o * q

    def fwd_cond(i, _r):
        return i < mem_size

    # noinspection PyShadowingNames
    # pylint: disable=shadowed-name
    def fwd_body(i, r):
        si = S[i]
        yi = Y[i]
        rho = rho_TA.read(i)
        a = alpha_TA.read(i)
        b = rho * _dot(yi, r)
        return i + 1, r + si * (a - b)

    _, r = tf.while_loop(
        fwd_cond, fwd_body, loop_vars=(0, r),
        maximum_iterations=mem_size, parallel_iterations=1
    )
    return r  # opt dtype


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

    def bb():
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
    def dmatch():
        gTd = _dot(g, d_prev)
        g2 = _dot(g, g)
        gam = -gTd / tf.maximum(g2, eps_div)
        return tf.clip_by_value(gam, gam_lo, gam_hi)

    return tf.case(
        pred_fn_pairs=[
            (tf.equal(mode_code, 0), const),
            (tf.logical_and(tf.equal(mode_code, 1), mem_size > 0), bb),
            (tf.logical_and(tf.equal(mode_code, 2), _norm(d_prev) > 0.0), dmatch),
        ],
        default=const, exclusive=False
    )


@tf_compile
def _powell_damp_with_ss_opt(s_m, y_m, gamma_o, s_s_m, gam_lo_o, eps_div_o):
    s = tf.cast(s_m, gamma_o.dtype)
    y = tf.cast(y_m, gamma_o.dtype)
    s_s = tf.cast(s_s_m, gamma_o.dtype)
    gam_lo = tf.cast(gam_lo_o, gamma_o.dtype)
    eps_div = tf.cast(eps_div_o, gamma_o.dtype)

    sTy = _dot(s, y)
    inv_g = 1.0 / tf.maximum(gamma_o, gam_lo)
    sBs = inv_g * s_s

    def no():
        return y

    def yes():
        theta = 0.8 * sBs / tf.maximum(sBs - sTy, eps_div)
        Bs = inv_g * s
        return theta * y + (1.0 - theta) * Bs

    return tf.cond(sTy >= 0.2 * sBs, no, yes)  # opt dtype


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

class LBFGS_GRAPH:
    """
    Initialize once. Call step(K) inside your own @tf.function training loop to advance K iterations.
    Supports:
      - line_search: "nonmonotone_armijo" or "hager_zhang"
      - memory_update: "fifo" or "quality_prune"
    """

    def __init__(self,
                 loss_and_grad, x0,
                 m=20,
                 line_search="nonmonotone_armijo",
                 memory_update="fifo",
                 armijo_c1=1e-4, armijo_window=5, backtrack_factor=0.5,
                 max_evals_per_iter=20, wolfe_c2=0.9,
                 powell_damping=True,
                 init_scaling="bb", init_gamma=1.0,
                 eps_curv=1e-12,
                 dtype=tf.float32,  # model dtype (default fp32)
                 opt_dtype=None,  # accumulation dtype (None → choose by model dtype)
                 debug_checks=False):  # runtime finite checks

        # dtypes
        self.dtype_model = dtype
        self.dtype_opt = dtype if opt_dtype is None else opt_dtype

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

        # warning: consider if the threshold levels should be dependent on dtype_model or dtype_opt
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
        elif self.dtype_opt == tf.float32:
            eps_div = 1e-32
            eps_q = 1e-8
            tol_alpha = 1e-8
            alpha_floor = 1e-6
        else:  # fp16/bf16 accumulation (uncommon)
            eps_div = 1e-7
            eps_q = 1e-4
            tol_alpha = 1e-4
            alpha_floor = 1e-3

        self.eps_div = tf.constant(eps_div, self.dtype_opt)
        self.eps_q = tf.constant(eps_q, self.dtype_opt)
        self.tol_alpha = tf.constant(tol_alpha, self.dtype_opt)
        self.alpha_floor = tf.constant(alpha_floor, self.dtype_opt)

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

            # Wrap user's loss_and_grad_list(var_list, need_gradient) -> (loss, grads_list)
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
            # tensor mode; user provided loss_and_grad(x) or loss_and_grad(x, need_grad)
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
        self.m = tf.constant(int(m), tf.int32)
        self.S = tf.Variable(tf.zeros([self.m, n], dtype=self.dtype_model), trainable=False)
        self.Y = tf.Variable(tf.zeros([self.m, n], dtype=self.dtype_model), trainable=False)
        self.mem_size = tf.Variable(0, dtype=tf.int32, trainable=False)
        self.Q = tf.Variable(tf.fill([self.m], tf.constant(0., self.dtype_model)), trainable=False)

        # Bookkeeping
        self.alpha_prev = tf.Variable(tf.cast(1.0, self.dtype_opt), dtype=self.dtype_opt, trainable=False)  # opt dtype
        self.d_prev = tf.Variable(tf.zeros_like(self.x), trainable=False)  # model dtype

        self.f_hist = tf.Variable(tf.fill([self.window], tf.cast(self.f, self.dtype_model)), trainable=False)
        self.f_hist_size = tf.Variable(1, dtype=tf.int32, trainable=False)

    @tf_compile
    def step(self, iters: tf.Tensor):
        """
        Advance 'iters' iterations. Returns current state and per-iteration history for this call.
        """

        max_iters = tf.cast(iters, tf.int32)

        f_TA = tf.TensorArray(self.dtype_model, size=max_iters, clear_after_read=False)
        # noinspection SpellCheckingInspection
        gnorm_TA = tf.TensorArray(self.dtype_model, size=max_iters, clear_after_read=False)
        alpha_TA = tf.TensorArray(self.dtype_opt, size=max_iters, clear_after_read=False)
        evals_TA = tf.TensorArray(tf.int32, size=max_iters, clear_after_read=False)
        # noinspection SpellCheckingInspection
        msize_TA = tf.TensorArray(tf.int32, size=max_iters, clear_after_read=False)
        # noinspection SpellCheckingInspection
        qual_TA = tf.TensorArray(self.dtype_model, size=max_iters, clear_after_read=False)

        def cond(i, _fTA, _gTA, _aTA, _eTA, _mTA, _qTA):
            return i < max_iters

        def body(i, fTA, gTA, aTA, eTA, mTA, qTA):
            gamma_o = _initial_gamma_opt(self.gamma_mode, self.gamma_init,
                                         self.S, self.Y, self.mem_size, self.g, self.d_prev,
                                         self.gam_lo, self.gam_hi, self.eps_div)

            d_o = -_two_loop_opt(self.S[:self.mem_size], self.Y[:self.mem_size],
                                 self.mem_size, gamma_o, self.g, self.eps_div)

            # If the two-loop produced any non-finite, fall back to steepest descent (opt dtype)
            all_finite_d = tf.reduce_all(tf.math.is_finite(d_o))
            d_o = tf.cond(all_finite_d, lambda: d_o, lambda: -tf.cast(self.g, self.dtype_opt))

            # Debug: direction must be finite
            _ = _debug_assert_list(self.debug_checks, [d_o])

            alpha0_o = tf.maximum(self.alpha_floor, tf.minimum(tf.cast(1.0, self.dtype_opt), 2.0 * self.alpha_prev))

            if self.line_search == "nonmonotone_armijo":
                alpha_o, f_new_m, g_new_m, evals, backs = nonmonotone_armijo(
                    self.loss_and_grad, self.x, self.f, self.g, d_o,
                    self.f_hist, self.f_hist_size,
                    alpha0_o, self.c1, self.bt, self.max_evals
                )
                evals_or_backs = backs
            else:
                alpha_o, f_new_m, g_new_m, evals = hager_zhang(
                    self.loss_and_grad, self.x, self.f, self.g, d_o,
                    alpha0_o, self.c1, self.c2, self.max_evals, self.tol_alpha
                )
                evals_or_backs = evals

            # Bail if new values are non-finite
            bad_new = tf.logical_or(
                tf.logical_not(tf.math.is_finite(f_new_m)),
                tf.logical_not(tf.reduce_all(tf.math.is_finite(g_new_m)))
            )

            def _clear_hist_no_move():
                self.S.assign(tf.zeros_like(self.S))
                self.Y.assign(tf.zeros_like(self.Y))
                self.Q.assign(tf.zeros_like(self.Q))
                self.mem_size.assign(0)
                return tf.constant(0.0, self.dtype_opt), self.f.read_value(), self.g.read_value()

            alpha_o, f_new_m, g_new_m = tf.cond(
                bad_new,
                _clear_hist_no_move,
                lambda: (alpha_o, f_new_m, g_new_m)
            )

            # Debug: post-line-search values finite (when not forced into no-move)
            _ = _debug_assert_list(self.debug_checks, [f_new_m, g_new_m])

            x_new_m = self.x + tf.cast(alpha_o, self.dtype_model) * tf.cast(d_o, self.dtype_model)
            s_m = x_new_m - self.x
            y_m = g_new_m - self.g

            # Precompute once (model dtype)
            s_s_m = _dot(s_m, s_m)
            s_norm_m = tf.sqrt(tf.maximum(0., s_s_m))

            # Powell damping in opt, returns opt y; then cast back to model
            y_used_o = tf.cond(self.powell > 0,
                               lambda: _powell_damp_with_ss_opt(s_m, y_m, gamma_o, s_s_m, self.gam_lo, self.eps_div),
                               lambda: tf.cast(y_m, self.dtype_opt))
            y_used_m = tf.cast(y_used_o, self.dtype_model)

            # These are re-used by accept+log
            sTy_used_o = _dot(tf.cast(s_m, self.dtype_opt), y_used_o)
            y_s_used_o = _dot(y_used_o, y_used_o)
            y_norm_o = tf.sqrt(tf.maximum(0., y_s_used_o))

            # Accept pair?
            accept = sTy_used_o > self.eps_curv * tf.cast(s_norm_m, self.dtype_opt) * y_norm_o

            # quality metric in opt, store in model
            q_new_o = sTy_used_o / tf.maximum(tf.cast(s_norm_m, self.dtype_opt) * y_norm_o, self.eps_q)
            q_new_m = tf.cast(q_new_o, self.dtype_model)

            def upd_fifo():
                return _append_fifo_with_q(self.S, self.Y, self.Q, self.mem_size, s_m, y_used_m, q_new_m, self.m)

            def upd_qp():
                return _append_quality_prune_with_q(self.S, self.Y, self.Q, self.mem_size, s_m, y_used_m, q_new_m,
                                                    self.m)

            S2, Y2, Q2, ms2 = tf.cond(accept,
                                      lambda: (upd_fifo() if self.memory_update == "fifo" else upd_qp()),
                                      lambda: (self.S, self.Y, self.Q, self.mem_size))

            # Commit state
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

            # Debug: committed tensors finite
            _ = _debug_assert_list(self.debug_checks, [self.x.read_value(), self.f.read_value(), self.g.read_value()])

            # Update f-history window for Armijo
            # noinspection SpellCheckingInspection
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
                    qTA.write(i, q_new_m))

        # noinspection SpellCheckingInspection
        _, f_TA, gnorm_TA, alpha_TA, evals_TA, msize_TA, qual_TA = tf.while_loop(
            cond, body,
            loop_vars=(tf.constant(0, tf.int32), f_TA, gnorm_TA, alpha_TA, evals_TA, msize_TA, qual_TA),
            maximum_iterations=max_iters, parallel_iterations=1
        )

        history = {
            "f": f_TA.stack(),
            "g_norm": gnorm_TA.stack(),
            "alpha": alpha_TA.stack(),  # opt dtype
            "evals_or_backs": evals_TA.stack(),
            "m": msize_TA.stack(),
            "quality": qual_TA.stack()  # model dtype
        }

        return {
            "x": self.x.read_value(),
            "f": self.f.read_value(),
            "g_norm": self.g_norm.read_value(),
            "mem_size": self.mem_size.read_value(),
            "S": self.S[:self.mem_size],
            "Y": self.Y[:self.mem_size],
            "Q": self.Q[:self.mem_size],
            "history": history
        }
