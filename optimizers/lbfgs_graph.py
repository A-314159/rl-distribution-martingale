import tensorflow as tf
from utilities.tensorflow_config import tf_compile


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


# ======================= small math helpers ==================================

@tf_compile
def _dot(a, b):
    return tf.tensordot(a, b, axes=1)


@tf_compile
def _norm(a):
    return tf.sqrt(tf.maximum(0.0, _dot(a, a)))


# ======================== L-BFGS primitives ==================================

@tf_compile
def _two_loop(S, Y, mem_size, gamma, g, eps_div):
    """L-BFGS two-loop recursion, all in TF control flow."""
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

    r = gamma * q

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
    return r


@tf_compile
def _initial_gamma(mode_code, init_gamma, S, Y, mem_size, g, d_prev, gam_lo, gam_hi, eps_div):
    def const():
        return tf.clip_by_value(init_gamma, gam_lo, gam_hi)

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


# ======================== L-BFGS primitives ==================================


@tf_compile
def _powell_damp_with_ss(s, y, gamma, s_s, gam_lo, eps_div):
    sTy = _dot(s, y)
    inv_g = 1.0 / tf.maximum(gamma, gam_lo)  # use clamp floor
    sBs = inv_g * s_s

    def no():
        return y

    def yes():
        theta = 0.8 * sBs / tf.maximum(sBs - sTy, eps_div)
        Bs = inv_g * s
        return theta * y + (1.0 - theta) * Bs

    return tf.cond(sTy >= 0.2 * sBs, no, yes)


@tf_compile
def _append_fifo_with_q(S, Y, Q, mem_size, s_new, y_new, q_new, m):
    def not_full():
        idx = tf.cast(mem_size, tf.int32)
        S2 = tf.tensor_scatter_nd_update(S, [[idx]], [s_new])
        Y2 = tf.tensor_scatter_nd_update(Y, [[idx]], [y_new])
        Q2 = tf.tensor_scatter_nd_update(Q, [[idx]], [q_new])
        return S2, Y2, Q2, mem_size + 1

    def full():
        S2 = tf.concat([S[1:], tf.expand_dims(s_new, 0)], 0)
        Y2 = tf.concat([Y[1:], tf.expand_dims(y_new, 0)], 0)
        Q2 = tf.concat([Q[1:], tf.expand_dims(q_new, 0)], 0)
        return S2, Y2, Q2, mem_size

    return tf.cond(mem_size < m, not_full, full)


@tf_compile
def _append_quality_prune(S, Y, mem_size, s_new, y_new, m):
    def not_full():
        idx = tf.cast(mem_size, tf.int32)
        S2 = tf.tensor_scatter_nd_update(S, indices=[[idx]], updates=[s_new])
        Y2 = tf.tensor_scatter_nd_update(Y, indices=[[idx]], updates=[y_new])
        return S2, Y2, mem_size + 1

    # noinspection SpellCheckingInspection

    def full():
        dot_sy = tf.reduce_sum(S * Y, axis=1)
        qual = dot_sy / tf.maximum(tf.norm(S, axis=1) * tf.norm(Y, axis=1), 1e-12)
        worst = tf.argmin(qual, output_type=tf.int32)
        S2 = tf.tensor_scatter_nd_update(S, indices=[[worst]], updates=[s_new])
        Y2 = tf.tensor_scatter_nd_update(Y, indices=[[worst]], updates=[y_new])
        return S2, Y2, mem_size

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


# =================== Line searches ===================

@tf_compile
def _nonmonotone_armijo(loss_and_grad, x, f, g, d,
                        f_hist, f_hist_size,
                        alpha0, c1, backtrack, max_evals):
    gTd = _dot(g, d)
    tf_false = tf.constant(False)
    tf_true = tf.constant(True)

    def fail_dir():
        # Not a descent direction → no search, no move.
        return tf.constant(0.0, x.dtype), f, g, tf.constant(0, tf.int32), tf.constant(0, tf.int32)

    def search():
        # same as your code, but we track `accepted`
        f_ref = tf.cond(f_hist_size > 0,
                        lambda: tf.reduce_max(f_hist[:f_hist_size]),
                        lambda: f)
        rhs_scale = c1 * gTd

        # noinspection PyShadowingNames
        # pylint: disable=shadowed-name
        # noinspection SpellCheckingInspection
        def cond(_alpha, _f_acc, evals, _backs, accepted):
            return tf.logical_and(evals < max_evals, tf.logical_not(accepted))

        # noinspection PyShadowingNames
        # pylint: disable=shadowed-name
        # noinspection SpellCheckingInspection
        def body(alpha, f_acc, evals, backs, _accepted):
            x_try = x + alpha * d
            f_try, _ = loss_and_grad(x_try, tf_false)
            ok = f_try <= f_ref + alpha * rhs_scale
            alpha_next = tf.where(ok, alpha, alpha * backtrack)
            evals_next = evals + 1
            backs_next = tf.where(ok, backs, backs + 1)
            f_acc_next = tf.where(ok, f_try, f_acc)
            return alpha_next, f_acc_next, evals_next, backs_next, ok

        alpha_fin, f_acc, evals_fin, backs_fin, accepted = tf.while_loop(
            cond, body,
            loop_vars=(alpha0, f, tf.constant(0, tf.int32), tf.constant(0, tf.int32), tf_false),
            maximum_iterations=max_evals, parallel_iterations=1
        )

        def on_accept():
            x_new = x + alpha_fin * d
            f_new, g_new = loss_and_grad(x_new, tf_true)  # grad computed ONCE
            return alpha_fin, f_new, g_new, evals_fin, backs_fin

        def on_fail():
            # No Armijo step found → no move (or implement a tiny fallback step if you prefer)
            return tf.constant(0.0, x.dtype), f, g, evals_fin, backs_fin

        return tf.cond(accepted, on_accept, on_fail)

    # ← This is where fail_dir() is used.
    return tf.cond(gTd < 0.0, search, fail_dir)


@tf_compile
def _hager_zhang(loss_and_grad, x, f, g, d,
                 alpha0, c1, c2, max_evals, tol_alpha):
    """
    Exact Hager–Zhang strong Wolfe line search (bracket + zoom).
    Returns: alpha, f_new, g_new, evals
    """
    dtype = x.dtype
    tf_true = tf.constant(True)

    g0d = _dot(g, d)  # ← descent check
    c1g0d = c1 * g0d
    neg_c2g0d = -c2 * g0d
    f0 = f
    tol_alpha = tf.convert_to_tensor(tol_alpha, dtype)

    def fail_dir():
        # Not a descent direction: skip search, no move.
        return tf.constant(0.0, dtype), f0, g, tf.constant(0, tf.int32)

    def run_hz_search():
        a_prev = tf.constant(0.0, dtype)
        f_prev = f0
        g_prev = g

        a_cur = tf.convert_to_tensor(alpha0, dtype)
        f_cur, g_cur = loss_and_grad(x + a_cur * d, tf_true)
        f_cur = tf.convert_to_tensor(f_cur, dtype)
        g_cur = tf.convert_to_tensor(g_cur, dtype)
        evals = tf.constant(1, tf.int32)

        a_lo = tf.constant(0.0, dtype)
        f_lo = f0
        g_lo = g

        a_hi = tf.constant(0.0, dtype)
        f_hi = f0
        g_hi = g

        bracketed = tf.constant(False)

        # noinspection PyShadowingNames
        # pylint: disable=shadowed-name
        # noinspection SpellCheckingInspection
        def bcond(_a_prev, _f_prev, _g_prev,
                  _a_cur, _f_cur, _g_cur,
                  _a_lo, _f_lo, _g_lo,
                  _a_hi, _f_hi, _g_hi,
                  bracketed, evals):
            return tf.logical_and(tf.logical_not(bracketed), evals < max_evals)

        # noinspection PyShadowingNames
        # pylint: disable=shadowed-name
        # noinspection SpellCheckingInspection
        def bbody(a_prev, f_prev, g_prev,
                  a_cur, f_cur, g_cur,
                  a_lo, f_lo, g_lo,
                  a_hi, f_hi, g_hi,
                  bracketed, evals):
            gcurd = _dot(g_cur, d)
            cond1 = tf.logical_or(f_cur > f0 + a_cur * c1g0d, f_cur >= f_prev)
            cond2 = tf.abs(gcurd) <= neg_c2g0d
            cond3 = gcurd >= 0.0

            def case1():
                return (a_prev, f_prev, g_prev,
                        a_cur, f_cur, g_cur,
                        a_prev, f_prev, g_prev,
                        a_cur, f_cur, g_cur,
                        tf.constant(True), evals)

            def case2():
                return (a_cur, f_cur, g_cur,
                        a_cur, f_cur, g_cur,
                        a_cur, f_cur, g_cur,
                        a_cur, f_cur, g_cur,
                        tf.constant(True), evals)

            def case3():
                a_n = a_cur * 2.0
                f_n, g_n = loss_and_grad(x + a_n * d, tf_true)
                f_n = tf.convert_to_tensor(f_n, dtype)
                g_n = tf.convert_to_tensor(g_n, dtype)
                return (a_cur, f_cur, g_cur,
                        a_n, f_n, g_n,
                        a_lo, f_lo, g_lo,
                        a_hi, f_hi, g_hi,
                        bracketed, evals + 1)

            def case4():
                return (a_prev, f_prev, g_prev,
                        a_cur, f_cur, g_cur,
                        a_cur, f_cur, g_cur,
                        a_prev, f_prev, g_prev,
                        tf.constant(True), evals)

            return tf.case([(cond1, case1), (cond2, case2), (cond3, case4)],
                           default=case3, exclusive=False)

        (a_prev, f_prev, g_prev,
         a_cur, f_cur, g_cur,
         a_lo, f_lo, g_lo,
         a_hi, f_hi, g_hi,
         bracketed, evals) = tf.while_loop(
            bcond, bbody,
            loop_vars=(a_prev, f_prev, g_prev,
                       a_cur, f_cur, g_cur,
                       a_lo, f_lo, g_lo,
                       a_hi, f_hi, g_hi,
                       bracketed, evals),
            maximum_iterations=max_evals, parallel_iterations=1
        )

        a_star = tf.constant(0.0, dtype)
        f_star = f0
        g_star = g
        done = tf.constant(False)

        # noinspection PyShadowingNames
        # pylint: disable=shadowed-name
        # noinspection SpellCheckingInspection
        def zcond(_a_lo, _f_lo, _g_lo,
                  _a_hi, _f_hi, _g_hi,
                  evals, done, _a_star, _f_star, _g_star):
            small = tf.abs(_a_hi - _a_lo) < tol_alpha
            return tf.logical_and(tf.logical_not(done),
                                  tf.logical_and(evals < max_evals, tf.logical_not(small)))

        # noinspection PyShadowingNames
        # pylint: disable=shadowed-name
        # noinspection SpellCheckingInspection
        def zbody(a_lo, f_lo, g_lo,
                  a_hi, f_hi, g_hi,
                  evals, done, a_star, f_star, g_star):
            a = 0.5 * (a_lo + a_hi)
            f_try, g_try = loss_and_grad(x + a * d, tf_true)
            f_try = tf.convert_to_tensor(f_try, dtype)
            g_try = tf.convert_to_tensor(g_try, dtype)
            gtryd = _dot(g_try, d)

            cond1 = tf.logical_or(f_try > f0 + a * c1g0d, f_try >= f_lo)
            cond2 = tf.abs(gtryd) <= neg_c2g0d

            def case1():
                return a_lo, f_lo, g_lo, a, f_try, g_try, evals + 1, done, a_star, f_star, g_star

            def case2():
                return a, f_try, g_try, a_hi, f_hi, g_hi, evals + 1, tf.constant(True), a, f_try, g_try

            def case3():
                sign_change = gtryd * _dot(g_lo, d) < 0.0

                def flip():
                    return a, f_try, g_try, a_lo, f_lo, g_lo, evals + 1, done, a_star, f_star, g_star

                def keep():
                    return a, f_try, g_try, a_hi, f_hi, g_hi, evals + 1, done, a_star, f_star, g_star

                return tf.cond(sign_change, flip, keep)

            return tf.case([(cond1, case1), (cond2, case2)], default=case3, exclusive=False)

        (a_lo, f_lo, g_lo,
         a_hi, f_hi, g_hi,
         evals, done, a_star, f_star, g_star) = tf.while_loop(
            zcond, zbody,
            loop_vars=(a_lo, f_lo, g_lo,
                       a_hi, f_hi, g_hi,
                       evals, done, a_star, f_star, g_star),
            maximum_iterations=max_evals, parallel_iterations=1
        )

        return a_star, f_star, g_star, evals

    # Descent guard at the top-level:
    return tf.cond(g0d < 0.0, run_hz_search, fail_dir)


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

class LBFGSStepper:
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
                 dtype=tf.float64):

        self.line_search = line_search
        self.memory_update = memory_update
        self.c1 = tf.constant(armijo_c1, dtype)
        self.window = tf.constant(int(armijo_window), tf.int32)
        self.bt = tf.constant(backtrack_factor, dtype)
        self.max_evals = tf.constant(int(max_evals_per_iter), tf.int32)
        self.c2 = tf.constant(wolfe_c2, dtype)
        self.powell = tf.constant(1 if powell_damping else 0, tf.int32)
        self.gamma_mode = tf.constant({"constant": 0, "bb": 1, "direction_match": 2}[init_scaling], tf.int32)
        self.gamma_init = tf.constant(init_gamma, dtype)
        self.eps_curv = tf.constant(eps_curv, dtype)

        if dtype == tf.float64:
            lo, hi = 1e-6, 1e6
        elif dtype == tf.float32:
            lo, hi = 1e-4, 1e4
        else:  # e.g., bf16/fp16
            lo, hi = 1e-2, 1e2
        self.gam_lo = tf.constant(lo, dtype)
        self.gam_hi = tf.constant(hi, dtype)

        # Dtype-aware small constants
        if dtype == tf.float64:
            eps_div = 1e-32  # denominator floors for dot products
            eps_q = 1e-16  # floors for norms/products in quality metrics
            tol_alpha = 1e-12  # tiny interval for HZ zoom
            alpha_floor = 1e-8
        elif dtype == tf.float32:
            eps_div = 1e-32
            eps_q = 1e-8
            tol_alpha = 1e-8
            alpha_floor = 1e-6
        elif dtype == tf.float16:
            eps_div = 1e-7  # >= subnormal scale for fp16
            eps_q = 1e-4
            tol_alpha = 1e-4
            alpha_floor = 1e-3
        else:  # bfloat16
            eps_div = 1e-30  # representable in bf16; exponent is wide
            eps_q = 1e-4
            tol_alpha = 1e-4
            alpha_floor = 1e-3

        self.eps_div = tf.constant(eps_div, dtype)
        self.eps_q = tf.constant(eps_q, dtype)
        self.tol_alpha = tf.constant(tol_alpha, dtype)
        self.alpha_floor = tf.constant(alpha_floor, dtype)  # replaces earlier line

        if callable(loss_and_grad):
            _assert_compilation_mode(loss_and_grad, "loss_and_grad")
        else:
            raise TypeError("loss_and_grad must be callable")

        # Support tensor x0 OR list-of-variables x0
        if isinstance(x0, list):
            # list-of-variables mode
            self.var_list = x0
            self.assign_from_flat = make_assign_fn(self.var_list)
            x_flat = pack(self.var_list)

            # Wrap user's loss_and_grad_list(var_list, need_gradient) -> (loss, grads_list)
            @tf_compile
            def _loss_and_grad(x, need_gradient: tf.Tensor):
                _ = self.assign_from_flat(x)

                def with_grad():
                    f, g_list = loss_and_grad(self.var_list, tf.constant(True))
                    g_list = [gi if gi is not None else tf.zeros_like(vi)
                              for gi, vi in zip(g_list, self.var_list)]
                    return f, pack(g_list)

                def no_grad():
                    f, g_list = loss_and_grad(self.var_list, tf.constant(False))
                    # If the function returns None or ignores grads, just zero-fill
                    g_flat = tf.zeros_like(x)
                    return f, g_flat

                return tf.cond(need_gradient, with_grad, no_grad)

            self.loss_and_grad = _loss_and_grad
            self.x = tf.Variable(x_flat, dtype=dtype, trainable=False)
        else:
            # tensor mode; user provided loss_and_grad(x) or loss_and_grad(x, need_grad)
            if callable(loss_and_grad):
                try:
                    # Try two-arg signature
                    test = loss_and_grad(tf.convert_to_tensor(x0, dtype), tf.constant(True))
                    _ = test
                    self.loss_and_grad = loss_and_grad
                except TypeError:

                    @tf_compile
                    def _wrap(x, need_gradient: tf.Tensor):
                        f, g = loss_and_grad(x)

                        def with_grad():
                            return f, g

                        def no_grad():
                            return f, tf.zeros_like(x)

                        return tf.cond(need_gradient, with_grad, no_grad)

                    self.loss_and_grad = _wrap
            else:
                raise TypeError("loss_and_grad must be callable")

            self.x = tf.Variable(tf.identity(x0), dtype=dtype, trainable=False)

        # pick a dtype-friendly floor (you can set these in __init__ once)
        self.alpha_floor = tf.constant(1e-8 if self.x.dtype == tf.float64 else 1e-6, self.x.dtype)

        tf_true = tf.constant(True)
        tf_false = tf.constant(False)

        # Check gradients path
        fT, gT = self.loss_and_grad(self.x, tf_true)
        tf.debugging.assert_rank_at_least(fT, 0)
        tf.debugging.assert_equal(tf.shape(gT)[0], tf.shape(self.x)[0], message="grad size mismatch")
        tf.debugging.assert_all_finite(fT, "loss not finite at init")
        tf.debugging.assert_all_finite(gT, "grad not finite at init")

        # Check loss-only path returns a loss Tensor
        fF, gF = self.loss_and_grad(self.x, tf_false)
        tf.debugging.assert_rank_at_least(fF, 0)

        # Evaluate initial f,g
        f0, g0 = self.loss_and_grad(self.x, tf.constant(True))
        self.f = tf.Variable(tf.convert_to_tensor(f0, dtype), trainable=False)
        self.g = tf.Variable(tf.convert_to_tensor(g0, dtype), trainable=False)
        self.g_norm = tf.Variable(_norm(self.g), trainable=False)

        # Memory
        n = tf.shape(self.x)[0]
        self.m = tf.constant(int(m), tf.int32)
        self.S = tf.Variable(tf.zeros([self.m, n], dtype=dtype), trainable=False)
        self.Y = tf.Variable(tf.zeros([self.m, n], dtype=dtype), trainable=False)
        self.mem_size = tf.Variable(0, dtype=tf.int32, trainable=False)
        self.Q = tf.Variable(tf.fill([self.m], tf.constant(0., self.x.dtype)),
                             trainable=False)

        # Bookkeeping
        self.alpha_prev = tf.Variable(1.0, dtype=dtype, trainable=False)
        self.d_prev = tf.Variable(tf.zeros_like(self.x), trainable=False)

        self.f_hist = tf.Variable(tf.fill([self.window], tf.cast(self.f, dtype)), trainable=False)
        self.f_hist_size = tf.Variable(1, dtype=tf.int32, trainable=False)

    @tf_compile
    def step(self, iters: tf.Tensor):
        """
        Advance 'iters' iterations. Returns current state and per-iteration history for this call.
        """

        max_iters = tf.cast(iters, tf.int32)
        dtype = self.x.dtype

        f_TA = tf.TensorArray(dtype, size=max_iters, clear_after_read=False)
        # noinspection SpellCheckingInspection
        gnorm_TA = tf.TensorArray(dtype, size=max_iters, clear_after_read=False)
        alpha_TA = tf.TensorArray(dtype, size=max_iters, clear_after_read=False)
        evals_TA = tf.TensorArray(tf.int32, size=max_iters, clear_after_read=False)
        # noinspection SpellCheckingInspection
        msize_TA = tf.TensorArray(tf.int32, size=max_iters, clear_after_read=False)
        # noinspection SpellCheckingInspection
        qual_TA = tf.TensorArray(dtype, size=max_iters, clear_after_read=False)

        def cond(i, _fTA, _gTA, _aTA, _eTA, _mTA, _qTA):
            return i < max_iters

        def body(i, fTA, gTA, aTA, eTA, mTA, qTA):
            gamma = _initial_gamma(self.gamma_mode, self.gamma_init,
                                   self.S, self.Y, self.mem_size, self.g, self.d_prev,
                                   self.gam_lo, self.gam_hi, self.eps_div)

            d = -_two_loop(self.S[:self.mem_size], self.Y[:self.mem_size],
                           self.mem_size, gamma, self.g, self.eps_div)

            # If the two-loop produced any non-finite, fall back to steepest descent
            all_finite_d = tf.reduce_all(tf.math.is_finite(d))
            d = tf.cond(all_finite_d, lambda: d, lambda: -self.g)

            alpha0 = tf.maximum(self.alpha_floor, tf.minimum(1.0, 2.0 * self.alpha_prev))

            if self.line_search == "nonmonotone_armijo":
                alpha, f_new, g_new, evals, backs = _nonmonotone_armijo(
                    self.loss_and_grad, self.x, self.f, self.g, d,
                    self.f_hist, self.f_hist_size,
                    alpha0, self.c1, self.bt, self.max_evals
                )
                evals_or_backs = backs
            else:
                alpha, f_new, g_new, evals = _hager_zhang(
                    self.loss_and_grad, self.x, self.f, self.g, d,
                    alpha0, self.c1, self.c2, self.max_evals, self.tol_alpha
                )
                evals_or_backs = evals

            bad_new = tf.logical_or(
                tf.logical_not(tf.math.is_finite(f_new)),
                tf.logical_not(tf.reduce_all(tf.math.is_finite(g_new)))
            )

            def _clear_hist_no_move():
                # Clear curvature history to recover next iter
                self.S.assign(tf.zeros_like(self.S))
                self.Y.assign(tf.zeros_like(self.Y))
                self.Q.assign(tf.zeros_like(self.Q))
                self.mem_size.assign(0)
                # No move: keep current f,g; force alpha=0 so x_new == self.x
                return tf.constant(0.0, dtype), self.f.read_value(), self.g.read_value()

            alpha, f_new, g_new = tf.cond(
                bad_new,
                _clear_hist_no_move,
                lambda: (alpha, f_new, g_new)
            )
            x_new = self.x + alpha * d
            s = x_new - self.x
            y = g_new - self.g

            # Precompute once
            s_s = _dot(s, s)  # == ‖s‖^2
            s_norm = tf.sqrt(tf.maximum(0., s_s))  # ‖s‖

            # Powell damping (same behavior, but use s_s to avoid recompute)
            y_used = tf.cond(self.powell > 0,
                             lambda: _powell_damp_with_ss(s, y, gamma, s_s, self.gam_lo, self.eps_div),  # small helper below
                             lambda: y)

            # These are re-used by accept+log
            sTy_used = _dot(s, y_used)
            y_s_used = _dot(y_used, y_used)  # == ‖y_used‖^2
            y_norm = tf.sqrt(tf.maximum(0., y_s_used))

            # Accept pair?
            accept = sTy_used > self.eps_curv * s_norm * y_norm

            q_new = sTy_used / tf.maximum(s_norm * y_norm, self.eps_q)

            def upd_fifo():
                return _append_fifo_with_q(self.S, self.Y, self.Q, self.mem_size, s, y_used, q_new, self.m)

            def upd_qp():
                return _append_quality_prune_with_q(self.S, self.Y, self.Q, self.mem_size, s, y_used, q_new, self.m)

            S2, Y2, Q2, ms2 = tf.cond(accept,
                                      lambda: (upd_fifo() if self.memory_update == "fifo" else upd_qp()),
                                      lambda: (self.S, self.Y, self.Q, self.mem_size))

            # Commit state
            self.x.assign(x_new)
            self.f.assign(f_new)
            self.g.assign(g_new)
            self.g_norm.assign(_norm(self.g))
            self.S.assign(S2)
            self.Y.assign(Y2)
            self.Q.assign(Q2)
            self.mem_size.assign(ms2)
            self.alpha_prev.assign(alpha)
            self.d_prev.assign(d)

            # Update f-history window for Armijo
            # noinspection SpellCheckingInspection
            def upd_fhist():
                size = self.f_hist_size

                def not_full():
                    idx = tf.cast(size, tf.int32)
                    self.f_hist.assign(tf.tensor_scatter_nd_update(self.f_hist, [[idx]], [f_new]))
                    self.f_hist_size.assign(size + 1)
                    return 0

                def full():
                    self.f_hist.assign(tf.concat([self.f_hist[1:], tf.expand_dims(f_new, 0)], axis=0))
                    return 0

                return tf.cond(size < self.window, not_full, full)

            tf.cond(tf.constant(True), upd_fhist, lambda: tf.constant(0))

            return (i + 1,
                    fTA.write(i, f_new),
                    gTA.write(i, self.g_norm.read_value()),
                    aTA.write(i, alpha),
                    eTA.write(i, evals_or_backs),
                    mTA.write(i, self.mem_size.read_value()),
                    qTA.write(i, q_new))

        # noinspection SpellCheckingInspection
        _, f_TA, gnorm_TA, alpha_TA, evals_TA, msize_TA, qual_TA = tf.while_loop(
            cond, body,
            loop_vars=(tf.constant(0, tf.int32), f_TA, gnorm_TA, alpha_TA, evals_TA, msize_TA, qual_TA),
            maximum_iterations=max_iters, parallel_iterations=1
        )

        history = {
            "f": f_TA.stack(),
            "g_norm": gnorm_TA.stack(),
            "alpha": alpha_TA.stack(),
            "evals_or_backs": evals_TA.stack(),
            "m": msize_TA.stack(),
            "quality": qual_TA.stack()
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
