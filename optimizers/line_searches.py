import tensorflow as tf
from utilities.tensorflow_config import tf_compile
from optimizers.helpers import _dot, _norm
# =================== Line searches ====================

@tf_compile
def nonmonotone_armijo(loss_and_grad, x_m, f_m, g_m, d_o,
                       f_hist_m, f_hist_size,
                       alpha0_o, c1_o, backtrack_o, max_evals):
    """
    d and alpha are opt dtype; x,f,g,f_hist are model dtype.
    We cast only where needed; x_try/x_new are formed in model dtype.
    """
    gTd = _dot(tf.cast(g_m, d_o.dtype), d_o)
    tf_false = tf.constant(False)
    tf_true = tf.constant(True)

    def fail_dir():
        return tf.constant(0.0, d_o.dtype), f_m, g_m, tf.constant(0, tf.int32), tf.constant(0, tf.int32)

    def search():
        f_ref_m = tf.cond(f_hist_size > 0,
                          lambda: tf.reduce_max(f_hist_m[:f_hist_size]),
                          lambda: f_m)
        rhs_scale_o = c1_o * gTd

        # noinspection PyShadowingNames
        # pylint: disable=shadowed-name
        # noinspection SpellCheckingInspection
        def cond(_alpha_o, _f_acc_m, evals, _backs, accepted):
            return tf.logical_and(evals < max_evals, tf.logical_not(accepted))

        # noinspection PyShadowingNames
        # pylint: disable=shadowed-name
        # noinspection SpellCheckingInspection
        def body(alpha_o, f_acc_m, evals, backs, _accepted):
            x_try_m = x_m + tf.cast(alpha_o, x_m.dtype) * tf.cast(d_o, x_m.dtype)
            f_try_m, _ = loss_and_grad(x_try_m, tf_false)
            rhs_m = f_ref_m + tf.cast(alpha_o * rhs_scale_o, x_m.dtype)
            ok = f_try_m <= rhs_m

            alpha_next_o = tf.where(ok, alpha_o, alpha_o * backtrack_o)
            evals_next = evals + 1
            backs_next = tf.where(ok, backs, backs + 1)
            f_acc_next_m = tf.where(ok, f_try_m, f_acc_m)
            return alpha_next_o, f_acc_next_m, evals_next, backs_next, ok

        alpha_fin_o, f_acc_m, evals_fin, backs_fin, accepted = tf.while_loop(
            cond, body,
            loop_vars=(alpha0_o, f_m, tf.constant(0, tf.int32), tf.constant(0, tf.int32), tf.constant(False)),
            maximum_iterations=max_evals, parallel_iterations=1
        )

        def on_accept():
            x_new_m = x_m + tf.cast(alpha_fin_o, x_m.dtype) * tf.cast(d_o, x_m.dtype)
            f_new_m, g_new_m = loss_and_grad(x_new_m, tf_true)
            return alpha_fin_o, f_new_m, g_new_m, evals_fin, backs_fin

        def on_fail():
            return tf.constant(0.0, d_o.dtype), f_m, g_m, evals_fin, backs_fin

        return tf.cond(accepted, on_accept, on_fail)

    return tf.cond(gTd < 0.0, search, fail_dir)


@tf_compile
def hager_zhang(loss_and_grad, x_m, f_m, g_m, d_o,
                alpha0_o, c1_o, c2_o, max_evals, tol_alpha_o):
    """
    Strong Wolfe in opt dtype. x,f,g are model dtype; d,alpha,c1,c2,tol in opt.
    """
    dtype_o = d_o.dtype
    tf_true = tf.constant(True)

    g0d = _dot(tf.cast(g_m, dtype_o), d_o)
    c1g0d = c1_o * g0d
    neg_c2g0d = -c2_o * g0d
    f0_m = f_m
    tol_alpha_o = tf.convert_to_tensor(tol_alpha_o, dtype_o)

    def fail_dir():
        return tf.constant(0.0, dtype_o), f0_m, g_m, tf.constant(0, tf.int32)

    def run_hz_search():
        a_prev_o = tf.constant(0.0, dtype_o)
        f_prev_m = f0_m
        g_prev_m = g_m

        a_cur_o = tf.convert_to_tensor(alpha0_o, dtype_o)
        x_cur_m = x_m + tf.cast(a_cur_o, x_m.dtype) * tf.cast(d_o, x_m.dtype)
        f_cur_m, g_cur_m = loss_and_grad(x_cur_m, tf_true)
        evals = tf.constant(1, tf.int32)

        a_lo_o = tf.constant(0.0, dtype_o)
        f_lo_m = f0_m
        g_lo_m = g_m
        a_hi_o = tf.constant(0.0, dtype_o)
        f_hi_m = f0_m
        g_hi_m = g_m
        bracketed = tf.constant(False)

        # noinspection PyShadowingNames
        # pylint: disable=shadowed-name
        # noinspection SpellCheckingInspection
        def bcond(_a_prev_o, _f_prev_m, _g_prev_m,
                  _a_cur_o, _f_cur_m, _g_cur_m,
                  _a_lo_o, _f_lo_m, _g_lo_m,
                  _a_hi_o, _f_hi_m, _g_hi_m,
                  bracketed, evals):
            return tf.logical_and(tf.logical_not(bracketed), evals < max_evals)

        # noinspection PyShadowingNames
        # pylint: disable=shadowed-name
        # noinspection SpellCheckingInspection
        def bbody(a_prev_o, f_prev_m, g_prev_m,
                  a_cur_o, f_cur_m, g_cur_m,
                  a_lo_o, f_lo_m, g_lo_m,
                  a_hi_o, f_hi_m, g_hi_m,
                  bracketed, evals):
            gcurd = _dot(tf.cast(g_cur_m, dtype_o), d_o)
            cond1 = tf.logical_or(f_cur_m > f0_m + tf.cast(a_cur_o * c1g0d, f0_m.dtype), f_cur_m >= f_prev_m)
            cond2 = tf.abs(gcurd) <= neg_c2g0d
            cond3 = gcurd >= 0.0

            def case1():
                return (a_prev_o, f_prev_m, g_prev_m,
                        a_cur_o, f_cur_m, g_cur_m,
                        a_prev_o, f_prev_m, g_prev_m,
                        a_cur_o, f_cur_m, g_cur_m,
                        tf.constant(True), evals)

            def case2():
                return (a_cur_o, f_cur_m, g_cur_m,
                        a_cur_o, f_cur_m, g_cur_m,
                        a_cur_o, f_cur_m, g_cur_m,
                        a_cur_o, f_cur_m, g_cur_m,
                        tf.constant(True), evals)

            def case3():
                a_n_o = a_cur_o * 2.0
                x_n_m = x_m + tf.cast(a_n_o, x_m.dtype) * tf.cast(d_o, x_m.dtype)
                f_n_m, g_n_m = loss_and_grad(x_n_m, tf_true)
                return (a_cur_o, f_cur_m, g_cur_m,
                        a_n_o, f_n_m, g_n_m,
                        a_lo_o, f_lo_m, g_lo_m,
                        a_hi_o, f_hi_m, g_hi_m,
                        bracketed, evals + 1)

            def case4():
                return (a_prev_o, f_prev_m, g_prev_m,
                        a_cur_o, f_cur_m, g_cur_m,
                        a_cur_o, f_cur_m, g_cur_m,
                        a_prev_o, f_prev_m, g_prev_m,
                        tf.constant(True), evals)

            return tf.case([(cond1, case1), (cond2, case2), (cond3, case4)],
                           default=case3, exclusive=False)

        (a_prev_o, f_prev_m, g_prev_m,
         a_cur_o, f_cur_m, g_cur_m,
         a_lo_o, f_lo_m, g_lo_m,
         a_hi_o, f_hi_m, g_hi_m,
         bracketed, evals) = tf.while_loop(
            bcond, bbody,
            loop_vars=(a_prev_o, f_prev_m, g_prev_m,
                       a_cur_o, f_cur_m, g_cur_m,
                       a_lo_o, f_lo_m, g_lo_m,
                       a_hi_o, f_hi_m, g_hi_m,
                       bracketed, evals),
            maximum_iterations=max_evals, parallel_iterations=1
        )

        a_star_o = tf.constant(0.0, dtype_o)
        f_star_m = f0_m
        g_star_m = g_m
        done = tf.constant(False)

        # noinspection PyShadowingNames
        # pylint: disable=shadowed-name
        # noinspection SpellCheckingInspection
        def zcond(_a_lo_o, _f_lo_m, _g_lo_m,
                  _a_hi_o, _f_hi_m, _g_hi_m,
                  evals, done, _a_star_o, _f_star_m, _g_star_m):
            small = tf.abs(_a_hi_o - _a_lo_o) < tol_alpha_o
            return tf.logical_and(tf.logical_not(done),
                                  tf.logical_and(evals < max_evals, tf.logical_not(small)))

        # noinspection PyShadowingNames
        # pylint: disable=shadowed-name
        # noinspection SpellCheckingInspection
        def zbody(a_lo_o, f_lo_m, g_lo_m,
                  a_hi_o, f_hi_m, g_hi_m,
                  evals, done, a_star_o, f_star_m, g_star_m):
            a_o = 0.5 * (a_lo_o + a_hi_o)
            x_try_m = x_m + tf.cast(a_o, x_m.dtype) * tf.cast(d_o, x_m.dtype)
            f_try_m, g_try_m = loss_and_grad(x_try_m, tf.constant(True))
            gtryd = _dot(tf.cast(g_try_m, dtype_o), d_o)

            cond1 = tf.logical_or(f_try_m > f0_m + tf.cast(a_o * c1g0d, f0_m.dtype), f_try_m >= f_lo_m)
            cond2 = tf.abs(gtryd) <= neg_c2g0d

            def case1():
                return a_lo_o, f_lo_m, g_lo_m, a_o, f_try_m, g_try_m, evals + 1, done, a_star_o, f_star_m, g_star_m

            def case2():
                return a_o, f_try_m, g_try_m, a_hi_o, f_hi_m, g_hi_m, evals + 1, tf.constant(
                    True), a_o, f_try_m, g_try_m

            def case3():
                sign_change = gtryd * _dot(tf.cast(g_lo_m, dtype_o), d_o) < 0.0

                def flip():
                    return a_o, f_try_m, g_try_m, a_lo_o, f_lo_m, g_lo_m, evals + 1, done, a_star_o, f_star_m, g_star_m

                def keep():
                    return a_o, f_try_m, g_try_m, a_hi_o, f_hi_m, g_hi_m, evals + 1, done, a_star_o, f_star_m, g_star_m

                return tf.cond(sign_change, flip, keep)

            return tf.case([(cond1, case1), (cond2, case2)], default=case3, exclusive=False)

        (a_lo_o, f_lo_m, g_lo_m,
         a_hi_o, f_hi_m, g_hi_m,
         evals, done, a_star_o, f_star_m, g_star_m) = tf.while_loop(
            zcond, zbody,
            loop_vars=(a_lo_o, f_lo_m, g_lo_m,
                       a_hi_o, f_hi_m, g_hi_m,
                       evals, done, a_star_o, f_star_m, g_star_m),
            maximum_iterations=max_evals, parallel_iterations=1
        )

        return a_star_o, f_star_m, g_star_m, evals

    return tf.cond(g0d < 0.0, run_hz_search, fail_dir)
