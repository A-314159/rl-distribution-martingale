import tensorflow as tf
from utilities.tensorflow_config import tf_compile
from optimizers.helpers import _dot, _norm


# noinspection PyShadowingNames
# pylint: disable=shadowed-name
@tf_compile
def armijo_engine(loss_and_grad,
                  x_m, f_m, g_m, d_o,
                  f_hist_m, f_hist_size,
                  alpha0_o, c1_o,
                  max_evals,
                  # policy knobs
                  window_sz,    # tf.int32
                  use_cubic,    # tf.bool
                  backtrack_o,  # opt dtype
                  tolx_o,       # opt dtype
                  max_norm_d,   # opt dtype
                  use_wolfe,    # tf.bool
                  wolfe_c2_o):  # opt dtype
    """
    Returns: (alpha_o, f_new_m, g_new_m, evals, backs)
    - Non-finite f_try/g_try are treated as failed trials → keep backtracking.
    - Only return α=0 when α < alamin (Numerical Recipes).
    """
    dtype_m = x_m.dtype
    dtype_o = d_o.dtype
    tf_true = tf.constant(True)
    tf_false = tf.constant(False)

    gTd_o = _dot(tf.cast(g_m, dtype_o), d_o)

    def _fail_dir():
        return tf.constant(0.0, dtype_o), f_m, g_m, tf.constant(0, tf.int32), tf.constant(0, tf.int32)

    def _search():
        win = tf.minimum(window_sz, f_hist_size)
        f_ref_m = tf.cond(win > 0,
                          lambda: tf.reduce_max(f_hist_m[:win]),
                          lambda: f_m)
        rhs_scale_o = c1_o * gTd_o

        # ===== NR step cap + alamin =====
        norm_d = tf.norm(d_o)
        d_o_capped = tf.cond(
            tf.logical_and(tf.math.is_finite(max_norm_d), max_norm_d > 0.0),
            lambda: tf.where(norm_d > max_norm_d, d_o * (max_norm_d / norm_d), d_o),
            lambda: d_o
        )
        d_m_capped = tf.cast(d_o_capped, dtype_m)

        abs_x = tf.abs(x_m)
        safe_abs_x = tf.where(abs_x < 1.0, 1.0, abs_x)
        max_ratio = tf.reduce_max(tf.abs(d_m_capped) / safe_abs_x)
        # dtype-aware epsilon to avoid div by 0
        eps_small = tf.constant(1e-30, dtype_o) if dtype_o == tf.float64 else \
                    (tf.constant(1e-30, dtype_o) if dtype_o == tf.float32 else tf.constant(1e-12, dtype_o))
        alamin_o = tolx_o / tf.maximum(max_ratio, eps_small)
        # =================================

        alpha_o   = tf.identity(alpha0_o)
        alpha2_o  = tf.constant(0.0, dtype_o)
        f2_m      = f_m
        evals     = tf.constant(0, tf.int32)
        backs     = tf.constant(0, tf.int32)
        accepted  = tf.constant(False)

        def cond(alpha_o, alpha2_o, f2_m, evals, backs, accepted, f_acc_m):
            return tf.logical_and(evals < max_evals, tf.logical_not(accepted))

        def body(alpha_o, alpha2_o, f2_m, evals, backs, accepted, f_acc_m):
            x_try_m = x_m + tf.cast(alpha_o, dtype_m) * d_m_capped
            f_try_m, _ = loss_and_grad(x_try_m, tf_false)

            # -------- NEW: treat non-finite f_try as a hard backtrack --------
            f_try_finite = tf.reduce_all(tf.math.is_finite(f_try_m))
            rhs_m        = f_ref_m + tf.cast(alpha_o * rhs_scale_o, dtype_m)
            ok_armijo    = tf.logical_and(f_try_finite, (f_try_m <= rhs_m))
            # -----------------------------------------------------------------

            def _check_wolfe_then_accept():
                # If Wolfe is requested, compute gradient *only now* and guard non-finite
                def _wolfe():
                    f_g_m, g_try_m = loss_and_grad(x_try_m, tf_true)
                    g_try_finite = tf.reduce_all(tf.math.is_finite(g_try_m))
                    gtryd_o = _dot(tf.cast(g_try_m, dtype_o), d_o_capped)
                    curv_ok = tf.logical_and(g_try_finite, (tf.abs(gtryd_o) <= wolfe_c2_o * tf.abs(gTd_o)))
                    # use f_try_m (already computed) as f_eff; if gradient is non-finite → fail curvature
                    return curv_ok, f_try_m

                return tf.cond(use_wolfe, _wolfe, lambda: (True, f_try_m))

            curv_ok, f_eff_m = tf.cond(ok_armijo, _check_wolfe_then_accept, lambda: (False, f_try_m))
            ok_total = tf.logical_and(ok_armijo, curv_ok)

            # Next alpha (geometric or cubic), but if trial was non-finite → force geometric shrink
            def _geometric():
                return alpha_o * backtrack_o

            def _cubic():
                # NR-style cubic (guarded); if previous α not set, fallback to geometric
                rhs_alpha_m  = f_ref_m + tf.cast(alpha_o * rhs_scale_o, dtype_m)
                rhs_alpha2_m = f_ref_m + tf.cast(alpha2_o * rhs_scale_o, dtype_m)
                rhs1 = tf.cast(f_eff_m - rhs_alpha_m,  dtype_o)
                rhs2 = tf.cast(f2_m   - rhs_alpha2_m, dtype_o)

                tiny = tf.cast(1e-30, dtype_o)
                denom = tf.maximum(alpha_o - alpha2_o, tiny)
                a = (rhs1/(alpha_o*alpha_o) - rhs2/(alpha2_o*alpha2_o)) / denom
                b = (-alpha2_o*rhs1/(alpha_o*alpha_o) + alpha_o*rhs2/(alpha2_o*alpha2_o)) / denom

                def _quad():
                    return tf.where(tf.abs(b) > tiny, -gTd_o/(2.0*b), 0.5*alpha_o)

                def _cubic_inner():
                    disc = b*b - 3.0*a*gTd_o
                    def _disc_neg():
                        return 0.5*alpha_o
                    def _disc_pos():
                        sqrt_disc = tf.sqrt(tf.maximum(disc, tf.cast(0.0, dtype_o)))
                        num = tf.where(b <= 0.0, -b + sqrt_disc, -gTd_o)
                        den = tf.where(b <= 0.0, 3.0*a, b + sqrt_disc)
                        step = tf.where(tf.abs(den) > tiny, num/den, 0.5*alpha_o)
                        return step
                    return tf.cond(disc < 0.0, _disc_neg, _disc_pos)

                tmplam = tf.cond(tf.abs(a) <= tiny, _quad, _cubic_inner)
                tmplam = tf.clip_by_value(tmplam, 0.1*alpha_o, 0.5*alpha_o)
                return tmplam

            # If f_try was non-finite: treat as failure and force geometric
            alpha_next_failed = _geometric()

            use_cubic_now = tf.logical_and(use_cubic, alpha2_o > 0.0)
            alpha_next_o = tf.where(
                ok_total,
                alpha_o,
                tf.where(
                    tf.logical_not(f_try_finite),      # non-finite → force geometric
                    alpha_next_failed,
                    tf.cond(use_cubic_now, _cubic, _geometric)
                )
            )

            evals_next     = evals + 1
            backs_next     = tf.where(ok_total, backs, backs + 1)
            alpha2_next_o  = tf.where(ok_total, alpha2_o, alpha_o)
            f2_next_m      = tf.where(ok_total, f2_m, f_eff_m)
            f_acc_next_m   = tf.where(ok_total, f_eff_m, f_acc_m)

            # Stop if alpha below alamin → accept α=0 (no move)
            too_small      = alpha_next_o < alamin_o
            accepted_next  = tf.where(too_small, tf.constant(True), ok_total)
            alpha_final_o  = tf.where(too_small, tf.cast(0.0, dtype_o), alpha_next_o)

            return alpha_final_o, alpha2_next_o, f2_next_m, evals_next, backs_next, accepted_next, f_acc_next_m

        alpha_fin_o, _, _, evals_fin, backs_fin, accepted, _ = tf.while_loop(
            cond, body,
            loop_vars=(alpha_o, alpha2_o, f2_m, evals, backs, accepted, f_m),
            maximum_iterations=max_evals, parallel_iterations=1)

        def _on_accept_nonzero():
            x_new_m = x_m + tf.cast(alpha_fin_o, dtype_m) * d_m_capped
            f_new_m, g_new_m = loss_and_grad(x_new_m, tf_true)
            return alpha_fin_o, f_new_m, g_new_m, evals_fin, backs_fin

        def _on_accept_zero():
            return tf.cast(0.0, dtype_o), f_m, g_m, evals_fin, backs_fin

        return tf.cond(accepted,
                       lambda: tf.cond(alpha_fin_o > 0.0, _on_accept_nonzero, _on_accept_zero),
                       _on_accept_zero)

    return tf.cond(gTd_o < 0.0, _search, _fail_dir)
