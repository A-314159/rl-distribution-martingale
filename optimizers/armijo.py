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
                  # policy knobs (all opt dtype / ints / bools)
                  window_sz,  # tf.int32, e.g. 1 for monotone, >1 for non-monotone
                  use_cubic,  # tf.bool (False→geometric backtracking)
                  backtrack_o,  # β in (0,1), used when not cubic
                  alamin_o,  # minimum alpha; stop if below
                  step_max_o,  # 0 or negative → no cap, else cap ||d||
                  use_wolfe,  # tf.bool
                  wolfe_c2_o):  # only used if use_wolfe=True
    """
    Returns: (alpha_o, f_new_m, g_new_m, evals, backs)
    - If Armijo never accepts: returns alpha=0 and (f_m, g_m).
    """

    dtype_m = x_m.dtype
    dtype_o = d_o.dtype
    tf_true = tf.constant(True)
    tf_false = tf.constant(False)

    gTd_o = _dot(tf.cast(g_m, dtype_o), d_o)

    def _fail_dir():
        return tf.constant(0.0, dtype_o), f_m, g_m, tf.constant(0, tf.int32), tf.constant(0, tf.int32)

    def _search():
        # Non/Monotone reference
        win = tf.minimum(window_sz, f_hist_size)
        f_ref_m = tf.cond(win > 0,
                          lambda: tf.reduce_max(f_hist_m[:win]),
                          lambda: f_m)
        rhs_scale_o = c1_o * gTd_o

        # Optional step cap on ||d|| (model norm)
        d_m = tf.cast(d_o, dtype_m)
        if_cap = tf.logical_and(tf.math.is_finite(step_max_o), step_max_o > 0)

        def _cap_dir():
            n = _norm(d_m)
            scale = step_max_o / tf.maximum(n, tf.cast(1.0, dtype_o))
            scale = tf.minimum(scale, tf.cast(1.0, dtype_o))
            return tf.cast(scale, dtype_m) * d_m, tf.cast(scale, dtype_o)

        def _no_cap():
            return d_m, tf.cast(1.0, dtype_o)

        d_m_capped, cap_scale_o = tf.cond(if_cap, _cap_dir, _no_cap)

        # State vars for loop
        alpha_o = tf.identity(alpha0_o)
        alpha2_o = tf.constant(0.0, dtype_o)  # previous alpha (for cubic)
        f2_m = f_m  # f at alpha2
        evals = tf.constant(0, tf.int32)
        backs = tf.constant(0, tf.int32)
        accepted = tf.constant(False)

        # Armijo loop
        def cond(alpha_o, alpha2_o, f2_m, evals, backs, accepted, f_acc_m):
            return tf.logical_and(evals < max_evals, tf.logical_not(accepted))

        def body(alpha_o, alpha2_o, f2_m, evals, backs, accepted, f_acc_m):
            x_try_m = x_m + tf.cast(alpha_o, dtype_m) * d_m_capped
            f_try_m, _ = loss_and_grad(x_try_m, tf_false)

            rhs_m = f_ref_m + tf.cast(alpha_o * rhs_scale_o, dtype_m)
            ok_armijo = f_try_m <= rhs_m

            # If Armijo is ok but Wolfe is requested, check curvature lightly
            def _check_wolfe_then_accept():
                if_not_wolfe = (lambda: (True, f_try_m))  # accept

                def _wolfe():
                    # Only now compute gradient to check curvature
                    _, g_try_m = loss_and_grad(x_try_m, tf_true)
                    gtryd_o = _dot(tf.cast(g_try_m, dtype_o), d_o)
                    curv_ok = tf.abs(gtryd_o) <= wolfe_c2_o * tf.abs(gTd_o)
                    return curv_ok, f_try_m

                curv_ok, f_eff_m = tf.cond(use_wolfe, _wolfe, if_not_wolfe)
                return curv_ok, f_eff_m

            curv_ok, f_eff_m = tf.cond(ok_armijo, _check_wolfe_then_accept, lambda: (False, f_try_m))
            ok_total = tf.logical_and(ok_armijo, curv_ok)

            # Next alpha (geometric or cubic)
            def _geometric():
                return alpha_o * backtrack_o

            def _cubic():
                # Numerical-Recipes-ish safeguarded step using (alpha2,f2) and (alpha,f)
                # rhs = f0 + c1*alpha*gTd (in model dtype); build residuals in model dtype
                rhs_alpha_m = f_ref_m + tf.cast(alpha_o * rhs_scale_o, dtype_m)
                rhs_alpha2_m = f_ref_m + tf.cast(alpha2_o * rhs_scale_o, dtype_m)

                rhs1_m = f_eff_m - rhs_alpha_m  # f(α) - model
                rhs2_m = f2_m - rhs_alpha2_m  # f(α2) - model

                # Switch to opt dtype for algebra
                rhs1 = tf.cast(rhs1_m, dtype_o)
                rhs2 = tf.cast(rhs2_m, dtype_o)

                # Quadratic / cubic fallback logic (NR §9.7)
                # a*α^2 + b*α + c fits residual; derive safe tmplam
                # Use the standard guarded formulas:
                a = (rhs1 / (alpha_o * alpha_o) - rhs2 / (alpha2_o * alpha2_o)) / tf.maximum(alpha_o - alpha2_o,
                                                                                             tf.cast(1e-12, dtype_o))
                b = (-alpha2_o * rhs1 / (alpha_o * alpha_o) + alpha_o * rhs2 / (alpha2_o * alpha2_o)) / tf.maximum(
                    alpha_o - alpha2_o, tf.cast(1e-12, dtype_o))

                # If a is tiny → quadratic step; else cubic with discriminant
                def _quad():
                    # tmplam = -slope / (2b) in NR; here slope=gTd, already included in rhs
                    # Use conservative half step if b ~ 0
                    return tf.where(tf.abs(b) > tf.cast(1e-20, dtype_o),
                                    - (gTd_o) / (2.0 * b),
                                    0.5 * alpha_o)

                def _cubic_inner():
                    disc = b * b - 3.0 * a * gTd_o

                    def _disc_neg():
                        return 0.5 * alpha_o

                    def _disc_pos():
                        sqrt_disc = tf.sqrt(tf.maximum(disc, tf.cast(0.0, dtype_o)))
                        num = tf.where(b <= 0.0, -b + sqrt_disc, -gTd_o)
                        den = tf.where(b <= 0.0, 3.0 * a, b + sqrt_disc)
                        # Protect division
                        step = tf.where(tf.abs(den) > tf.cast(1e-30, dtype_o),
                                        num / den,
                                        0.5 * alpha_o)
                        return step

                    return tf.cond(disc < 0.0, _disc_neg, _disc_pos)

                tmplam = tf.cond(tf.abs(a) <= tf.cast(1e-30, dtype_o), _quad, _cubic_inner)
                # Safeguards: keep within (0.1 α, 0.5 α)
                lo = 0.1 * alpha_o
                hi = 0.5 * alpha_o
                tmplam = tf.clip_by_value(tmplam, lo, hi)
                return tmplam

            alpha_next_o = tf.cond(ok_total,
                                   lambda: alpha_o,
                                   lambda: tf.cond(use_cubic & (alpha2_o > 0.0), _cubic, _geometric))

            evals_next = evals + 1
            backs_next = tf.where(ok_total, backs, backs + 1)
            # Update the "previous point" only when backtracking
            alpha2_next_o = tf.where(ok_total, alpha2_o, alpha_o)
            f2_next_m = tf.where(ok_total, f2_m, f_eff_m)
            f_acc_next_m = tf.where(ok_total, f_eff_m, f_acc_m)

            # If curvature fails but Armijo ok and Wolfe on → do a light shrink and continue
            def _light_shrink_when_curv_fails():
                # shrink a bit (geometric), but keep accepted=False to loop
                return alpha_o * tf.where(use_cubic, tf.cast(0.7, dtype_o), backtrack_o)

            alpha_next_o = tf.where(ok_armijo & tf.logical_not(curv_ok) & use_wolfe,
                                    _light_shrink_when_curv_fails(),
                                    alpha_next_o)

            # Stop if alpha below alamin
            too_small = alpha_next_o < alamin_o
            accepted_next = tf.where(too_small, tf.constant(True), ok_total)  # accept “no move” if too small
            alpha_final_o = tf.where(too_small, tf.cast(0.0, dtype_o), alpha_next_o)

            return alpha_final_o, alpha2_next_o, f2_next_m, evals_next, backs_next, accepted_next, f_acc_next_m

        alpha_fin_o, alpha2_o, f_acc_m, evals_fin, backs_fin, accepted, f_acc_m = tf.while_loop(
            cond, body,
            loop_vars=(alpha_o, alpha2_o, f2_m, evals, backs, accepted, f_m),
            maximum_iterations=max_evals, parallel_iterations=1
        )

        # If accepted with α>0 → compute g once (needed by caller anyway)
        def _on_accept_nonzero():
            x_new_m = x_m + tf.cast(alpha_fin_o, dtype_m) * d_m_capped
            f_new_m, g_new_m = loss_and_grad(x_new_m, tf_true)
            return alpha_fin_o, f_new_m, g_new_m, evals_fin, backs_fin

        # If accepted with α==0 → no move
        def _on_accept_zero():
            return tf.cast(0.0, dtype_o), f_m, g_m, evals_fin, backs_fin

        # If somehow not accepted (shouldn’t happen) → treat as α==0
        return tf.cond(accepted,
                       lambda: tf.cond(alpha_fin_o > 0.0, _on_accept_nonzero, _on_accept_zero),
                       _on_accept_zero)

    return tf.cond(gTd_o < 0.0, _search, _fail_dir)
