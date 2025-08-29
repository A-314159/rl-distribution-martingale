from enum import Enum, IntEnum


class CvgFlags(IntEnum):
    # names chosen with fixed length
    g_descent__ = 0
    loss_tol___ = 1
    grad_tol___ = 2
    x_chg_tol__ = 3
    x_stationar = 4
    tiny_curv__ = 5
    pos_slope__ = 6
    suff_dec___ = 7
    src_maxiter = 8
    bug________ = 9
    full_newton = 10
    line_min___ = 11
    initialized = 12
    none_grad__ = 13
    x_is_none__ = 14


converged = [CvgFlags.x_stationar, CvgFlags.loss_tol___, CvgFlags.grad_tol___]
tolerance = [CvgFlags.loss_tol___, CvgFlags.grad_tol___]
