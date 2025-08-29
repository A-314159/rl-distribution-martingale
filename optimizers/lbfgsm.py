import math
import numpy as np
import tensorflow as tf
from scipy.optimize import minimize_scalar as min1
from utilities.tensorflow_config import tf_compile
from utilities.misc import set_attributes
from optimizers.convergence_flag import CvgFlags


def canonical_vector(dim):
    u = np.zeros(dim)
    u[0] = 1
    return tf.constant(u, dtype=tf.keras.backend.floatx())


def sign(x):
    return tf.constant(1.0 if x > 0 else -1.0, dtype=tf.keras.backend.floatx())


dtype = tf.keras.backend.floatx()


class BFGSM():
    def __init__(self, params=None,
                 x=None, model=None, f=None,
                 memory: int = 15,
                 tolerance_loss=1e-12, tolerance_x=1e-32, tolerance_gradient=1e-20,
                 line_search_c1=1e-4, line_search_c2=0.9,  # line_search_c1=1e-4, line_search_c2=0.9,
                 wolfe=False, mascia=False, maximum_line_iteration=200, max_iterations=100, force_gd=False):
        """
        Minimization of a loss function by Standard limited memory BFGS, with
        Because it is difficult to apply quasi-Newton optimization with stochastic gradient, the loss function
        must be computed with a very large number of samples and the set of samples must be kept constant.

        We use notations of the paper (except that we renamed direction p by d in the case of positive Hessian).

        :param model:
        :param f: list of loss functions: the total loss function will be the weighted sum of their losses
        :param loss_weights: list of weights applicable to the losses
        :param tolerance_loss: tolerance on the loss value
        :param tolerance_x: tolerance on the norm of the change of solution
        :param tolerance_gradient: tolerance on the norm of the gradient
        :param line_search_c1: Armijo's rule coefficient for the line search algorithm
        :param line_search_c2: Wolfe condition for the line search algorithm
        :param maximum_line_iteration: maximum iteration for the line search algorithm
        """
        super(BFGSM, self).__init__()

        # Set values passed as arguments or default values
        self.x, self.model, self.f, self.memory = x, model, f, memory
        self.tolerance_loss, self.tolerance_gradient = tolerance_loss, tolerance_gradient  # tolerance on loss and grad
        self.tolerance_x = tolerance_x  # tolerance on the change of the solution
        self.line_search_c1, self.line_search_c2 = line_search_c1, line_search_c2
        self.maximum_line_iteration, self.max_iterations = maximum_line_iteration, max_iterations
        self.mascia, self.wolfe, self.force_gd = mascia, wolfe, force_gd

        # Override attribute values if in the parameter dictionary
        if params is not None: set_attributes(self, params)

        # The list of tensors of trainable variables is flattened in a 1D tensor
        # and so is the gradient and all other vectors.

        if mascia: self.wolfe = False

        # todo: change the code to allow for longer memory, in the meantime, we cap it
        self.memory = min(int(tf.size(self.x).numpy()), self.memory)

        self.m = 0
        self.m_offset = 0
        self.dim = tf.size(util.tf_flatten(model.nn.trainable_variables)).numpy()

        self.loss = 0  # loss at x

        self.g = None  # gradient of loss at x
        self.g_norm = None

        self.d = None  # search direction
        self.d_norm = None

        self.prior_x = None
        self.prior_g = None
        self.prior_d = None
        self.prior_d_norm = None
        self.angle = 0
        self.positive_curvature = True
        self.tiny_curvature = False
        self.__status = util.CvgFlags.initialized

        self.sTs = None
        self.yTy = None
        self.sTy = None
        self.sTg = None
        self.yTg = None

        self.s = []  # change of x between two iterations
        self.y = []  # change of the gradient between two iterations
        for m in range(self.memory):
            self.s.append(tf.zeros(self.dim, dtype=util.dtype))
            self.y.append(tf.zeros(self.dim, dtype=util.dtype))
        self.alpha = np.zeros(self.dim)
        self.rho = np.zeros(self.dim)
        self.canonical_vector = canonical_vector(self.dim)  # vector (1,0,..., 0)
        self.neg_curvature = False
        self.x = None  # vector x
        self.g, self.loss = self.derivatives(x=x, derivative_order=1)
        self.g_descent(x, learning_rate=1e-12)  # Initialize with a simple gradient descent

    @property
    def sqrt_loss(self):
        return math.sqrt(self.loss.numpy())

    @property
    def header(self):
        return 'opt: status #,status,pos. curv.,small curv.,angle,grad. norm,max. x'

    @property
    def msg(self):
        return '%.4e, %s, %s, %d, %d, φ=%.1f π, |∇|=%.1e, max|ω|=%.1e' % (self.loss,
                                                                          self.__status.name,
                                                                          "NEG" if self.neg_curvature else "---",
                                                                          self.positive_curvature, self.tiny_curvature,
                                                                          self.angle / math.pi, self.g_norm,
                                                                          tf.reduce_max(self.x).numpy())

    @property
    def short_msg(self):
        return '%d,%s,%d,%d,%.1f,%.1e,%.1e' % (self.__status.value, self.__status.name,
                                               self.positive_curvature, self.tiny_curvature,
                                               self.angle / math.pi, self.g_norm, tf.reduce_max(self.x).numpy())

    @property
    def status(self):
        return self.__status

    @property
    def converged(self):
        return self.__status in util.converged

    def derivatives(self, x, derivative_order: int = 0):
        l, g = self.f(x)
        if derivative_order == 0:
            return l
        else:
            if tf.math.count_nonzero(tf.math.is_nan(g)) != 0:
                raise Exception('gradient is nan in BFGS derivatives')
            return g, l

    def reset(self, memory=None):
        self.x = None
        self.m = 0
        self.m_offset = 0
        if memory is not None:
            self.s = []  # change of x between two iterations
            self.y = []  # change of the gradient between two iterations
            self.memory = memory
            for m in range(memory):
                self.s.append(tf.zeros(self.dim, dtype=util.dtype))
                self.y.append(tf.zeros(self.dim, dtype=util.dtype))

    def update(self, x, l, g, d):
        if g is None:
            g, l = self.derivatives(x=x, derivative_order=1)

        self.prior_x = self.x
        self.prior_g = self.g
        self.prior_d = self.d
        self.prior_d_norm = self.d_norm

        self.x = x
        self.loss = l
        self.g = g
        self.d = d

        self.g_norm = tf.norm(g)
        self.d_norm = tf.norm(d)
        cosine = (tf.tensordot(self.d, self.prior_d, 1) / self.d_norm / self.prior_d_norm).numpy()

        if cosine > 1:
            self.angle = 0
        elif cosine < -1:
            self.angle = -3.15
        else:
            self.angle = math.acos(cosine)
        self.set_s_y()

    def g_descent(self, x, learning_rate=1e-6, status=util.CvgFlags.g_descent__):
        self.x = x
        self.d = -self.g
        self.d_norm = self.d_norm = tf.norm(self.d)
        if self.d is None:
            g, loss = self.derivatives(x, derivative_order=1)  # added on july 26th 2025 because sometimes is g None!
            if g is None:
                print('Gradient is None')
                g = tf.zeros_like(x)
                status = util.CvgFlags.none_grad__
            self.d = -g
        if x is None:
            print('x is None in gd')
            return util.CvgFlags.x_is_none__
        _x = self.x + learning_rate * self.d
        _g, _loss = self.derivatives(_x, derivative_order=1)
        self.update(_x, _loss, _g, self.d)
        self.m = self.m_offset = 0
        self.__status = status
        return status

    def index(self):
        return (self.m + self.m_offset) % self.memory

    def s_y_dot(self):
        index = self.index()
        s = self.s[index]
        y = self.y[index]
        self.sTs = tf.tensordot(s, s, 1)
        self.yTy = tf.tensordot(y, y, 1)
        self.sTy = tf.tensordot(s, y, 1)
        return

    def set_s_y(self):
        index = self.index()
        self.s[index] = self.x - self.prior_x
        self.y[index] = self.g - self.prior_g
        self.s_y_dot()

    def copy_prior_s_y(self):
        if self.m == 0:
            raise Exception("Tiny curvature for m=0. The code does not handle this case.")
        index = self.index()
        prior_index = (self.m + self.m_offset - 1) % self.memory
        self.s[index] = self.s[prior_index]
        self.y[index] = self.y[prior_index]
        self.s_y_dot()

    def curvature(self):

        sTy_floor = math.sqrt(1e-16 * self.sTs * self.yTy)
        self.positive_curvature = curvature_condition = (self.sTy > sTy_floor).numpy()
        self.tiny_curvature = tiny_sTy = math.fabs(self.sTy) < sTy_floor
        self.neg_curvature = False
        if self.mascia:
            if self.sTy < 0:  # Negative curvature: replace y by -y
                self.y[self.index()] = -self.y[self.index()]
                self.sTy = -self.sTy
                self.neg_curvature = True
            if tiny_sTy:  # Tiny curvature: increase y
                multiplier = sTy_floor / self.sTy
                self.y[self.index()] = self.y[self.index()] * multiplier
                self.sTy = sTy_floor
                self.yTy = self.yTy * multiplier * multiplier
            curvature_condition = True
            tiny_sTy = False

        if (not curvature_condition) and self.wolfe:
            raise Exception(
                "Curvature condition not satisfied despite applying Wolfe condition. Need to apply damping or copying")
        return curvature_condition, tiny_sTy

    def gd(self):
        # warning very weird: we need to use a very large lr, otherwise extremely slow. With 10 it is fast but errors goes up sometimes,
        #  with 100, it converges to a null distribution immediatley with 0 gradient
        self.g_descent(self.x, learning_rate=1e-3, status=util.CvgFlags.g_descent__)
        return util.CvgFlags.g_descent__

    def __call__(self):
        """
        update the vector x
        :return:    convergence status
        """
        if self.force_gd:
            return self.gd()

        if self.loss < self.tolerance_loss:
            self.__status = util.CvgFlags.loss_tol___
            return util.CvgFlags.loss_tol___

        if math.sqrt(self.sTs) < self.tolerance_x:
            self.__status = util.CvgFlags.x_chg_tol__
            self.line_search_c1 = 1e-6
            self.reset(memory=10)
            self._status = self.g_descent(self.x, learning_rate=1e-5, status=util.CvgFlags.x_chg_tol__)
            return self._status

        if self.g_norm < self.tolerance_gradient:
            self.__status = util.CvgFlags.grad_tol___
            return util.CvgFlags.grad_tol___

        positive_curvature, tiny_curvature = self.curvature()
        if tiny_curvature: self.copy_prior_s_y()  # Set a and y as prior s and y (not recommended by Nocedal)

        """
        Note:
        1)  If we apply Wolfe condition, the curvature condition should be satisfied. Therefore, we should be 
            able to always apply lbfgs, except when the curvature is tiny (i.e. the loss function is close to linear).
        2)  When the curvature is tiny, we could apply damping rather than reusing prior s and y.
        3)  Under Mascia method, if we dont apply Wolfe condition and if the curvature is negative, 
            we replace y by -y. 
            The lbfgs update will be positive. The matrix is not a proxi of the Hessian anymore, but it should 
            set the search direction as a mix a positive curvature direction and the opposite of negative curvature.         
        """

        if positive_curvature:  # Curvature condition is satisfied, proceed with L-BFGS
            _d, _p, _curvilinear, _min_eigenvalue = self.lbfgs_direction(), None, False, 0
        else:  # Proceed with memory-less curvilinear search
            _d, _p, _curvilinear, _min_eigenvalue = self.curvilinear_direction()

        if self.loss < 1e-18 or self.neg_curvature:  # if loss is small enough, we perform full optimization (This is slow!)
            # warning: july 2025: test to minimize if neg curvature
            m = min1(lambda l: self.derivatives(self.x + l * _d))
            _loss = m.fun
            _x = self.x + m.x * _d
            # _g = None # july 2025
            _g, _loss = self.derivatives(x=_x, derivative_order=1)  # july 2025
            self.__status = util.CvgFlags.line_min___
        else:
            n = tf.norm(_d)

            _g, _loss, _x, self.__status = self.line_search(d=_d, p=_p, curvilinear=_curvilinear,
                                                            eigenvalue=_min_eigenvalue)

        if self.__status == util.CvgFlags.pos_slope__ or self.__status == util.CvgFlags.x_stationar:
            # This should not happen. We perform a simple gradient descent as a poor idea
            # We raise an exception anyhow to force debugging and find better idea
            #raise Exception('Positive slope in bfgsm')
            if self.__status == util.CvgFlags.pos_slope__: print('Positive slope')
            #self.line_search_c1 = 1e-6
            x = self.x
            self.reset(memory=10)
            self.g, self.loss = self.derivatives(x=x, derivative_order=1)
            self.__status = self.g_descent(x, learning_rate=1e-5, status=self.__status)
        self.update(_x, _loss, _g, _d)

    def curvilinear_direction(self):
        """
        Implementation of "A memoryless BFGS neural network training algorithm"
        by Apostolopoulou, Sotiropoulos, Livieris, Pintelas.
        """
        sTg = tf.tensordot(self.s, self.g, 1)
        yTg = tf.tensordot(self.y, self.g, 1)
        theta = self.sTs / self.sTy

        # compute the minimum eigenvalue lambda1 of the approximate Hessian
        a = 1 + theta * self.yTy / self.sTy  # Note: a>=2 always
        if a > 2:
            discriminant = math.sqrt(a * a - 4)
            lambda1 = min(1 / theta, (a - discriminant) / 2 / theta, (a + discriminant) / 2 / theta)
        else:
            lambda1 = a

        if lambda1 > 0:  # The approximate Hessian is positive
            raise Exception("This should not happen as the curvature condition is satisfied")
        else:  # The approximate Hessian is not positive
            # set direction p
            p = -self.g
            # compute direction d from the eigenvector u1 of minimum eigenvalue
            if a == 2:
                d = -self.canonical_vector if self.g[0] > 0 else self.canonical_vector
            else:
                # compute the eigenvector u1 of the minimum eigenvalue lambda1
                lambda_hat = -lambda1 * (1 - 1e-5)
                gamma = (1 / theta + lambda_hat) * (
                        lambda_hat * lambda_hat + a * lambda_hat / theta + 1 / theta / theta)
                gamma0 = lambda_hat * lambda_hat + (a + 1) * (lambda_hat + 1 / theta) / theta
                gamma1 = lambda_hat + (a + 1) / theta
                gamma_theta2 = gamma * theta * theta
                one_gamma1_theta = 1 - gamma1 * theta
                # we start with u0 = g. we could start with any vector, in particular (1,...1)
                sTu = sTg
                yTu = yTg
                coef_g = (one_gamma1_theta + gamma0 * theta * theta) / gamma_theta2
                coef_s = (one_gamma1_theta * sTu + theta * yTu) / gamma_theta2 / self.sTs
                coef_y = ((one_gamma1_theta + a) * theta * yTu - sTu) / gamma_theta2 / self.sTy
                u1 = -coef_g * self.g + coef_s * self.s - coef_y * self.y
                d = -sign(tf.tensordot(u1, self.g, 1)) * u1 / tf.norm(u1)
            self.m_offset = self.m_offset + self.m
            self.m = 1
            return d, p, True, lambda1

    def lbfgs_direction(self):
        """
        Classical limited memory BFGS
        """
        index = self.index()
        self.rho[index] = 1.0 / self.sTy
        gamma = self.sTy / self.yTy
        q = self.g
        _M = self.memory if self.m + 1 >= self.memory else self.m + 1
        for i in range(_M):
            self.alpha[index] = self.rho[index] * tf.tensordot(self.s[index], q, 1)
            q = q - self.alpha[index] * self.y[index]
            index = (index - 1) % self.memory
        index = (self.m + self.m_offset - _M + 1) % self.memory
        r = gamma * q
        for i in range(_M):
            beta = self.rho[index] * tf.tensordot(self.y[index], r, 1)
            r = r + (self.alpha[index] - beta) * self.s[index]
            index = (index + 1) % self.memory
        d = -r
        self.m = self.m + 1
        return d

    def line_search(self, d, p=None, step_max=1, curvilinear=False, eigenvalue=0):
        """
        Implementation and adaptation of the search of Numerical Recipes in C, chapter 9.7.
        Using the same notations of Numerical Recipes. (alam means lambda)
        The same algorithm is described in section 3.5 of Numerical Optimization, 2nd edition, by Nocedal and Wright

        Search lambda such that
            new_x = x + lambda d                    if linear
            new_x = x + lambda d + lambda**2 p      if curvilinear

        :param d: linear direction
        :param p: quadratic direction
        :param step_max: cap on the norm of direction d (equivalent to cap the max step in the direction d)
        :param curvilinear: boolean
        :param eigenvalue: for curvilinear search only
        :return: new x and convergence status
        """
        if not curvilinear:
            # Shrink d if its norm is larger than step_max (only for linear case)
            norm_d = tf.norm(d)
            d = d * step_max / norm_d if norm_d > step_max else d
            eigenvalue = 0

        slope = tf.tensordot(self.g, d, 1)  # slope at lambda=0 (the same whether linear or curvilinear)
        if slope >= 0:
            return None, self.loss, self.x, util.CvgFlags.pos_slope__

        abs_x = tf.math.abs(self.x)
        max_ratio = tf.reduce_max(tf.math.abs(d) / tf.where(abs_x < 1, 1, abs_x))
        alamin = self.tolerance_x / max_ratio

        alam = 1
        alam2 = 0
        f0 = self.loss
        f2 = 0
        for i in range(self.maximum_line_iteration):
            if curvilinear:
                new_x = self.x + alam * d + alam ** 2 * p
            else:
                new_x = self.x + alam * d
            if tf.math.count_nonzero(tf.math.is_nan(new_x)) != 0:
                error_on_d = tf.math.count_nonzero(tf.math.is_nan(d)) != 0
                raise Exception(
                    'new x is nan in line_search with alam=%f and error on direction: %s' % (alam, error_on_d))
            f = self.derivatives(x=new_x, derivative_order=0)

            g = None
            if alam < alamin:
                if f > f0:
                    return g, f, new_x, util.CvgFlags.pos_slope__
                else:
                    # print(max_ratio)
                    return g, f, new_x, util.CvgFlags.x_stationar
            if not math.isnan(f):
                if f < f0 + self.line_search_c1 * alam * (slope + 0.5 * eigenvalue):  # Armijo condition
                    stop = True
                    if self.wolfe:  # Wolfe condition
                        g, f = self.derivatives(x=new_x, derivative_order=1)
                        new_slope = tf.tensordot(g, d, 1)
                        if new_slope < self.line_search_c2 * slope:
                            stop = False
                    if stop:
                        if alam == 1:
                            return g, f, new_x, util.CvgFlags.full_newton
                        else:
                            return g, f, new_x, util.CvgFlags.suff_dec___
                if i == self.maximum_line_iteration - 1:
                    return g, f, new_x, util.CvgFlags.src_maxiter
                if alam == 1:
                    tmplam = -slope / 2 / (f - f0 - slope)
                else:
                    rhs1 = f - f0 - alam * slope
                    rhs2 = f2 - f0 - alam2 * slope
                    a = (rhs1 / (alam * alam) - rhs2 / (alam2 * alam2)) / (alam - alam2)
                    b = (-alam2 * rhs1 / (alam * alam) + alam * rhs2 / (alam2 * alam2)) / (alam - alam2)
                    if a == 0:
                        tmplam = -slope / 2 / b
                    else:
                        disc = b * b - 3.0 * a * slope
                        if disc < 0.0:
                            tmplam = 0.5 * alam
                        elif b <= 0.0:
                            tmplam = (-b + math.sqrt(disc)) / (3.0 * a)
                        else:
                            tmplam = -slope / (b + math.sqrt(disc))
                    if math.isnan(tmplam) or tmplam > 0.5 * alam:
                        tmplam = 0.5 * alam
                alam2 = alam
                f2 = f
                alam = max(tmplam, 0.1 * alam)
                if math.isnan(alam):
                    raise Exception('bug (this should not happen): alam is nan in line_search')
            else:
                alam = 0.5 * alam
        raise Exception('bug in line search. The execution should never reach this line.')
