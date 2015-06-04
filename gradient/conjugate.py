__author__ = 'y ss i'

from gradient.algorithm import Algorithm
import numpy as np
from math import sqrt
from scipy.optimize import minimize_scalar
from gradient.quadratic import Quadratic_Func


class Conjugate_Gradient_Quadratic(Algorithm):
    def __init__(self, init_point, function, eps_x=1e-7, eps_f=1e-7, eps_f1=1e-7, max_iter=10000, eps_h = 1e-13):
        super().__init__(init_point, function)
        self.eps_x = eps_x
        self.eps_f = eps_f
        self.eps_f1 = eps_f1
        self.eps_h = eps_h
        self.max_iter = len(init_point) + 1
        self.h = [np.array(-function.diff(self.points[0]).getA1(), ndmin=1, dtype=np.float_)]

    def _norm(self, x1, x2):
        return sqrt(sum(map(lambda x: x**2, np.array(x1, ndmin=1) - np.array(x2, ndmin=1))))

    def _get_mask(self):
        return '#{number:<7} x: {point}\nf: {value}\ngrad: {grad}\n\n'

    def _generate_strings(self):
        mask = self._get_mask()
        for number in range(len(self.points)):
            point = self.points[number]
            yield mask.format(number=number, point=str(point), value=self.function.val(point),
                              grad=self.function.diff(point).ravel())

    def iteration(self):
        alpha = self._calc_alpha_k(self.points[-1], self.h[-1])
        next = self.points[-1] + alpha * self.h[-1]
        beta = self._calc_beta_prev(next, self.h[-1])
        h_next = - self.function.diff(next).getA1() + beta * self.h[-1]
        self.h.append(h_next)
        print(alpha, beta, next, h_next)
        return next

    def _calc_beta_prev(self, x_k, h_prev):
        h_T = np.matrix(h_prev).T
        vec = self.function.A * h_T
        return np.tensordot(self.function.diff(x_k), vec) /\
               np.tensordot(h_T, vec)

    def _calc_alpha_k(self, x_k, h_k):
        x_T = np.matrix(x_k).T
        h_T = np.matrix(h_k).T
        frac1 = np.tensordot(self.function.b, h_T) +  np.tensordot(self.function.A * x_T, h_T)
        frac2 = np.tensordot(self.function.A * h_T, h_T)
        return - frac1 / frac2

    def launch(self):
        if type(self.function) is not Quadratic_Func:
            raise Exception('Function does not match method requirements (quadratic required)')
        while True:
            if len(self.points) > self.max_iter:
                print('Iteration limit reached')
                break

            next = self.iteration()
            if (np.linalg.norm(self.h[-1]) < self.eps_h):
                print('Solution found: ', self.points[-1])
                print('Iteration count: ', len(self.points) - 1)
                break
            self.points.append(next)



class Conjugate_Gradient_Quadratic_Positive(Algorithm):
    def __init__(self, init_point, function, eps_x=1e-7, eps_f=1e-7, eps_f1=1e-6, max_iter=10000,
                 eps_h = 1e-13, eps_near=1e-7, verbose=True):
        super().__init__(init_point, function)
        self.eps_x = eps_x
        self.eps_f = eps_f
        self.eps_f1 = eps_f1
        self.eps_h = eps_h
        self.eps_near = eps_near
        self.max_iter = max_iter
        self.verbose = verbose
        self.subpoints = []
        self.h = []

    def _norm(self, x1, x2):
        return sqrt(sum(map(lambda x: x**2, np.array(x1, ndmin=1) - np.array(x2, ndmin=1))))

    def _get_mask(self):
        return '#{number:<7} x: {point}\nf: {value}\ngrad: {grad}\n\n'

    def _generate_strings(self):
        mask = self._get_mask()
        for number in range(len(self.points)):
            point = self.points[number]
            yield mask.format(number=number, point=str(point), value=self.function.val(point),
                              grad=self.function.diff(point).ravel())

    def iteration(self, func):
        alpha = self._calc_alpha_k(func, self.subpoints[-1], self.h[-1])
        self.h[-1][np.where(self._is_nearzero(self.h[-1]))] = 0
        neg_h, *_ = np.where(self.h[-1] < 0)
        alpha2 = np.min(np.append(-self.subpoints[-1][neg_h] / self.h[-1][neg_h],alpha))
        next = self.subpoints[-1] + alpha2 * self.h[-1]
        beta = self._calc_beta_prev(func, next, self.h[-1])
        h_next = - func.diff(next).getA1() + beta * self.h[-1]
        self.h.append(h_next)
        #print(alpha, beta, next, h_next)
        return next

    def _calc_beta_prev(self, func, x_k, h_prev):
        h_T = np.matrix(h_prev).T
        vec = func.A * h_T
        return np.tensordot(func.diff(x_k), vec) /\
               np.tensordot(h_T, vec)

    def _calc_alpha_k(self, func, x_k, h_k):
        x_T = np.matrix(x_k).T
        h_T = np.matrix(h_k).T
        frac1 = np.tensordot(func.b, h_T) + np.tensordot(func.A * x_T, h_T)
        frac2 = np.tensordot(func.A * h_T, h_T)
        return - frac1 / frac2

    def _is_nearzero(self, val):
        return abs(val) < self.eps_near

    def _launch_on_subset(self, subset):
        A = self.function.A
        b = self.function.b
        arg_x, arg_y = np.meshgrid(subset, subset)
        del self.subpoints, self.h
        # function replacing with actual border-specific one
        f_new = Quadratic_Func(matrA=A[arg_x, arg_y], vecB=np.matrix(b[subset]).T, constC=0)
        self.subpoints = [self.points[-1][subset]]
        self.h = [np.array(-f_new.diff(self.subpoints[0]).getA1(), ndmin=1, dtype=np.float_)]
        while True:
            if len(self.subpoints) > self.max_iter:
                print('Iteration limit reached')
                break
            next_item = self.iteration(f_new)
            self.subpoints.append(next_item)
            if (np.linalg.norm(self.h[-1]) < self.eps_h):
                break
            if (np.linalg.norm(self.subpoints[-1] - next_item) < self.eps_x):
                break
        return self.subpoints[-1]

    def launch(self):
        if type(self.function) is not Quadratic_Func:
            raise Exception('Function does not match method requirements (quadratic required)')
        while True:
            if len(self.points) > self.max_iter:
                if self.verbose:
                    print('Iteration limit reached')
                break
            zero_argset, *_ = np.where(self._is_nearzero(self.points[-1]))
            diff_array = self.function.diff(self.points[-1]).getA1()
            zero_diff, *_ = np.where(abs(diff_array) < self.eps_f1)
            if len(np.union1d(zero_argset,zero_diff)) == self.dimension:
                #first case
                be_zero_diff = np.logical_or(self._is_nearzero(diff_array), diff_array > 0)
                if be_zero_diff.all():
                    break  #solution found
                zero_arg = self._is_nearzero(self.points[-1])
                subset, *_ = np.where(np.logical_not(np.logical_and(be_zero_diff,
                                                               zero_arg)))
                iteration_res = self._launch_on_subset(subset)
            else:
                #second case
                subset, *_ = np.where(np.logical_not(self._is_nearzero(self.points[-1])))
                iteration_res = self._launch_on_subset(subset)
            next_point = np.zeros(self.dimension, dtype=np.float_)
            next_point[subset] = iteration_res
            if self.verbose:
                print("Next point: ", next_point, 'Diff: ', diff_array)
            self.points.append(next_point)
            if len(self.points) > self.max_iter:
                break
        if self.verbose:
            print('Possible solution: ', self.points[-1])
        return {'result':self.points[-1], 'iterations':len(self.points),
                'success':len(self.points) < self.max_iter}


class Conjugate_Gradient(Algorithm):
    def __init__(self, init_point, function, eps_x=1e-7, eps_f=1e-7, eps_f1=1e-7, max_iter=10000):
        super().__init__(init_point, function)
        self.eps_x = eps_x
        self.eps_f = eps_f
        self.eps_f1 = eps_f1
        self.max_iter = max_iter
        self.h = [np.array(-function.diff(self.points[0]).getA1(), ndmin=1, dtype=np.float_)]
        self.counter = 0

    def _norm(self, x1, x2):
        return sqrt(sum(map(lambda x: x**2, np.array(x1, ndmin=1) - np.array(x2, ndmin=1))))

    def _get_mask(self):
        return '#{number:<7} x: {point}\nf: {value}\ngrad: {grad}\n\n'

    def _generate_strings(self):
        mask = self._get_mask()
        for number in range(len(self.points)):
            point = self.points[number]
            yield mask.format(number=number, point=str(point), value=self.function.val(point),
                              grad=self.function.diff(point).ravel())

    def iteration(self):
        alpha = self._calc_alpha_k(self.points[-1], self.h[-1])
        next = self.points[-1] + alpha * self.h[-1]
        self.counter += 1
        if self.counter % self.dimension == 0:
            beta = 0
            counter = 0
        else:
            beta = self._calc_beta_prev(next, self.points[-1])
        h_next = - self.function.diff(next).getA1() + beta * self.h[-1]
        self.h.append(h_next)
        print(alpha, beta, next, h_next)
        return next

    def _calc_beta_prev(self, x_k, x_prev):
        return np.tensordot(self.function.diff(x_k), self.function.diff(x_k) - self.function.diff(x_prev)) /\
               np.tensordot(self.function.diff(x_prev), self.function.diff(x_prev))

    def _calc_alpha_k(self, x_k, h_k):
        f = lambda alpha : self.function.val(x_k + alpha * h_k)
        res = minimize_scalar(f, bracket=(0,10), method='brent')
        return res['x']

    def launch(self):
        while True:
            if len(self.points) > self.max_iter:
                print('Iteration limit reached')
                break
            next = self.iteration()
            if self._norm(self.points[-1], next) < self.eps_x and\
                    self._norm(self.function.val(self.points[-1]), self.function.val(next)) < self.eps_f and\
                    self._norm(self.function.diff(self.points[-1]), self.function.diff(next)) < self.eps_f1:
                break
            self.points.append(next)