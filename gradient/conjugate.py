__author__ = 'vlad'

from gradient.algorithm import Algorithm
import numpy as np
from math import sqrt
from scipy.optimize import minimize_scalar
from gradient.quadratic import Quadratic_Func


class Conjugate_Gradient_Quadratic(Algorithm):
    def __init__(self, init_point, function, eps_x=1e-7, eps_f=1e-7, eps_f1=1e07, max_iter=10000):
        super().__init__(init_point, function)
        self.eps_x = eps_x
        self.eps_f = eps_f
        self.eps_f1 = eps_f1
        self.max_iter = len(init_point) + 1
        self.h = [np.array(-function.diff(self.points[0]).getA1(), ndmin=1, dtype=np.float_)]
        print(self.h)

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
        frac1 = np.tensordot(self.function.b, h_T) + 2 * np.tensordot(self.function.A * x_T, h_T)
        frac2 = 2 * np.tensordot(self.function.A * h_T, h_T)
        return - frac1 / frac2

    def launch(self):
        if type(self.function) is not Quadratic_Func:
            raise Exception('Function does not match method requirements (quadratic required)')
        while True:
            if len(self.points) > self.max_iter:
                print('Iteration limit reached')
                break
            next = self.iteration()
            self.points.append(next)
            if (self.h[-1] == np.array([0., 0.])).all():
                print('Solution found: ', self.points[-1])
                break


class Conjugate_Gradient(Algorithm):
    def __init__(self, init_point, function, eps_x=1e-7, eps_f=1e-7, eps_f1=1e07, max_iter=10000):
        super().__init__(init_point, function)
        self.eps_x = eps_x
        self.eps_f = eps_f
        self.eps_f1 = eps_f1
        self.max_iter = max_iter
        self.h = [np.array(-function.diff(self.points[0]).getA1(), ndmin=1, dtype=np.float_)]
        self.counter = 0
        print(self.h)

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
        return minimize_scalar(f, method='Golden')['x']

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