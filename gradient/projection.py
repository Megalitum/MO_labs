__author__ = 'fedoramy'

from gradient.algorithm import Algorithm
import numpy as np
from math import sqrt

class ProjectionMethod(Algorithm):
    def __init__(self, init_point, function, constr, eps_x=1e-7, eps_f=1e-7, eps_f1=1e-7, max_iter=10000):
        super().__init__(init_point, function)
        '''
        ploshchyna zadaetsya (p, x) = B
        '''
        self.p = np.array(constr._diffs, dtype = float)
        self.B = -constr.val(np.zeros(len(constr._args)))
        self.eps_x = eps_x
        self.eps_f = eps_f
        self.eps_f1 = eps_f1
        self.max_iter = max_iter
        print(self.projection([-3, 0, -3]))

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
        curvalue = self.function.val(self.points[-1])  #current value
        h = self.function.diff(self.points[-1]).getA1()# our gradient flattend in array
        alpha = beta = 0.5
        next = self.projection(- alpha * h + self.points[-1]) # minus - antigradient
        while self.function.val(next.flatten()) > curvalue: #infinite loop danger
            alpha *= beta
            next = self.projection(- alpha * h + self.points[-1])
        print(alpha, next)
        return next.flatten() # return one array(not array of array)

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

    def projection(self, a):
        return np.array(a + (self.B - self.p*np.matrix(a).T)/(np.linalg.norm(self.p))**2 * self.p)