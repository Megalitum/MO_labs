__author__ = 'vlad'

from gradient.algorithm import Algorithm
import numpy as np
from math import sqrt


class DividingMethod(Algorithm):
    def __init__(self, init_point, function, eps_x, eps_f=1e-7):
        super().__init__(init_point, function)
        self.eps_x = eps_x
        self.eps_f = eps_f

    def _norm(self, x1, x2):
        return sqrt(sum(np.array(x1) - np.array(x2)))

    def iteration(self):
        curvalue = self.function.val(self.points[-1])
        h = self.function.diff(self.points[-1]).T
        alpha = beta = 0.5
        next = - alpha * h + self.points[-1]
        while self.function.val(next) >= curvalue:
            alpha *= beta
            next = - alpha * h + self.points[-1]
        return np.array(next, dtype=float).reshape(-1, )


    def launch(self):
        while True:
            next = self.iteration()
            if self._norm(self.points[-1], next) < self.eps_x: #wtf happens here?
                break
            self.points.append(next)



