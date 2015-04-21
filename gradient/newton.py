__author__ = 'vlad'

from gradient.algorithm import Algorithm
import numpy as np
from math import sqrt


class NewtonMethod(Algorithm):
    def __init__(self, init_point, function, eps_x=1e-7, eps_f=1e-7, eps_f1=1e07):
        super().__init__(init_point, function)
        self.eps_x = eps_x
        self.eps_f = eps_f
        self.eps_f1 = eps_f1

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
        lpoint = self.points[-1]
        h = -(self.function.gesse(lpoint).getI()*self.function.diff(lpoint)).getA1()
        next = h + self.points[-1]
        print('h=', h, 'next=', next)
        return next.flatten()


    def launch(self):
        while True:
            next = self.iteration()
            if self._norm(self.points[-1], next) < self.eps_x and\
                    self._norm(self.function.val(self.points[-1]), self.function.val(next)) < self.eps_f and\
                    self._norm(self.function.diff(self.points[-1]), self.function.diff(next)) < self.eps_f1:
                break
            self.points.append(next)



