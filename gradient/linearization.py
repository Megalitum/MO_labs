__author__ = 'vlad'

from gradient.algorithm import Algorithm
import numpy as np


class Linearization(Algorithm):
    def __init__(self, init_point, function, eps_x=1e-7, eps_f=1e-7, eps_f1=1e-7, max_iter=10000, *args):
        super().__init__(init_point, function)
        self.eps_x = eps_x
        self.eps_f = eps_f
        self.eps_f1 = eps_f1
        self.max_iter = max_iter
        self.bounds = list(args) # assume functions passed as f_i(x), where border is interpreted as f_i(x)<=0
        self.h = [np.array(-function.diff(self.points[0]).getA1(), ndmin=1, dtype=np.float_)]
        self.counter = 0


    def _get_mask(self):
        return '#{number:<7} x: {point}\nf: {value}\ngrad: {grad}\n\n'

    def _generate_strings(self):
        mask = self._get_mask()
        for number in range(len(self.points)):
            point = self.points[number]
            yield mask.format(number=number, point=str(point), value=self.function.val(point),
                              grad=self.function.diff(point).ravel())

    def iteration(self):
        next_item = 0
        # store parameters N, delta
        # check Pshenichny on page
        # need to add function building and solve it with
        # phi(u) = 0.5 <Au,u> + <b,u> + c
        # A = [<f_i'(x), f_j'(x)>] i,j in I_delta
        return next_item


    def launch(self):
        while True:
            if len(self.points) > self.max_iter:
                print('Iteration limit reached')
                break
            next = self.iteration()
            if self._norm(self.points[-1], next) < self.eps_x and\
                    np.linalg.norm(self.function.val(self.points[-1]), self.function.val(next)) < self.eps_f and\
                    np.linalg.norm(self.function.diff(self.points[-1]), self.function.diff(next)) < self.eps_f1:
                break # possibly bad stop criterion
            self.points.append(next)