__author__ = 'vlad'

from gradient.algorithm import Algorithm
import numpy as np
from gradient.quadratic import Quadratic_Func
from gradient.conjugate import Conjugate_Gradient_Quadratic_Positive


class Linearization(Algorithm):
    def __init__(self, init_point, function, *args, **kwargs):
        super().__init__(init_point, function)
        self.eps_x = kwargs.get('eps_x',1e-7)
        self.max_iter =kwargs.get('max_iter',10000)
        self.delta = kwargs.get('delta',1)
        self.N = kwargs.get('N',1)
        self.eps = kwargs.get('eps', 0.1)
        self.bounds = list(args) # assume functions passed as f_i(x), where border is interpreted as f_i(x)<=0
        self.counter = 0

    def _get_mask(self):
        return '#{number:<7} x: {point}\nf: {value}\ngrad: {grad}\n\n'

    def _generate_strings(self):
        mask = self._get_mask()
        for number in range(len(self.points)):
            point = self.points[number]
            yield mask.format(number=number, point=str(point), value=self.function.val(point),
                              grad=self.function.diff(point).ravel())

    def _F(self, point):
        _max = 0
        for func in self.bounds:
            _max = max(func.val(point), _max)
        return _max

    def _delta_set(self, point):
        F = self._F(point)
        values = np.array([func.val(point) for func in self.bounds])
        return values >= F - self.delta

    def _help_func(self, point):
        return self.function.val(point) + self.N*self._F(point)

    def iteration(self):
        cur_item = self.points[-1]
        d_set = self._delta_set(cur_item)
        size = d_set.sum()
        A = np.matrix(np.zeros((size, size)))
        b = np.matrix(np.zeros((1,size)))
        vals = []
        diffs = [self.function.diff(cur_item)]
        for i in range(len(self.bounds)):
            if d_set[i]:
                diffs.append(self.bounds[i].diff(cur_item))
                vals.append(self.bounds[i].val(cur_item))
        for i in range(size):
            for j in range(i, size):
                A[i, j] = np.tensordot(diffs[i + 1], diffs[j + 1])
        for i in range(size):
            for j in range(i):
                A[i, j] = A[j, i]
        for i in range(size):
            b[0, i] = np.tensordot(diffs[i + 1], diffs[0]) - vals[i]
        # c = 0.5 * np.tensordot(diffs[0], diffs[0])
        temp_func = Quadratic_Func(matrA=A,vecB=b,constC=0)
        method = Conjugate_Gradient_Quadratic_Positive(np.ones(size),temp_func, verbose=False, max_iter=1000)
        result = method.launch()
        if not result['success']:
            raise ValueError('Bad delta')
        u = result['result']
        if self.N <= np.sum(u):
            self.N = 2*np.sum(u)
        p = - diffs[0]
        for i in range(size):
            p -= u[i]*diffs[i + 1]
        alpha = 1
        template = self._help_func(cur_item)
        pnorm = np.linalg.norm(p)
        if pnorm < self.eps_x:
            return cur_item
        index = 0
        while (self._help_func(cur_item + alpha * p.getA1())) > (template - alpha * self.eps * pnorm**2):
            alpha /= 2
            index += 1
            if index > 1000:
                raise Exception('Not decreasing')
        next_item = cur_item + alpha * p.getA1()
        print('alpha = ', alpha, ' p = ', p.T)
        print('point = ', next_item, ' value = [', self.function.val(next_item),']')
        return next_item


    def launch(self):
        while True:
            if len(self.points) > self.max_iter:
                print('Iteration limit reached')
                break
            try:
                next_item = self.iteration()
            except ValueError:
                self.delta /= 2
                continue
            except Exception as e:
                print('method failed', e.args)
                break
            if np.linalg.norm(self.points[-1] - next_item) < self.eps_x:
                break
            self.points.append(next_item)