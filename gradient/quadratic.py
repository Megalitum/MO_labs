#!/usr/bin/python
# coding=UTF-8

__author__ = 'fedoramy'

import numpy as np
from gradient.function_input import FunctionInput
import re


class Quadratic_Func(object):
    '''
    f(x)=1/2(Ax,x) + (b,x) + c
    '''

    def __init__(self, *args, **kwargs):
        if len(kwargs) == 0:
            if len(args[1]) == 0:
                raise Exception('No arguments passed')
            f = FunctionInput()
            self._args = args[1]
            self.A = f.create_matrix_A(args[0], args[1]) * 2
            self.b = f.create_b(args[0], args[1])
            const = re.search(r'(?:[+-]|^)\d+(?:\.\d+)?(?:$|[+-])', args[0])
            if const is not None:
                self.c = float(const.group())
            else:
                self.c = 0
        else:
            self.A = np.matrix(kwargs['matrA'])
            self.b = np.matrix(kwargs['vecB']).T
            self.c = kwargs['constC']

    def get_args(self):
        return self._args.copy()

    def _quad_part(self, x):
        return np.dot(np.dot(self.A, x), x)  / 2. # returns 0.5(Ax,x)

    def _linear_part(self, x):
        return np.dot(self.b.T, x)  # returns (b,x)

    def val(self, x):
        return (self._quad_part(x) + self._linear_part(x) + self.c)[0, 0]

    def diff(self, x):
        return np.dot(self.A, x).T + self.b

    def gesse(self, x=[]):
        return self.A


class Alg_Quadratic(object):
    MAX_ITER = 100000

    class ToFile(object):
        def __init__(self, i, xk, f, f_deriv, norm1, norm2, norm3):
            self.i = i
            self.xk = xk
            self.f = f
            self.f_deriv = f_deriv
            self.norm1 = norm1
            self.norm2 = norm2
            self.norm3 = norm3

        def tostr(self):
            rep = "######################################################" + 'iters: ' + str(self.i) + '\nXi=' \
                  + str(self.xk) + '\nF(Xi)=' + str(self.f) + "\nF'(Xk)=" \
                  + str(self.f_deriv) + '\n||Xi+1 - X||=' + str(self.norm1) + '\n||F(Xi+1)-F(Xi)||=' \
                  + str(self.norm2) + "\n||F'(Xi+1)-F'(Xi)||=" + str(self.norm3) + '\n'
            return rep

    def __init__(self, matrix, vector, x0, e1, e2, e3, iterations):
        self.A = matrix
        self.B = vector
        self.X0 = x0
        self.eps1 = e1
        self.eps2 = e2
        self.eps3 = e3
        self.iters = iterations

    def function(self, x):
        '''
         return 1/2(Ax, x) + (b, x)
         (A * np.matrix(x)).trans()*
        '''
        return 1 / 2 * float(np.dot(np.transpose(np.dot(self.A, x)), x)) + float(np.dot(np.transpose(self.B), x))

    def dif(self, x):
        return np.dot(self.A, x) + self.B

    def Hk(self, x):
        return -1 * (self.dif(x))

    def alpha(self, x):
        '''
        :param x: input print
        :return: -(f'(x), h)/(Ah,h)
        '''
        m = np.dot(np.transpose(self.dif(x)), self.Hk(x))
        n = np.dot(np.transpose(np.dot(self.A, self.Hk(x))), self.Hk(x))
        return -1 * float(m) / float(n)

    def norm_max(self, x):
        return np.linalg.norm(x, np.inf)

    def iteration(self, path='output.txt'):
        xk = self.X0
        filename = open(path, 'a+')
        i = 0
        while (True):
            i += 1
            if i > self.MAX_ITER:
                print('iter more than ' + str(self.MAX_ITER))
                filename.close()
                return 0
            xk1 = xk + self.alpha(xk) * self.Hk(xk)
            f = self.function(xk)
            f1 = self.function(xk1)
            f_deriv1 = self.dif(xk1)
            norm1 = self.norm_max(xk1 - xk)
            norm2 = abs(f1 - f)
            norm3 = self.norm_max(f_deriv1)
            a = self.ToFile(i, xk1, f1, f_deriv1, norm1, norm2, norm3)
            self.output(filename, a)
            if norm1 < self.eps1 and (norm2 < self.eps2) and (norm3 < self.eps3):
                filename.close()
                print(a.tostr())
                return 0
            xk = xk1

    def output(self, f, a):
        f.write(a.tostr())













