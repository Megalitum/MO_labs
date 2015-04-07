__author__ = 'vlad'

from sympy import S
import numpy as np


class Function(object):
    def __init__(self, string, arguments):
        self._args = arguments
        self._func = S(string)
        self._diffs = [self._func.diff(coord) for coord in self._args]
        self._gesse = [[diff.diff(coord) for coord in self._args] for diff in self._diffs]

    def val(self, x):
        dict = {key: value for key, value in zip(self._args, x)}
        return self._func.subs(dict)

    def diff(self, x):
        dict = {key: value for key, value in zip(self._args, x)}
        values = [diff.subs(dict) for diff in self._diffs]
        return np.matrix(values).T

    def gesse(self, x):
        dict = {key: value for key, value in zip(self._args, x)}
        values = [[diff2.subs(dict) for diff2 in row] for row in self._gesse]
        return np.matrix(values)