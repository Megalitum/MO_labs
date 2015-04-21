__author__ = 'vlad'

from sympy import S
import numpy as np


class Function(object):
    def __init__(self, string, arguments):
        self._args = arguments
        self._func = S(string)
        temp_diff = [self._func.diff(coord) for coord in self._args]
        self._gesse = [[str(diff.diff(coord)) for coord in self._args] for diff in temp_diff]
        self._diffs = list(map(str,temp_diff))

    def val(self, x):
        arguments = {key: value for key, value in zip(self._args, x)}
        return eval(str(self._func), arguments)

    def diff(self, x):
        arguments = {key: value for key, value in zip(self._args, x)}
        values = [eval(diff,arguments) for diff in self._diffs]
        return np.matrix(values).T

    def gesse(self, x):
        arguments = {key: value for key, value in zip(self._args, x)}
        values = [[eval(diff2,arguments) for diff2 in row] for row in self._gesse]
        return np.matrix(values)