__author__ = 'vlad'

from sympy import S
import numpy as np
from math import *

def Abs(x):
    return abs(x)


class Function(object):
    def __init__(self, string, arguments):
        self._args = arguments
        self._func = S(string)
        temp_diff = [self._func.diff(coord) for coord in self._args]
        self._gesse = [[str(diff.diff(coord)) for coord in self._args] for diff in temp_diff]
        self._diffs = list(map(str,temp_diff))
        self._fstring = str(self._func)

    def get_args(self):
        return self._args.copy()

    def val(self, x):
        arguments = {key: value for key, value in zip(self._args, x)}
        return eval(self._fstring, globals(), arguments)

    def diff(self, x):
        arguments = {key: value for key, value in zip(self._args, x)}
        values = [eval(diff,globals(),arguments) for diff in self._diffs]
        return np.matrix(values).T

    def gesse(self, x):
        arguments = {key: value for key, value in zip(self._args, x)}
        values = [[eval(diff2,globals(),arguments) for diff2 in row] for row in self._gesse]
        return np.matrix(values)