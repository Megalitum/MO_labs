__author__ = 'yana'

from nose.tools import *
import gradient.function_input as fi
from numpy import matrix

def test_1():
    string = '1*x**2-1*y**2+2*x*y+2*x+y+5'
    _ = fi.FunctionInput()
    result = _.create_matrix_A(string, ['x', 'y'])
    expected = matrix([[1, 1],[1, -1]])
    print(result.all())
    assert_true((result == expected).all())

