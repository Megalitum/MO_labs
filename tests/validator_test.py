__author__ = 'vlad'

from nose.tools import *
import gradient.function_input as fi


def test_func_not_quad1():
    string = 'x**2+x*y+x*y*z'
    _ = fi.FunctionInput()
    result = _.check_quadratic(string)
    expected = False
    assert_equal(result, expected)

def test_func_not_quad2():
    string = '1*x**3+2*x*y'
    _ = fi.FunctionInput()
    result = _.check_quadratic(string)
    expected = False
    assert_equal(result, expected)

def test_func_quad1():
    string = '1*x**2+3*x*y'
    _ = fi.FunctionInput()
    result = _.check_quadratic(string)
    expected = True
    assert_equal(result, expected)

def test_func_quad2():
    string = '(-1)*x**2+1*y**2+1*z+1*t+4+1*u+7*U'
    _ = fi.FunctionInput()
    result = _.check_quadratic(string)
    expected = True
    assert_equal(result, expected)
