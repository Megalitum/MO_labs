__author__ = 'vlad'

from nose.tools import *
from gradient.quadratic import Quadratic_Func as qfunc

def test_params():
    string = '1*x**2-1*y**2+2*x*y+2*x-3*y+4'
    args = ['x','y']
    f = qfunc(string, args)
    print(f.A)
    print(f.b)
    print(f.c)

def test_val():
    string = '1*x**2-1*y**2+2*x*y+2*x-3*y+4'
    args = ['x','y']
    point = [1,1]
    f = qfunc(string, args)
    print(f.val(point))

def test_diff():
    string = '1*x**2-1*y**2+2*x*y+2*x-3*y+4'
    args = ['x','y']
    point = [1,1]
    f = qfunc(string, args)
    print(f.diff(point))

def test_gesse():
    string = '1*x**2-1*y**2+2*x*y+2*x-3*y+4'
    args = ['x','y']
    f = qfunc(string, args)
    print(f.gesse())

test_params()
test_val()
test_diff()
test_gesse()