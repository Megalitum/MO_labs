__author__ = 'vlad'

from nose.tools import *
from gradient.function import Function as func
import numpy as np

def test_val():
    string = '1*x**2-1*y**2+2*x*y+2*x-3*y+4'
    args = ['x','y']
    point = [1,1]
    f = func(string, args)
    print(f.val(point))

def test_diff():
    string = '1*x**2-1*y**2+2*x*y+2*x-3*y+4'
    args = ['x','y']
    point = [1,1]
    f = func(string, args)
    print(f.diff(point))

def test_gesse():
    string = '1*x**2-1*y**2+2*x*y+2*x-3*y+4'
    args = ['x','y']
    point = [1,1]
    f = func(string, args)
    print(f.gesse(point))

def test_val_vec():
    string = '1*x**2-1*y**2+2*x*y+2*x-3*y+4'
    args = ['x','y']
    point = np.array([1,1])
    f = func(string, args)
    print(f.val(point))


test_val()
test_diff()
test_gesse()
test_val_vec()