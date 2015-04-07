__author__ = 'vlad'

from nose.tools import *
from gradient.function import Function as func
from gradient.dividing import DividingMethod
import numpy as np

def test_1():
    f = func('x**4+x**2+1',['x'])
    method = DividingMethod([200.12],f,0.00001)
    method.launch()
    print(method.points)

test_1()