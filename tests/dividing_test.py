__author__ = 'vlad'

from nose.tools import *
from gradient.function import Function as func
from gradient.dividing import DividingMethod
import numpy as np

def test_1():
    f = func('1000000*x**4+x**2+y**2+1',['x','y'])
    method = DividingMethod([200.12, 200],f,0.00001)
    method.launch()
    method.print_to_file('output.txt')
    method.plot_graph()

