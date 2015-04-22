__author__ = 'vlad'


from gradient.quadratic import Alg_Quadratic
import numpy as np # must be removed
import pylab as plb
from gradient.function_input import FunctionInput
from gradient.function import Function as func
from gradient.dividing import DividingMethod
from gradient.newton import NewtonMethod
from gradient.projection import ProjectionMethod

def main(filepath='input.txt'):
    choice = input('Choose method (1 - quadratic, 2 - dividing, 3 - newton, 4 - projection > ')
    if choice == '1':
        ob = FunctionInput()
        list1 = ob.input_data(filepath)
        if ob.check_quadratic(list1[0]):
            matrix = ob.create_matrix_A(list1[0], list1[1])
            b = ob.create_b(list1[0], list1[1])
            X = np.matrix(list1[2])
            x = np.transpose(X)
            eps1 = list1[3]
            eps2 = list1[4]
            eps3 = list1[5]
            iters = list1[6]
            print(matrix)
            print(b)
            print(x)

            obj = Alg_Quadratic(2 * matrix, b, x, eps1, eps2, eps3, iters)
            obj.iteration()
        else:
            print('Input func was not parsed')

    elif choice == '2':
        #f = func('(x-4)**2+400*(y+4)**2+2*(x-4)*(y+4)',['x','y'])
        #f = func('x**2 +3*y**2 +2*z**2', ['x','y', 'z'])
        #method = DividingMethod([10, 9, 8],f,0.000001)
        f = func('x**2 +8*y**2 +0.001*x*y-x-y', ['x','y'])
        method = DividingMethod([10000000, 10],f,0.00000001)
        method.launch()
        method.print_to_file('output.txt')
        #method.graph_3d()
        method.graph_path()
        plb.show()
    elif choice == '3':
        f = func('x**2 +8*y**2 +0.001*x*y-x-y', ['x','y'])
        method = NewtonMethod([10000000, 10],f)
        method.launch()
        method.print_to_file('output.txt')
        method.graph_path()
        plb.show()
    elif choice == '4':
        f = func('x**2 +3*y**2 +2*z**2', ['x','y', 'z'])
        H = func('2*x+y+3*z -1', ['x', 'y', 'z'])
        method = ProjectionMethod([3,0,3], f, H)
        method.launch()
        method.print_to_file('output3.txt')


main()