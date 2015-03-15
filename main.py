__author__ = 'vlad'


from gradient.quadratic import Alg_Quadratic
import numpy as np # must be removed
from gradient.function_input import FunctionInput

ob = FunctionInput()
list1 = ob.input_data()
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

obj = Alg_Quadratic(matrix, b, x, eps1, eps2, eps3, iters)
obj.iteration()