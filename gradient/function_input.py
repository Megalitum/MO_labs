__author__ = 'vlad'

import numpy as np
import re
import copy

class FunctionInput():

    def _input_from_file(self, path):
        lines = []
        try:
            f = open(path, 'r')
            lines = f.readlines()
        finally:  # add catch
            f.close()
        return lines

    def _str_to_float(self, str):
        try:
            return float(str)
        except:
            return 0

    def _str_to_int(self, str):
        try:
            return int(str)
        except:
            return 0

    def _string_to_float(self, str):
        try:
            x = []
            for i in range(len(str)):
                x.append(float(str[i]))
            return x
        except:
            return 0

    def input_data(self, path = 'input.txt'):
        lines = self._input_from_file(path)

        arguments = lines[1].split()
        x_string = lines[2].split()
        x0 = self._string_to_float(x_string)
        eps1 = float(lines[3])
        eps2 = float(lines[4])
        eps3 = float(lines[5])
        iterations = int(lines[6])
        return [lines[0], arguments, x0, eps1, eps2, eps3, iterations]

    def is_empty_list(self, l):
        if len(l) == 0:
            return True
        else:
            return False

    def create_matrix_A(self, function, arguments):
        matrix = []
        mas = []
        n = len(arguments)
        # print(n)
        for i in range(n):
            for j in range(n):
                if i == j:
                    p = re.compile('([+-]?\d+|[+-]?\d+\.\d+)\*?' + arguments[i] + '\*{2}2')
                    try:
                        mas.append(float(p.findall(function)[0]))
                    except:
                        mas.append(0.0)
                else:
                    p = re.compile('([+-]?\d+|[+-]?\d+\.\d+)\*?' + arguments[i] + '\*' + arguments[j])
                    z = re.compile('([+-]?\d+|[+-]?\d+\.\d+)\*?' + arguments[j] + '\*' + arguments[i])
                    p_l = p.findall(function)
                    if self.is_empty_list(p_l):
                        p_l.append(0.0)

                    z_l = z.findall(function)
                    if self.is_empty_list(z_l):
                        z_l.append(0.0)
                    try:
                        mas.append((float(p_l[0]) + float(z_l[0])) / 2)
                    except:
                        mas.append(0.0)

            matrix.append(copy.deepcopy(mas))
            del mas[:]
        Matrix = np.copy(matrix)
        Matrix = 2 * Matrix
        return Matrix

    def create_b(self, function, arguments):
        n = len(arguments)
        b = []
        for i in range(n):
            p = re.compile('([+-]?\d+|[+-]?\d+\.\d+)\*?' + arguments[i] + '[^*]')
            p_l = p.findall(function)
            if self.is_empty_list(p_l):
                b.append([0.0])
            else:
                b.append([float(p_l[0])])
        B = np.copy(b)
        return B
