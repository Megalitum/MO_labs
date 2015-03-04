#!/usr/bin/python
#coding=UTF-8

#from sympy import *
import copy
import re

import numpy as np


class Alg_Qudratic():
    def input_from_file(self, path="input.txt"):
        lines = []
        try:
            f = open(path, 'r')
            lines = f.readlines()
        finally:
            f.close()
        return lines

    def str_to_float(self, str):
        try:
            return float(str)
        except:
            return 0

    def str_to_int(self, str):
        try:
            return int(str)
        except:
            return 0

    def string_to_float(self, str):
        try:
            x = []
            for i in range(len(str)):
                x.append(float(str[i]))
            return x
        except:
            return 0

    def input_data(self):
        lines = self.input_from_file()
        arguments = lines[1].split()
        x_string = lines[2].split()
        x0 = self.string_to_float(x_string)
        return [lines[0], arguments, x0]

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
                        mas.append(float(p.findall(function)[0]) / 2)
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
                        mas.append((float(p_l[0]) + float(z_l[0])) /2)
                    except:
                        mas.append(0.0)

            matrix.append(copy.deepcopy(mas))
            del mas[:]

        Matrix = np.copy(matrix)
        return Matrix





ob = Alg_Qudratic()
list1 = ob.input_data()
matrix = ob.create_matrix_A(list1[0], list1[1])
print(matrix)










