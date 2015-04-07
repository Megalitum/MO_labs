__author__ = 'vlad'

import numpy as np
import re


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

    def input_data(self, path='input.txt'):
        lines = self._input_from_file(path)
        param = re.compile(r'(\S+?) \s* = \s* (.+?) \s*$', re.VERBOSE)
        dt = dict()
        for line in lines:
            res = param.search(line)
            if res:
                dt[res.group(1)] = res.group(2)
        try:
            function = dt['Function']
            function = re.sub(r'([-+])([a-zA-Z])', r'\g<1>1*\2', function)
            arguments = dt['Vars'].split()
            x_string = dt['Point'].split()
            x0 = self._string_to_float(x_string)
            eps1 = float(dt['Eps1'])
            eps2 = float(dt['Eps2'])
            eps3 = float(dt['Eps3'])
            iterations = int(dt['MaxIter'])
        except:
            print('File read failed')
            exit(0)

        return function, arguments, x0, eps1, eps2, eps3, iterations

    def is_empty_list(self, l):
        if len(l) == 0:
            return True
        else:
            return False

    def check_quadratic(self, function):
        parts = re.split(r'\b[-+]\b', function)
        quad_patt = re.compile(r'''
            ^(?:
                (?:\(-?\d+(?:\.\d+)?\))|
                \d+(?:\.\d+)?
            )
            (?:\*[a-zA-Z]\*\*2|(?:\*[a-zA-Z]){0,2})$
        ''', re.VERBOSE)
        for summand in parts:
            re.sub(r'\s', '', summand)
            if not quad_patt.match(summand):
                return False
        return True


    def read_quadratic(self):  # what's the point??
        return

    def create_matrix_A(self, function, arguments):
        matrix = []
        n = len(arguments)
        # print(n)
        for i in range(n):
            mas = []
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
            matrix.append(mas)
        Matrix = np.matrix(matrix)
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
