__author__ = 'vlad'

import numpy as np
import matplotlib.pyplot as plt


class Algorithm(object):

    def __init__(self, init_point, function):
        self.points = [np.array(init_point, ndmin=1)]
        self.dimension = len(self.points[0])
        self.function = function
        try:
            function(self.points[0])
        except Exception as ex:
            raise Exception('Failed calculating init point value', ex)


    def iteration(self):
        pass

    def launch(self):
        pass

    def _generate_strings(self):
        mask = '#{number:<7} x: {point}\nf: {value}'
        for number in range(len(self.points)):
            point = self.points[number]
            yield mask.format(number=number, point=str(point), value=self.function(point))

    def print_to_file(self, filename):
        file = open(filename, 'w')
        file.writelines(self._generate_strings())


    def plot_graph(self, color = 'r', alpha = 0.6):
        """
        works for 2-dimensional only
        """
        if self.dimension != 2:
            raise Exception('Bad idea')
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x, y = np.transpose(self.points)
        z = list(map(self.function, self.points))
        ax.scatter(x, y, z, c = color, alpha = alpha)
        plt.show()
