__author__ = 'vlad'

import numpy as np
import pylab as plb
from mpl_toolkits.mplot3d import Axes3D


class Algorithm(object):
    def __init__(self, init_point, function):
        self.points = [np.array(init_point, ndmin=1, dtype=np.float_)]
        self.dimension = len(self.points[0])
        self.function = function
        try:
            function.val(self.points[0])
        except Exception as ex:
            raise Exception('Failed calculating init point value', ex)


    def iteration(self):
        pass

    def launch(self):
        pass

    def get_last_point(self):
        return self.points[-1]

    def _get_mask(self):
        return '#{number:<7} x: {point}\nf: {value}\n'

    def _generate_strings(self):
        mask = self._get_mask()
        for number in range(len(self.points)):
            point = self.points[number]
            yield mask.format(number=number, point=str(point), value=self.function.val(point))

    def print_to_file(self, filename):
        file = open(filename, 'w')
        file.writelines(self._generate_strings())
        file.close()


    def graph_3d(self, color='r', alpha=0.6):
        """
        works for 2-dimensional only
        """
        if self.dimension != 2:
            raise Exception('Cannot build graph in this dimension')
        fig = plb.figure()
        ax = fig.add_subplot(111, projection='3d')
        x, y = np.transpose(self.points)
        z = list(map(self.function.val, self.points))
        ax.scatter(x, y, z, c=color, alpha=alpha)

    def graph_path(self, zoom = 1):
        if self.dimension != 2:
            raise Exception('Cannot build graph in this dimension')
        fig = plb.figure(2)
        x, y = np.transpose(self.points)
        plb.grid(True)
        xlast, ylast = self.points[-1]

        x_all = np.linspace(xlast - zoom, xlast + zoom, 50)
        y_all = np.linspace(ylast - zoom, ylast + zoom, 50)
        xx, yy = np.meshgrid(x_all, y_all)
        f = self.function.val((xx,yy))
        plb.contour(x_all, y_all, f, 50, alpha = 0.5)
        plb.plot(x,y)
        plb.xlim((xlast - zoom, xlast + zoom))
        plb.ylim((ylast - zoom, ylast + zoom))
        plb.scatter(x[:-1],y[:-1],c='r', marker='o', alpha=0.5)
        plb.scatter(xlast, ylast,s=50, c='b', marker='*')




