#!/usr/bin/env python
"""
@author Jesse Haviland
"""

import numpy as np


class Ellipse(object):

    def __init__(self, ax):

        super(Ellipse, self).__init__()

        self.ax = ax
        self.ell = None

    def draw_ellipsoid(self):
        if self.ell is not None:
            self.ax.collections.remove(self.ell)

        self.ell = self.ax.plot_wireframe(
            self.x, self.y, self.z,  rstride=6,
            cstride=6, color='#2980b9', alpha=0.2)

    def make_ellipsoid(self, A, centre=[0, 0, 0]):
        """
        Plot the 3-d Ellipsoid ell on the Axes3D ax.

        """

        # find the rotation matrix and radii of the axes
        U, s, rotation = np.linalg.svd(A)
        radii = 1.0 / np.sqrt(s)

        # points on unit sphere
        u = np.linspace(0.0, 2.0 * np.pi, 50)
        v = np.linspace(0.0, np.pi, 50)
        x = radii[0] * np.outer(np.cos(u), np.sin(v))
        y = radii[1] * np.outer(np.sin(u), np.sin(v))
        z = radii[2] * np.outer(np.ones_like(u), np.cos(v))

        # transform points to ellipsoid
        for i in range(len(x)):
            for j in range(len(x)):
                [x[i, j], y[i, j], z[i, j]] = \
                    np.dot([x[i, j], y[i, j], z[i, j]], rotation) + centre * 10

        self.x = x / 10
        self.y = y / 10
        self.z = z / 10
