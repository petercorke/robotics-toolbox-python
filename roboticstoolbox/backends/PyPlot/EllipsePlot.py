#!/usr/bin/env python
"""
@author Jesse Haviland
"""

import numpy as np
from spatialmath.base.argcheck import getvector


class EllipsePlot(object):

    def __init__(self, robot, q, etype, opt='trans', centre=[0, 0, 0], scale=0.1):

        super(EllipsePlot, self).__init__()

        try:
            centre = getvector(centre, 3)
        except ValueError:
            centre = getvector(centre, 2)
            centre = np.array([centre[0], centre[1], 0])
        except TypeError:
            if centre != 'ee':
                raise ValueError(
                    'Centre must be a three vector or \'ee\' meaning'
                    'end-effector')

        self.ell = None
        self.robot = robot
        self.opt = opt
        self.centre = centre
        self.ax = None
        self.scale = scale

        if q is None:
            self.q = robot.q
        else:
            self.q = q

        if etype == 'v':
            self.vell = True
            self.name = 'Velocity Ellipse'
        elif etype == 'f':
            self.vell = False
            self.name = 'Force Ellipse'

    def draw(self):
        self.make_ellipsoid()

        if self.ell is not None:
            self.ax.collections.remove(self.ell)

        self.ell = self.ax.plot_wireframe(
            self.x, self.y, self.z, rstride=6,
            cstride=6, color='#2980b9', alpha=0.2)

    def draw2(self):
        self.make_ellipsoid2()

        if self.ell is not None:
            self.ell[0].set_data(self.x, self.y)
        else:
            self.ell = self.ax.plot(
                self.x, self.y, color='#2980b9', alpha=0.2)

    def make_ellipsoid(self):
        """
        Plot the 3d Ellipsoid

        """

        if self.opt == 'trans':
            J = self.robot.jacobe(self.q)[3:, :]
            A = J @ J.T
        elif self.opt == 'rot':
            J = self.robot.jacobe(self.q)[:3, :]
            A = J @ J.T

        if not self.vell:
            # Do the extra step for the force ellipse
            A = np.linalg.inv(A)

        if isinstance(self.centre, str) and self.centre == 'ee':
            centre = self.robot.fkine(self.q).t
        else:
            centre = self.centre

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
                    np.dot([x[i, j], y[i, j], z[i, j]], rotation)

        self.x = x * self.scale + centre[0]
        self.y = y * self.scale + centre[1]
        self.z = z * self.scale + centre[2]

    def make_ellipsoid2(self):
        """
        Plot the 2d Ellipsoid

        """

        if self.opt == 'trans':
            J = self.robot.jacobe(self.q)[:2, :]
            A = J @ J.T
        elif self.opt == 'rot':
            raise ValueError(
                "Can not do rotational ellipse for a 2d robot plot."
                " Set opt='trans'")

        if not self.vell:
            # Do the extra step for the force ellipse
            A = np.linalg.inv(A)

        if isinstance(self.centre, str) and self.centre == 'ee':
            centre = self.robot.fkine(self.q).t
        else:
            centre = self.centre

        # find the rotation matrix and radii of the axes
        U, s, rotation = np.linalg.svd(A)
        radii = 1.0 / np.sqrt(s)

        # points on unit sphere
        u = np.linspace(0.0, 2.0 * np.pi, 50)
        v = np.linspace(0.0, np.pi, 50)
        x = radii[0] * np.outer(np.cos(u), np.sin(v))
        y = radii[1] * np.outer(np.sin(u), np.sin(v))

        # transform points to ellipsoid
        for i in range(len(x)):
            for j in range(len(x)):
                [x[i, j], y[i, j]] = \
                    np.dot([x[i, j], y[i, j]], rotation)

        self.x = x * self.scale + centre[0]
        self.y = y * self.scale + centre[1]
