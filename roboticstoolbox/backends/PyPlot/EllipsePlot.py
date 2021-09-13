#!/usr/bin/env python
"""
@author Jesse Haviland
"""

import numpy as np
import scipy as sp
from spatialmath import base
import matplotlib.pyplot as plt


class ShapePlot():

    def __init__(self, shape, wireframe=True, **kwargs):

        self.shape = shape  # reference to the spatialgeom shape
        self.wireframe = wireframe
        self.args = kwargs
        self.mpl = None

    def plot(self):
        if ax is None:
            ax = self.ax

        if ax is None:
            fig = plt.figure()
            ax = plt.axes(projection="3d")
            self.ax = ax

        self.draw()

    def draw(self, ax=None):
        # TODO only remove and redraw if base has changed
        if self.mpl is not None:
            self.mpl.remove()

        if self.shape.stype == 'box':
            # scale
            self.mpl = base.plot_cuboid(sides=self.shape.scale, 
                pose=self.shape.base, ax=ax)

        elif self.shape.stype == 'sphere':
            print(self.shape.base.t)
            self.mpl = base.plot_sphere(self.shape.radius, pose=self.shape.base, ax=ax)

        elif self.shape.stype == 'cylinder':
            # radius, length
            pass

    def make(self):
        pass

class EllipsePlot:
    def __init__(self, robot, q, etype, opt="trans", centre=[0, 0, 0], scale=1):

        super(EllipsePlot, self).__init__()

        try:
            centre = base.getvector(centre, 3)
        except ValueError:
            centre = base.getvector(centre, 2)
            centre = np.array([centre[0], centre[1], 0])
        except TypeError:
            if centre != "ee":
                raise ValueError(
                    "Centre must be a three vector or 'ee' meaning" "end-effector"
                )

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

        if etype == "v":
            self.vell = True
            self.name = "Velocity Ellipse"
        elif etype == "f":
            self.vell = False
            self.name = "Force Ellipse"

    def draw(self):
        self.make_ellipsoid()

        if self.ell is not None:
            self.ax.collections.remove(self.ell)

        self.ell = self.ax.plot_wireframe(
            self.x, self.y, self.z, rstride=6, cstride=6, color="#2980b9", alpha=0.2
        )

    def draw2(self):
        self.make_ellipsoid2()

        if self.ell is not None:
            self.ell[0].set_data(self.x, self.y)
        else:
            self.ell = self.ax.plot(self.x, self.y, color="#2980b9")

    def plot(self, ax=None):
        if ax is None:
            ax = self.ax

        if ax is None:
            fig = plt.figure()
            ax = plt.axes(projection="3d")
            self.ax = ax

        self.draw()

    def plot2(self, ax=None):
        if ax is None:
            ax = self.ax

        if ax is None:
            ax = plt.axes()
            self.ax = ax

        self.draw2()

    def make_ellipsoid(self):
        """
        Plot the 3d Ellipsoid

        """

        if self.opt == "trans":
            J = self.robot.jacobe(self.q)[3:, :]
            A = J @ J.T
        elif self.opt == "rot":
            J = self.robot.jacobe(self.q)[:3, :]
            A = J @ J.T

        if not self.vell:
            # Do the extra step for the force ellipse
            A = np.linalg.inv(A)

        if isinstance(self.centre, str) and self.centre == "ee":
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
                [x[i, j], y[i, j], z[i, j]] = np.dot(
                    [x[i, j], y[i, j], z[i, j]], rotation
                )

        self.x = x * self.scale + centre[0]
        self.y = y * self.scale + centre[1]
        self.z = z * self.scale + centre[2]

    def make_ellipsoid2(self):
        """
        Plot the 2d Ellipsoid

        """

        if self.opt == "trans":
            J = self.robot.jacob0(self.q)[:2, :]
            A = J @ J.T
        elif self.opt == "rot":
            raise ValueError(
                "Can not do rotational ellipse for a 2d robot plot." " Set opt='trans'"
            )

        # if not self.vell:
        #     # Do the extra step for the force ellipse
        #     try:
        #         A = np.linalg.inv(A)
        #     except:
        #         A = np.zeros((2,2))

        if isinstance(self.centre, str) and self.centre == "ee":
            centre = self.robot.fkine(self.q).t
        else:
            centre = self.centre

        # points on unit circle
        theta = np.linspace(0.0, 2.0 * np.pi, 50)
        y = np.array([np.cos(theta), np.sin(theta)])
        # RVC2 p 602
        # x = sp.linalg.sqrtm(A) @ y

        x, y = base.ellipse(A, inverted=True, centre=centre[:2], scale=self.scale)
        self.x = x
        self.y = y
        # = x[0,:] * self.scale + centre[0]
        # self.y = x[1,:] * self.scale + centre[1]
