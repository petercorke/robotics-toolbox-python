#!/usr/bin/env python
"""
@author Jesse Haviland
"""

import numpy as np
import roboticstoolbox as rp
from spatialmath import SE3


class RobotPlot(object):

    def __init__(
            self, robot, ax, readonly, display=True,
            jointaxes=True, eeframe=True, shadow=True, name=True):

        super(RobotPlot, self).__init__()

        # Readonly - True for this robot is for displaying only
        self.readonly = readonly

        # To show to robot in the plot or not
        # If not displayed, the robot is still simulated
        self.display = display

        self.robot = robot
        self.ax = ax

        # Line plot of robot links
        self.links = None

        # Z-axis Coordinate frame (quiver) of joints
        self.joints = []

        # Text of the robots name
        self.name = None

        # Shadow of the the line plot on the x-y axis
        self.sh_links = None

        # Coordinate frame of the ee (three quivers)
        self.ee_axes = []

        # Robot has been drawn
        self.drawn = False

        # Display options
        self.eeframe = eeframe
        self.jointaxes = jointaxes
        self.shadow = shadow
        self.showname = name

    def axes_calcs(self):
        # Joint and ee poses
        T = self.robot.fkine_all(self.robot.q)

        try:
            Te = self.robot.fkine(self.robot.q)
        except ValueError:
            print(
                "\nError: Branched robot's not yet supported "
                "with PyPlot backend\n")
            raise

        Tb = self.robot.base

        # Joint and ee position matrix
        loc = np.zeros([3, len(self.robot.links) + 1])
        loc[:, 0] = Tb.t

        # Joint axes position matrix
        joints = np.zeros((3, self.robot.n))

        # Axes arrow transforms
        Tjx = SE3.Tx(0.06)
        Tjy = SE3.Ty(0.06)
        Tjz = SE3.Tz(0.06)

        # ee axes arrows
        Tex = Te * Tjx
        Tey = Te * Tjy
        Tez = Te * Tjz

        # Joint axes arrow calcs
        if isinstance(self.robot, rp.ERobot):
            i = 0
            j = 0
            for link in self.robot.links:
                loc[:, i + 1] = link._fk.t

                if link.isjoint:
                    if link.v.axis == 'Rz' or link.v.axis == 'tz':
                        Tji = link._fk * Tjz

                    elif link.v.axis == 'Ry' or link.v.axis == 'ty':
                        Tji = link._fk * Tjy

                    elif link.v.axis == 'Rx' or link.v.axis == 'tx':
                        Tji = link._fk * Tjx

                    joints[:, j] = Tji.t
                    j += 1

                i += 1
            loc = np.c_[loc, loc[:, -1]]
        else:
            # End effector offset (tool of robot)
            loc = np.c_[loc, Te.t]

            for i in range(self.robot.n):
                loc[:, i + 1] = T[i].t
                Tji = T[i] * Tjz
                joints[:, i] = Tji.t

        return loc, joints, [Tex, Tey, Tez]

    def draw(self):
        if not self.display:
            return

        if not self.drawn:
            self.init()
            return

        loc, joints, ee = self.axes_calcs()

        # Remove old ee coordinate frame
        if self.eeframe:
            self.ee_axes[0].remove()
            self.ee_axes[1].remove()
            self.ee_axes[2].remove()

            # Plot ee coordinate frame
            self.ee_axes[0] = \
                self._plot_quiver(
                    loc[:, -1], ee[0].t, '#EE9494', 2)
            self.ee_axes[1] = \
                self._plot_quiver(
                    loc[:, -1], ee[1].t, '#93E7B0', 2)
            self.ee_axes[2] = \
                self._plot_quiver(
                    loc[:, -1], ee[2].t, '#54AEFF', 2)

        # Remove oldjoint z coordinates
        if self.jointaxes:
            j = 0

            for joint in self.joints:
                self.ax.collections.remove(joint)

            del self.joints
            self.joints = []

            # Plot joint z coordinates
            for i in range(len(self.robot.links)):
                if isinstance(self.robot, rp.DHRobot) or \
                        self.robot.links[i].isjoint:
                    self.joints.append(
                        self._plot_quiver(
                            loc[:, i+1], joints[:, j], '#8FC1E2', 2))
                    j += 1

            # for i in range(len(self.robot.links)):
            #     self.joints[i] = \
            #         self._plot_quiver(loc[:, i+1], joints[:, i], '#8FC1E2', 2)

        # Update the robot links
        self.links[0].set_xdata(loc[0, :])
        self.links[0].set_ydata(loc[1, :])
        self.links[0].set_3d_properties(loc[2, :])

        # Update the shadow of the robot links
        if self.shadow:
            self.sh_links[0].set_xdata(loc[0, :])
            self.sh_links[0].set_ydata(loc[1, :])
            self.sh_links[0].set_3d_properties(0)

    def draw2(self):
        if not self.display:
            return

        if not self.drawn:
            self.init2()
            return

        loc, joints, ee = self.axes_calcs()

        # Remove old ee coordinate frame
        if self.eeframe:
            self.ee_axes[0].remove()
            self.ee_axes[1].remove()

            # Plot ee coordinate frame
            self.ee_axes[0] = \
                self._plot_quiver2(
                    loc[:, -1], ee[0].t, '#EE9494', 2)
            self.ee_axes[1] = \
                self._plot_quiver2(
                    loc[:, -1], ee[1].t, '#93E7B0', 2)

        # Update the robot links
        self.links[0].set_xdata(loc[0, :])
        self.links[0].set_ydata(loc[1, :])

    def init(self):

        self.drawn = True

        # Joint and ee poses
        Tb = self.robot.base
        loc, joints, ee = self.axes_calcs()

        # Plot robot name
        if self.showname:
            self.name = self.ax.text(
                Tb.t[0], Tb.t[1], 0.05, self.robot.name)

        # Plot ee coordinate frame
        if self.eeframe:
            self.ee_axes.append(
                self._plot_quiver(
                    loc[:, -1], ee[0].t, '#EE9494', 2))
            self.ee_axes.append(
                self._plot_quiver(
                    loc[:, -1], ee[1].t, '#93E7B0', 2))
            self.ee_axes.append(
                self._plot_quiver(
                    loc[:, -1], ee[2].t, '#54AEFF', 2))

        # Plot joint z coordinates
        if self.jointaxes:
            j = 0
            for i in range(len(self.robot.links)):
                if isinstance(self.robot, rp.DHRobot) or \
                        self.robot.links[i].isjoint:
                    self.joints.append(
                        self._plot_quiver(
                            loc[:, i+1], joints[:, j], '#8FC1E2', 2))
                    j += 1

        # Plot the shadow of the robot links, draw first so robot is always
        # in front
        if self.shadow:
            self.sh_links = self.ax.plot(
                loc[0, :], loc[1, :],
                linewidth=3, color='lightgrey')

        # Plot the robot links
        self.links = self.ax.plot(
            loc[0, :], loc[1, :], loc[2, :], linewidth=5, color='#E16F6D')

    def init2(self):

        self.drawn = True

        # Joint and ee poses
        Tb = self.robot.base
        loc, joints, ee = self.axes_calcs()

        # Plot robot name
        if self.showname:
            self.name = self.ax.text(
                Tb.t[0] + 0.05, Tb.t[1], self.robot.name)

        # Plot ee coordinate frame
        if self.eeframe:
            self.ee_axes.append(
                self._plot_quiver2(
                    loc[:, -1], ee[0].t, '#EE9494', 2))
            self.ee_axes.append(
                self._plot_quiver2(
                    loc[:, -1], ee[1].t, '#93E7B0', 2))

        # Plot the robot links
        self.links = self.ax.plot(
            loc[0, :], loc[1, :], linewidth=5, color='#E16F6D')

    def _plot_quiver(self, p0, p1, col, width):
        qv = self.ax.quiver(
            p0[0], p0[1], p0[2],
            p1[0] - p0[0],
            p1[1] - p0[1],
            p1[2] - p0[2],
            linewidth=width,
            color=col
        )

        return qv

    def _plot_quiver2(self, p0, p1, col, width):
        qv = self.ax.quiver(
            p0[0], p0[1],
            p1[0] - p0[0],
            p1[1] - p0[1],
            linewidth=width,
            color=col
        )

        return qv
