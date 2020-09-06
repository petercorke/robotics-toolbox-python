#!/usr/bin/env python
"""
@author Jesse Haviland
"""

import numpy as np
import ropy as rp
import spatialmath as sm


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
        T = self.robot.allfkine()
        Te = self.robot.fkine()
        Tb = self.robot.base

        # Joint and ee position matrix
        loc = np.zeros([3, self.robot.n + 2])
        loc[:, 0] = Tb.t
        loc[:, self.robot.n + 1] = Te.t

        # Joint axes position matrix
        joints = np.zeros((3, self.robot.n))

        # Axes arrow transforms
        Tjx = sm.SE3.Tx(0.06)
        Tjy = sm.SE3.Ty(0.06)
        Tjz = sm.SE3.Tz(0.06)

        # ee axes arrows
        Tex = Te * Tjx
        Tey = Te * Tjy
        Tez = Te * Tjz

        # Joint axes arrow calcs
        for i in range(self.robot.n):
            loc[:, i + 1] = T[i].t

            if isinstance(self.robot, rp.SerialLink) \
                    or self.robot.ets[self.robot.q_idx[i]].axis == 'Rz' \
                    or self.robot.ets[self.robot.q_idx[i]].axis == 'tz':
                Tji = T[i] * Tjz

            elif self.robot.ets[self.robot.q_idx[i]].axis == 'Ry' \
                    or self.robot.ets[self.robot.q_idx[i]].axis == 'ty':
                Tji = T[i] * Tjy

            elif self.robot.ets[self.robot.q_idx[i]].axis == 'Rx' \
                    or self.robot.ets[self.robot.q_idx[i]].axis == 'tx':
                Tji = T[i] * Tjx

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
                    loc[:, self.robot.n + 1], ee[0].t, '#EE9494', 2)
            self.ee_axes[1] = \
                self._plot_quiver(
                    loc[:, self.robot.n + 1], ee[1].t, '#93E7B0', 2)
            self.ee_axes[2] = \
                self._plot_quiver(
                    loc[:, self.robot.n + 1], ee[2].t, '#54AEFF', 2)

        # Remove oldjoint z coordinates
        if self.jointaxes:
            for i in range(self.robot.n):
                self.joints[i].remove()

            # Plot joint z coordinates
            for i in range(self.robot.n):
                self.joints[i] = \
                    self._plot_quiver(loc[:, i+1], joints[:, i], '#8FC1E2', 2)

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
                    loc[:, self.robot.n + 1], ee[0].t, '#EE9494', 2)
            self.ee_axes[1] = \
                self._plot_quiver2(
                    loc[:, self.robot.n + 1], ee[1].t, '#93E7B0', 2)

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
                    loc[:, self.robot.n + 1], ee[0].t, '#EE9494', 2))
            self.ee_axes.append(
                self._plot_quiver(
                    loc[:, self.robot.n + 1], ee[1].t, '#93E7B0', 2))
            self.ee_axes.append(
                self._plot_quiver(
                    loc[:, self.robot.n + 1], ee[2].t, '#54AEFF', 2))

        # Plot joint z coordinates
        if self.jointaxes:
            for i in range(self.robot.n):
                self.joints.append(
                    self._plot_quiver(loc[:, i+1], joints[:, i], '#8FC1E2', 2))

        # Plot the robot links
        self.links = self.ax.plot(
            loc[0, :], loc[1, :], loc[2, :], linewidth=5, color='#E16F6D')

        # Plot the shadow of the robot links
        if self.shadow:
            self.sh_links = self.ax.plot(
                loc[0, :], loc[1, :],
                linewidth=3, color='#464646')

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
                    loc[:, self.robot.n + 1], ee[0].t, '#EE9494', 2))
            self.ee_axes.append(
                self._plot_quiver2(
                    loc[:, self.robot.n + 1], ee[1].t, '#93E7B0', 2))

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
