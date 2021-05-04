#!/usr/bin/env python
"""
@author Jesse Haviland
"""

import numpy as np
import roboticstoolbox as rp
from spatialmath import SE3


class RobotPlot:
    def __init__(
        self,
        robot,
        env,
        readonly,
        display=True,
        jointaxes=True,
        jointlabels=False,
        eeframe=True,
        shadow=True,
        name=True,
        options=None,
    ):

        super(RobotPlot, self).__init__()

        # Readonly - True for this robot is for displaying only
        self.readonly = readonly

        # To show to robot in the plot or not
        # If not displayed, the robot is still simulated
        self.display = display

        self.robot = robot
        self.env = env
        self.ax = env.ax

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
        self.jointlabels = jointlabels
        self.shadow = shadow
        self.showname = name

        defaults = {
            "robot": {"color": "#E16F6D", "linewidth": 5},
            "shadow": {"color": "lightgrey", "linewidth": 3},
            "jointaxes": {"color": "#8FC1E2", "linewidth": 2},
            "jointlabels": {},
            "jointaxislength": 0.2,
            "eex": {"color": "#F84752", "linewidth": 2},  # '#EE9494'
            "eey": {"color": "#BADA55", "linewidth": 2},  # '#93E7B0'
            "eez": {"color": "#54AEFF", "linewidth": 2},
            "eelength": 0.06,
        }

        if options is not None:
            for key, value in options.items():
                defaults[key] = {**defaults[key], **options[key]}
        self.options = defaults

    def draw(self):
        if not self.display:
            return

        if not self.drawn:
            self.init()
            return

        ## Update the robot links

        # compute all link frames
        T = self.robot.fkine_all(self.robot.q)

        # draw all the line segments for the noodle plot
        for i, segment in enumerate(self.segments):
            linkframes = []
            for link in segment:
                if link is None:
                    linkframes.append(self.robot.base)
                else:
                    linkframes.append(T[link.number])
            points = np.array([linkframe.t for linkframe in linkframes])

            self.links[i].set_xdata(points[:, 0])
            self.links[i].set_ydata(points[:, 1])
            self.links[0].set_3d_properties(points[:, 2])

            # Update the shadow of the robot links
            if self.shadow:
                self.sh_links[i].set_xdata(points[:, 0])
                self.sh_links[i].set_ydata(points[:, 1])
                self.sh_links[i].set_3d_properties(0)

        ## Draw the end-effector coordinate frames

        # remove old ee coordinate frame
        if self.eeframes:
            for quiver in self.eeframes:
                quiver.remove()

            self.eeframes = []

        if self.eeframe:
            # Axes arrow transforms
            len = self.options["eelength"]
            Tjx = SE3([len, 0, 0])
            Tjy = SE3([0, len, 0])
            Tjz = SE3([0, 0, len])

            # add new ee coordinate frame
            for link in self.robot.ee_links:
                Te = T[link.number]

                # ee axes arrows
                Tex = Te * Tjx
                Tey = Te * Tjy
                Tez = Te * Tjz

                xaxis = self._plot_quiver(Te.t, Tex.t, self.options["eex"])
                yaxis = self._plot_quiver(Te.t, Tey.t, self.options["eey"])
                zaxis = self._plot_quiver(Te.t, Tez.t, self.options["eez"])

                self.eeframes.extend([xaxis, yaxis, zaxis])

        ## Joint axes

        # remove old joint z axes
        if self.joints:
            for joint in self.joints:
                joint.remove()

            # del self.joints
            self.joints = []

        # add new joint axes
        if self.jointaxes:
            # Plot joint z coordinates
            for link in self.robot:

                direction = None
                if isinstance(self.robot, rp.DHRobot):
                    # should test MDH I think
                    Tj = T[link.number - 1]
                    R = Tj.R
                    direction = R[:, 2]  # z direction
                elif link.isjoint:
                    Tj = T[link.number]
                    R = Tj.R
                    if link.v.axis[1] == "z":
                        direction = R[:, 2]  # z direction
                    elif link.v.axis[1] == "y":
                        direction = R[:, 1]  # y direction
                    elif link.v.axis[1] == "x":
                        direction = R[:, 0]  #  direction

                if direction is not None:
                    arrow = self._plot_quiver2(Tj.t, direction, link.jindex)
                    self.joints.extend(arrow)

    def init(self):

        self.drawn = True

        if self.env.limits is None:
            limits = np.r_[-1, 1, -1, 1, -1, 1] * self.robot.reach * 1.5
            self.ax.set_xlim3d([limits[0], limits[1]])
            self.ax.set_ylim3d([limits[2], limits[3]])
            self.ax.set_zlim3d([limits[4], limits[5]])

        self.segments = self.robot.segments()

        # Joint and ee poses
        Tb = self.robot.base
        # loc, joints, ee = self.axes_calcs()

        # Plot robot name
        if self.showname:
            self.name = self.ax.text(Tb.t[0], Tb.t[1], 0.05, self.robot.name)

        # Initialize the robot links
        self.links = []
        self.sh_links = []
        for i in range(len(self.segments)):

            # Plot the shadow of the robot links, draw first so robot is always
            # in front
            if self.shadow:
                (shadow,) = self.ax.plot([0], [0], zorder=1, **self.options["shadow"])
                self.sh_links.append(shadow)

            (line,) = self.ax.plot([0], [0], [0], **self.options["robot"])
            self.links.append(line)

        self.eeframes = []
        self.joints = []

    def _plot_quiver(self, p0, p1, options):
        qv = self.ax.quiver(
            p0[0], p0[1], p0[2], p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2], **options
        )
        return qv

    def _plot_quiver2(self, p0, dir, j):
        vec = dir * self.options["jointaxislength"]
        start = p0 - vec / 2
        qv = self.ax.quiver(
            start[0],
            start[1],
            start[2],
            vec[0],
            vec[1],
            vec[2],
            zorder=5,
            **self.options["jointaxes"],
        )

        if self.jointlabels:
            pl = p0 - vec * 0.6
            label = self.ax.text(
                pl[0], pl[1], pl[2], f"$q_{j}$", **self.options["jointlabels"]
            )
            return [qv, label]
        else:
            return [qv]
