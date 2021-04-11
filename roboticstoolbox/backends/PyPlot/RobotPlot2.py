#!/usr/bin/env python
"""
@author Jesse Haviland
"""

# import numpy as np
from roboticstoolbox.backends.PyPlot.RobotPlot import RobotPlot
import numpy as np
from spatialmath import SE2
class RobotPlot2(RobotPlot):

    def __init__(
            self, robot, ax, readonly, display=True,
            eeframe=True, name=True):

        super().__init__(
            robot, ax, readonly, display=display,
            jointaxes=False, shadow=False, eeframe=eeframe, name=name
        )

    def axes_calcs(self):
        # Joint and ee poses
        T = self.robot.fkine_path(self.robot.q)

        # Joint and ee position matrix
        # one column per frame

        loc = T.t.T

        # Joint axes position matrix
        joints = np.zeros((3, self.robot.n))

        # Axes arrow transforms


        # ee axes arrows
        Tex = T[-1] * Tjx
        Tey = T[-1] * Tjy

        return loc, joints, [Tex, Tey]

    def draw(self):
        if not self.display:
            return

        if not self.drawn:
            self.init()
            return

        # loc, joints, ee = self.axes_calcs()


        # Update the robot links

        # compute all link frames
        T = self.robot.fkine_path(self.robot.q)
        
        # draw all the line segments for the noodle plog
        for i, segment in enumerate(self.segments):
            linkframes = []
            for link in segment:
                if link is None:
                    linkframes.append(self.robot.base)
                else:
                    linkframes.append(T[link.number + 1])
            points = np.array([linkframe.t for linkframe in linkframes])

            self.links[i].set_xdata(points[:,0])
            self.links[i].set_ydata(points[:,1])

        # draw the end-effectors
        # Remove old ee coordinate frame
        if self.eeframes:
            for quiver in self.eeframes:
                quiver.remove()

            self.eeframes = []

        Tjx = SE2(0.06, 0)
        Tjy = SE2(0, 0.06)
        red = '#F84752'  # '#EE9494'
        green = '#BADA55'  # '#93E7B0'

        # Plot ee coordinate frame
        for link in self.robot.ee_links:
            Te = T[link.number + 1]

            # ee axes arrows
            Tex = Te * Tjx
            Tey = Te * Tjy

            xaxis = self._plot_quiver(Te.t, Tex.t, red, 2)
            yaxis = self._plot_quiver(Te.t, Tey.t, green, 2)

            self.eeframes.extend([xaxis, yaxis])



    def init(self):

        self.drawn = True

        self.segments = self.robot.segments()

        # Joint and ee poses
        Tb = self.robot.base
        # loc, joints, ee = self.axes_calcs()

        # Plot robot name
        if self.showname:
            self.name = self.ax.text(
                Tb.t[0] + 0.05, Tb.t[1], self.robot.name)

        # # Plot ee coordinate frame
        # if self.eeframe:
        #     self.ee_axes.append(
        #         self._plot_quiver(
        #             loc[:, -1], ee[0].t, '#EE9494', 2))
        #     self.ee_axes.append(
        #         self._plot_quiver(
        #             loc[:, -1], ee[1].t, '#93E7B0', 2))

        # Initialize the robot links
        self.links = []
        for i in range(len(self.segments)):
            line,  = self.ax.plot(
                # loc[0, :], loc[1, :], linewidth=5, color='#E16F6D')
                0, 0, linewidth=5, color='#778899')
            self.links.append(line)
        
        self.eeframes = []

    def _plot_quiver(self, p0, p1, col, width):
        # draw arrow from p0 (tail) to p1 (head)
        qv = self.ax.quiver(
            p0[0], p0[1],
            p1[0] - p0[0],
            p1[1] - p0[1],
            linewidth=width,
            color=col
        )

        return qv