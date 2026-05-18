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
            self, robot, env, readonly, display=True,
            eeframe=True, name=True, options=None):

        super().__init__(
            robot, env, readonly, display=display,
            jointaxes=False, shadow=False, eeframe=eeframe, name=name
        )

        defaults = {
            'robot': {'color': '#E16F6D', 'linewidth': 5},
            'jointlabels': {},
            'eex': {'color': '#F84752', 'linewidth': 2}, # '#EE9494'
            'eey': {'color': '#BADA55', 'linewidth': 2}, # '#93E7B0'
            'eelength': 0.06,
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

            self.links[i].set_xdata(points[:,0])
            self.links[i].set_ydata(points[:,1])

        ## Draw the end-effectors

        # Remove old ee coordinate frame
        if self.eeframes:
            for quiver in self.eeframes:
                quiver.remove()

            self.eeframes = []

        if self.eeframe:
            len = self.options['eelength']
            Tjx = SE2(len, 0)
            Tjy = SE2(0, len)


            # add new ee coordinate frame
            for link in self.robot.ee_links:
                Te = T[link.number]

                # ee axes arrows
                Tex = Te * Tjx
                Tey = Te * Tjy

                xaxis = self._plot_quiver(Te.t, Tex.t, self.options['eex'])
                yaxis = self._plot_quiver(Te.t, Tey.t, self.options['eey'])

                self.eeframes.extend([xaxis, yaxis])

    def init(self):

        self.drawn = True

        limits = np.r_[-1, 1, -1, 1] * self.robot.reach * 1.5
        self.ax.set_xlim([limits[0], limits[1]])
        self.ax.set_ylim([limits[2], limits[3]])

        self.segments = self.robot.segments()

        # Joint and ee poses
        Tb = self.robot.base
        # loc, joints, ee = self.axes_calcs()

        # Plot robot name
        if self.showname:
            self.name = self.ax.text(
                Tb.t[0] + 0.05, Tb.t[1], self.robot.name)

        # Initialize the robot links
        self.links = []
        for i in range(len(self.segments)):
            line,  = self.ax.plot(
                0, 0, **self.options['robot'])
            self.links.append(line)
        
        self.eeframes = []

    def _plot_quiver(self, p0, p1, options):
        # draw arrow from p0 (tail) to p1 (head)
        qv = self.ax.quiver(
            p0[0], p0[1],
            p1[0] - p0[0],
            p1[1] - p0[1],
            **options
        )
        return qv