#!/usr/bin/env python
"""
@author Micah Huth
"""

from roboticstoolbox.backend.Connector import Connector

from vpython import canvas, sphere
from roboticstoolbox.backend.VPython.graphics_canvas import GraphicsCanvas3D, GraphicsCanvas2D


class VPython(Connector):

    def __init__(self):
        """
        Open a localhost session with no canvases

        """
        super(VPython, self).__init__()

        # Init vars
        self.canvas = None

        # Create a canvas to initiate the connection
        temp = canvas()
        sphere(scene=temp)

        # Delete the canvas to leave a blank screen
        temp.scene.append_to_caption('''
            <script type="text/javascript">
                let gs = document.getElementById('glowscript');
                gs.innerHTML = '';
            </script>
        ''')

    def launch(self, is_3d=True, height=500, width=888, title='', caption='', grid=True):
        """
        env = launch(args) launch the external program with an empty or
        specific scene as defined by args

        """

        super().launch()

        # Create the canvas with the given information
        if is_3d:
            self.canvas = GraphicsCanvas3D(height, width, title, caption, grid)
        else:
            self.canvas = GraphicsCanvas2D(height, width, title, caption, grid)

    def step(self):
        """
        state = step(args) triggers the external program to make a time step
        of defined time updating the state of the environment as defined by
        the robot's actions.

        The will go through each robot in the list and make them act based on
        their control type (position, velocity, acceleration, or torque). Upon
        acting, the other three of the four control types will be updated in
        the internal state of the robot object. The control type is defined
        by the robot object, and not all robot objects support all control
        types.

        """

        super().step()

        # Update positions to new frame

    def reset(self):
        """
        state = reset() triggers the external program to reset to the
        original state defined by launch

        """

        super().reset()

    def restart(self):
        """
        state = restart() triggers the external program to close and relaunch
        to thestate defined by launch

        """

        super().restart()

        # Close session

        # Load new session

    def close(self):
        """
        state = close() triggers the external program to gracefully close

        """

        super().close()

        # Close session

    def add(self):
        """
        id = add(robot) adds the robot to the external environment. robot must
        be of an appropriate class. This adds a robot object to a list of
        robots which will act upon the step() method being called.

        """

        super().add()

        # Add robot to canvas

    def remove(self):
        """
        id = remove(robot) removes the robot to the external environment.

        """

        super().remove()

        # Remove robot from canvas
