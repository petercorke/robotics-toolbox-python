#!/usr/bin/env python
"""
@author Micah Huth
"""

from roboticstoolbox.backend.Connector import Connector

from roboticstoolbox.backend.VPython.graphics_canvas import GraphicsCanvas3D, GraphicsCanvas2D
from roboticstoolbox.backend.VPython.common_functions import close_localhost_session


class VPython(Connector):

    def __init__(self):
        """
        Open a localhost session with no canvases

        """
        super(VPython, self).__init__()

        # Init vars
        self.canvases = []
        self.canvas_settings = []

        self._create_empty_session()

    def launch(self, is_3d=True, height=500, width=888, title='', caption='', grid=True):
        """
        env = launch(args) launch the external program with an empty or
        specific scene as defined by args

        """

        super().launch()

        self.canvas_settings = [is_3d, height, width, title, caption, grid]

        # Create the canvas with the given information
        if is_3d:
            self.canvases.append(GraphicsCanvas3D(height, width, title, caption, grid))
        else:
            self.canvases.append(GraphicsCanvas2D(height, width, title, caption, grid))

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

        if len(self.canvases) > 0:
            # Clear localhost
            self.canvases[0].scene.append_to_caption('''
                <script type="text/javascript">
                    let gs = document.getElementById('glowscript');
                    gs.innerHTML = '';
                </script>
                ''')

            # Delete all sessions
            self.canvases = []

            self._create_empty_session()
            self.launch(
                self.canvas_settings[0],
                self.canvas_settings[1],
                self.canvas_settings[2],
                self.canvas_settings[3],
                self.canvas_settings[4],
                self.canvas_settings[5],
            )

    def restart(self):
        """
        state = restart() triggers the external program to close and relaunch
        to the state defined by launch

        """

        super().restart()

        self.close()
        self._create_empty_session()
        self.launch(
            self.canvas_settings[0],
            self.canvas_settings[1],
            self.canvas_settings[2],
            self.canvas_settings[3],
            self.canvas_settings[4],
            self.canvas_settings[5],
        )

    def close(self):
        """
        state = close() triggers the external program to gracefully close

        """

        super().close()

        # Close session
        if len(self.canvases) > 0:
            # if a canvas made
            close_localhost_session(self.canvases[0])
        else:
            # No canvas, so make one
            temp = GraphicsCanvas2D()
            close_localhost_session(temp)

        self.canvases = []

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

    def _create_empty_session(self):
        # Create a canvas to initiate the connection
        temp = GraphicsCanvas2D()

        # Delete the canvas to leave a blank screen
        temp.scene.append_to_caption('''
            <script type="text/javascript">
                let gs = document.getElementById('glowscript');
                gs.innerHTML = '';
            </script>
        ''')
