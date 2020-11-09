#!/usr/bin/env python
"""
@author Micah Huth
"""

import importlib
from roboticstoolbox.backends.Connector import Connector
from roboticstoolbox.robot.DHLink import DHLink
from roboticstoolbox.robot.Robot import Robot as r

GraphicsCanvas3D = None
GraphicsCanvas2D = None
GraphicalRobot = None
close_localhost_session = None


def _imports():
    global GraphicsCanvas3D
    global GraphicsCanvas2D
    global GraphicalRobot
    global close_localhost_session

    try:
        canvas = importlib.import_module(
            'roboticstoolbox.backends.VPython.canvas')
        GraphicsCanvas3D = canvas.GraphicsCanvas3D
        GraphicsCanvas2D = canvas.GraphicsCanvas2D

        graphicalrobot = importlib.import_module(
            'roboticstoolbox.backends.VPython.graphicalrobot')
        GraphicalRobot = graphicalrobot.GraphicalRobot

        common_functions = importlib.import_module(
            'roboticstoolbox.backends.VPython.common_functions')
        close_localhost_session = common_functions.close_localhost_session

    except ImportError:
        print(
            '\nYou must install the VPython component of the toolbox, do: \n'
            'pip install roboticstoolbox[vpython]\n\n')


class VPython(Connector):  # pragma nocover
    """
    Graphical backend using VPython

    VPython is a Python API that connects to a JavaScript/WebGL 3D graphics
    engine in a browser tab.  It supports many 3D graphical primitives
    including meshes, boxes, ellipsoids and lines. It can not render in
    full color.

    Example:

    .. code-block:: python
        :linenos:

        import roboticstoolbox as rtb

        robot = rtb.models.DH.Panda()  # create a robot

        pyplot = rtb.backends.VPython() # create a VPython backend
        pyplot.add(robot)              # add the robot to the backend
        robot.q = robot.qz             # set the robot configuration
        pyplot.step()                  # update the backend and graphical view

    :references:

        - https://vpython.org

    """
    # TODO be able to add ellipsoids (vellipse, fellipse)
    # TODO be able add lines (for end-effector paths)

    def __init__(self):
        """
        Open a localhost session with no canvases

        """
        super(VPython, self).__init__()

        _imports()

        # Init vars
        self.canvases = []
        # 2D array of [is_3d, height, width, title, caption, grid] per canvas
        self.canvas_settings = []
        self.robots = []

        self._create_empty_session()

    def launch(
            self, is_3d=True, height=500, width=888,
            title='', caption='', grid=True):
        """
        Launch a graphical backend in a browser tab

        ``env = launch(args)` create a 3D scene in a new browser tab as
        defined by args, and returns a reference to the backend.

        """

        super().launch()

        self.canvas_settings.append(
            [is_3d, height, width, title, caption, grid])

        # Create the canvas with the given information
        if is_3d:
            self.canvases.append(
                GraphicsCanvas3D(height, width, title, caption, grid))
        else:
            self.canvases.append(
                GraphicsCanvas2D(height, width, title, caption, grid))

    def step(self, id, q=None, fig_num=0):
        """
        Update the graphical scene

        :param id: The Identification of the robot to remove. Can be either the
            DHRobot or GraphicalRobot
        :type id: :class:`~roboticstoolbox.robot.DHRobot.DHRobot`,
            :class:`roboticstoolbox.backends.VPython.graphics_robot.GraphicalRobot`
        :param q: The joint angles/configuration of the robot (Optional, if not
            supplied will use the stored q values).
        :type q: float ndarray(n)
        :param fig_num: The canvas index to delete the robot from, defaults to
            the initial one
        :type fig_num: int, optional
        :raises ValueError: Figure number must be between 0 and total number of
            canvases
        :raises TypeError: Input must be a DHLink or GraphicalRobot

        ``env.step(args)`` triggers an update of the 3D scene in the browser
        window referenced by ``env``.

        .. note::

            - Each robot in the scene is updated based on
              their control type (position, velocity, acceleration, or torque).
            - Upon acting, the other three of the four control types will be
              updated in the internal state of the robot object.
            - The control type is defined by the robot object, and not all
              robot objects support all control types.
            - Execution is blocked for the specified interval

        """

        super().step()

        if fig_num < 0 or fig_num >= len(self.canvases):
            raise ValueError(
                "Figure number must be between 0 and total number of canvases")

        # If DHRobot given
        if isinstance(id, r.Robot):
            robot = None
            # Find first occurrence of it that is in the correct canvas
            for i in range(len(self.robots)):
                if self.robots[i].robot is id and \
                        self.canvases[fig_num].is_robot_in_canvas(
                                                        self.robots[i]):
                    robot = self.robots[i]
                    break
            if robot is None:
                print("No robot")
                return
            else:
                poses = robot.fkine(q)
                robot.set_joint_poses(poses)
        # ElseIf GraphicalRobot given
        elif isinstance(id, GraphicalRobot):
            if self.canvases[fig_num].is_robot_in(id):
                poses = id.fkine(q)
                id.set_joint_poses(poses)
        # Else
        else:
            raise TypeError(
                "Input must be a Robot (or subclass) or "
                "GraphicalRobot, given {0}".format(type(id)))

    def reset(self):
        """
        Reset the graphical scene

        ``env.reset()`` triggers a reset of the 3D scene in the browser window
        referenced by ``env``. It is restored to the original state defined by
        ``launch()``.

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
            for settings in self.canvas_settings:
                # Create the canvas with the given information
                if settings[0]:
                    self.canvases.append(GraphicsCanvas3D(
                        settings[1], settings[2], settings[3],
                        settings[4], settings[5]))
                else:
                    self.canvases.append(GraphicsCanvas2D(
                        settings[1], settings[2], settings[3],
                        settings[4], settings[5]))

    def restart(self):
        """
        Restart the graphics display

        ``env.restart()`` triggers a restart of the browser view referenced by
        ``env``. It is closed and relaunched to the original state defined by
        ``launch()``.

        """

        super().restart()

        self.reset()

    def close(self):
        """
        Close the graphics display

        ``env.close()`` gracefully closes the browser tab browser view
        referenced by ``env``.

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

    def add(self, fig_num, name, dhrobot):
        """
        Add a robot to the graphical scene

        :param fig_num: The canvas number to place the robot in
        :type fig_num: int
        :param name: The name of the robot
        :type name: `str`
        :param dhrobot: The ``DHRobot`` object (if applicable)
        :type dhrobot: class:`~roboticstoolbox.robot.DHRobot.DHRobot`, None
        :raises ValueError: Figure number must be between 0 and number of
            figures created
        :return: object id within visualizer
        :rtype: int

        ``id = env.add(robot)`` adds the ``robot`` to the graphical
            environment.

        .. note::

            - ``robot`` must be of an appropriate class.
            - Adds the robot object to a list of robots which will be updated
              when the ``step()`` method is called.

        """

        # TODO - name can come from the robot object, maybe an override name?
        # TODO - why dhrobot "if applicable"?
        # TODO - what about other classes of robot?
        # TODO - what about adding ellipsoids?

        super().add()

        # Sanity check input
        if fig_num < 0 or fig_num > len(self.canvases) - 1:
            raise ValueError(
                "Figure number must be between 0 and number "
                "of figures created")

        # Add robot to canvas
        self.robots.append(
            GraphicalRobot(self.canvases[fig_num], name, dhrobot))
        # self.canvases[fig_num].add_robot(self.robots[len(self.robots)-1])

    def remove(self, id, fig_num=0):
        """
        Remove a robot to the graphical scene

        :param id: The id of the robot to remove. Can be either the DHLink or
            GraphicalRobot
        :type id: class:`~roboticstoolbox.robot.DHRobot.DHRobot`,
                  class:`roboticstoolbox.backends.VPython.graphics_robot.GraphicalRobot`
        :param fig_num: The canvas index to delete the robot from, defaults to
             the initial one
        :type fig_num: int, optional
        :raises ValueError: Figure number must be between 0 and total number
            of canvases
        :raises TypeError: Input must be a DHLink or GraphicalRobot

        ``env.remove(robot)`` removes the ``robot`` from the graphical
            environment.

        """

        super().remove()

        if fig_num < 0 or fig_num >= len(self.canvases):
            raise ValueError(
                "Figure number must be between 0 and total number of canvases")

        # If DHLink given
        if isinstance(id, DHLink):
            robot = None
            # Find first occurrence of it that is in the correct canvas
            for i in range(len(self.robots)):
                if self.robots[i].seriallink.equal(id) and \
                        self.canvases[fig_num].is_robot_in(self.robots[i]):
                    robot = self.robots[i]
                    break
            if robot is None:
                return
            else:
                self.canvases[fig_num].delete_robot(robot)
        # ElseIf GraphicalRobot given
        elif isinstance(id, GraphicalRobot):
            if self.canvases[fig_num].is_robot_in(id):
                self.canvases[fig_num].delete_robot(id)
        # Else
        else:
            raise TypeError("Input must be a DHLink or GraphicalRobot")

    #
    #  Private Methods
    #

    @staticmethod
    def _create_empty_session():
        """
        Create a canvas to ensure the localhost session has been opened.
        Then clear the browser tab
        """
        # Create a canvas to initiate the connection
        temp = GraphicsCanvas2D()

        # Delete the canvas to leave a blank screen
        temp.scene.append_to_caption('''
            <script type="text/javascript">
                let gs = document.getElementById('glowscript');
                gs.innerHTML = '';
            </script>
        ''')
