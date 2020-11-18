#!/usr/bin/env python
"""
@author Jesse Haviland
"""

from roboticstoolbox.backends.Connector import Connector
import roboticstoolbox as rp
import numpy as np
import spatialmath as sm
import time
from queue import Queue

_sw = None
sw = None


def _import_swift():     # pragma nocover
    import importlib
    global sw
    try:
        sw = importlib.import_module('swift')
        # from swift import start_servers
    except ImportError:
        print(
            '\nYou must install the python package swift, see '
            'https://github.com/jhavl/swift\n')
        raise


class Swift(Connector):  # pragma nocover
    """
    Graphical backend using Swift

    Swift is a web app built on three.js. It supports many 3D graphical
    primitives including meshes, boxes, ellipsoids and lines. It can render
    Collada objects in full color.

    :param realtime: Force the simulator to display no faster than real time,
        note that it may still run slower due to complexity
    :type realtime: bool
    :param display: Do not launch the graphical front-end of the simulator.
        Will still simulate the robot. Runs faster due to not needing to
        display anything.
    :type display: bool

    Example:

    .. code-block:: python
        :linenos:

        import roboticstoolbox as rtb

        robot = rtb.models.DH.Panda()  # create a robot

        pyplot = rtb.backends.Swift()   # create a Swift backend
        pyplot.add(robot)              # add the robot to the backend
        robot.q = robot.qz             # set the robot configuration
        pyplot.step()                  # update the backend and graphical view

    :references:

        - https://github.com/jhavl/swift

    """
    def __init__(self, realtime=True, display=True):
        super(Swift, self).__init__()

        self.sim_time = 0.0
        self.robots = []
        self.shapes = []
        self.outq = Queue()
        self.inq = Queue()

        self.realtime = realtime
        self.display = display

        self.recording = False

        if self.display and sw is None:
            _import_swift()

    #
    #  Basic methods to do with the state of the external program
    #

    def launch(self, browser=None):
        """
        Launch a graphical backend in Swift by default in the default browser
        or in the specified browser

        :param browser: browser to open in: one of
            'google-chrome', 'chrome', 'firefox', 'safari', 'opera'
            or see for full list
            https://docs.python.org/3.8/library/webbrowser.html#webbrowser.open_new
        :type browser: string

        ``env = launch(args)`` create a 3D scene in a running Swift instance as
        defined by args, and returns a reference to the backend.

        """

        super().launch()

        if self.display:
            sw.start_servers(self.outq, self.inq, browser=browser)
            self.last_time = time.time()

    def step(self, dt=0.05):
        """
        Update the graphical scene

        :param dt: time step in seconds, defaults to 0.05
        :type dt: int, optional

        ``env.step(args)`` triggers an update of the 3D scene in the Swift
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

        # TODO how is the pose of shapes updated prior to step?

        super().step

        self._step_robots(dt)
        self._step_shapes(dt)

        # Adjust sim time
        self.sim_time += dt

        # Send updated sim time to Swift
        if self.display:

            # If realtime is set, delay progress if we are running too quickly
            if self.realtime:
                time_taken = (time.time() - self.last_time)
                diff = dt - time_taken

                if diff > 0:
                    time.sleep(diff)

                self.last_time = time.time()

            self._draw_all()
            self._send_socket('sim_time', self.sim_time)

    def reset(self):
        """
        Reset the graphical scene

        ``env.reset()`` triggers a reset of the 3D scene in the Swift window
        referenced by ``env``. It is restored to the original state defined by
        ``launch()``.

        """

        super().reset

    def restart(self):
        """
        Restart the graphics display

        ``env.restart()`` triggers a restart of the Swift view referenced by
        ``env``. It is closed and relaunched to the original state defined by
        ``launch()``.

        """

        super().restart

    def close(self):
        """
        Close the graphics display

        ``env.close()`` gracefully disconnectes from the Swift visualizer
        referenced by ``env``.
        """

        super().close()

    #
    #  Methods to interface with the robots created in other environemnts
    #

    def add(
            self, ob, show_robot=True, show_collision=False,
            readonly=False):
        """
        Add a robot to the graphical scene

        :param ob: the object to add
        :type ob: Robot or Shape
        :param show_robot: Show the robot visual geometry,
            defaults to True
        :type show_robot: bool, optional
        :param show_collision: Show the collision geometry,
            defaults to False
        :type show_collision: bool, optional
        :return: object id within visualizer
        :rtype: int
        :param readonly: If true, swif twill not modify any robot attributes,
            the robot is only nbeing displayed, not simulated,
            defaults to False
        :type readonly: bool, optional

        ``id = env.add(robot)`` adds the ``robot`` to the graphical
            environment.

        .. note::

            - Adds the robot object to a list of robots which will be updated
              when the ``step()`` method is called.

        """
        # id = add(robot) adds the robot to the external environment. robot
        # must be of an appropriate class. This adds a robot object to a
        # list of robots which will act upon the step() method being called.

        # TODO can add more than a robot right?

        super().add()

        if isinstance(ob, rp.ERobot):
            robot = ob.to_dict()
            robot['show_robot'] = show_robot
            robot['show_collision'] = show_collision

            robot_object = {
                'ob': ob,
                'readonly': readonly
            }

            if self.display:
                id = self._send_socket('robot', robot)

                loaded = 0
                while loaded == 0:
                    loaded = int(self._send_socket('is_loaded', id))
                    time.sleep(0.1)
            else:
                id = len(self.robots)

            self.robots.append(robot_object)
            return id
        elif isinstance(ob, rp.Shape):
            shape = ob.to_dict()
            if self.display:
                id = self._send_socket('shape', shape)
            else:
                id = len(self.shapes)
            self.shapes.append(ob)
            return id

    def remove(self):
        """
        Remove a robot to the graphical scene

        ``env.remove(robot)`` removes the ``robot`` from the graphical
            environment.
        """

        # TODO - shouldn't this have an id argument? which robot does it remove
        # TODO - it can remove any entity?

        super().remove()

    def hold(self):           # pragma: no cover
        '''
        hold() keeps the browser tab open i.e. stops the browser tab from
        closing once the main script has finished.

        '''

        while True:
            time.sleep(1)

    def start_recording(self, file_name, framerate, format='webm'):
        """
        Start recording the canvas in the Swift simulator

        :param file_name: The file name for which the video will be saved as
        :type file_name: string
        :param framerate: The framerate of the video - to be timed correctly,
            this should equalt 1 / dt where dt is the time supplied to the
            step function
        :type framerate: float
        :param format: This is the format of the video, one of 'webm', 'gif',
            'png', or 'jpg'
        :type format: string

        ``env.start_recording(file_name)`` starts recording the simulation
            scene and will save it as file_name once
            ``env.start_recording(file_name)`` is called
        """

        valid_formats = ['webm', 'gif', 'png', 'jpg']

        if format not in valid_formats:
            raise ValueError(
                "Format can one of 'webm', 'gif', 'png', or 'jpg'")

        if not self.recording:
            self._send_socket(
                'start_recording', [framerate, file_name, format])
            self.recording = True
        else:
            raise ValueError(
                "You are already recording, you can only record one video"
                " at a time")

    def stop_recording(self):
        """
        Start recording the canvas in the Swift simulator. This is optional
        as the video will be automatically saved when the python script exits

        ``env.stop_recording()`` stops the recording of the simulation, can
            only be called after ``env.start_recording(file_name)``
        """

        if self.recording:
            self._send_socket('stop_recording')
        else:
            raise ValueError(
                "You must call swift.start_recording(file_name) before trying"
                " to stop the recording")

    def _step_robots(self, dt):

        for robot_object in self.robots:
            robot = robot_object['ob']

            if robot_object['readonly'] or robot.control_type == 'p':
                pass            # pragma: no cover

            elif robot.control_type == 'v':

                for i in range(robot.n):
                    robot.q[i] += robot.qd[i] * (dt)

                    if np.any(robot.qlim[:, i] != 0) and \
                            not np.any(np.isnan(robot.qlim[:, i])):
                        robot.q[i] = np.min([robot.q[i], robot.qlim[1, i]])
                        robot.q[i] = np.max([robot.q[i], robot.qlim[0, i]])

            elif robot.control_type == 'a':
                pass

            else:            # pragma: no cover
                # Should be impossible to reach
                raise ValueError(
                    'Invalid robot.control_type. '
                    'Must be one of \'p\', \'v\', or \'a\'')

    def _step_shapes(self, dt):

        for shape in self.shapes:

            T = shape.base
            t = T.t
            r = T.rpy('rad')

            t += shape.v[:3] * (dt)
            r += shape.v[3:] * (dt)

            shape.base = sm.SE3(t) * sm.SE3.RPY(r)

    def _draw_all(self):

        for i in range(len(self.robots)):
            self._send_socket(
                'robot_poses', [i, self.robots[i]['ob'].fk_dict()])

        for i in range(len(self.shapes)):
            self._send_socket(
                'shape_poses', [i, self.shapes[i].fk_dict()])

    def _send_socket(self, code, data=None):
        msg = [code, data]

        self.outq.put(msg)
        return self.inq.get()
