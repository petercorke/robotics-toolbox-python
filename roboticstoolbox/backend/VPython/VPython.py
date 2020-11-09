#!/usr/bin/env python
"""
@author Micah Huth
"""

import threading
from time import perf_counter, sleep
from PIL import Image
import os
import platform
import glob

from roboticstoolbox.backends.Connector import Connector

from roboticstoolbox.robot.DHLink import DHLink
from roboticstoolbox.robot.Robot import Robot as r

from roboticstoolbox.backends.VPython.canvas import GraphicsCanvas3D, \
    GraphicsCanvas2D
from roboticstoolbox.backends.VPython.graphicalrobot import \
    GraphicalRobot
from roboticstoolbox.backends.VPython.common_functions import \
    close_localhost_session


class VPython(Connector):

    def __init__(self):
        """
        Open a localhost session with no canvases

        """
        super(VPython, self).__init__()

        # Init vars
        self.canvases = []
        self.canvas_settings = []  # 2D array of [is_3d, height, width, title, caption, grid] per canvas
        self.robots = []
        self._recording = False
        self._recording_thread = None
        self._recording_fps = None
        self._thread_lock = threading.Lock()

        self._create_empty_session()

    def launch(self, is_3d=True, height=500, width=888, title='', caption='', grid=True, g_col=None, g_opc=1):
        """
        env = launch(args) launch the external program with an empty or
        specific scene as defined by args

        """

        super().launch()

        self.canvas_settings.append([is_3d, height, width, title, caption, grid, g_col, g_opc])

        # Create the canvas with the given information
        if is_3d:
            self.canvases.append(GraphicsCanvas3D(height, width, title, caption, grid, g_col, g_opc))
        else:
            self.canvases.append(GraphicsCanvas2D(height, width, title, caption, grid, g_col, g_opc))

    def step(self, id, q=None, fig_num=0):
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

        :param id: The Identification of the robot to remove. Can be either the DHRobot or GraphicalRobot
        :type id: class:`roboticstoolbox.robot.DHRobot.DHRobot`,
                  class:`roboticstoolbox.backend.VPython.graphics_robot.GraphicalRobot`
        :param q: The joint angles/configuration of the robot (Optional, if not supplied will use the stored q values).
        :type q: float ndarray(n)
        :param fig_num: The canvas index to delete the robot from, defaults to the initial one
        :type fig_num: `int`, optional
        :raises ValueError: Figure number must be between 0 and total number of canvases
        :raises TypeError: Input must be a DHLink or GraphicalRobot

        """

        super().step()

        if fig_num < 0 or fig_num >= len(self.canvases):
            raise ValueError("Figure number must be between 0 and total number of canvases")

        # If DHRobot given
        if isinstance(id, r):
            robot = None
            # Find first occurrence of it that is in the correct canvas
            for i in range(len(self.robots)):
                if self.robots[i].robot is id and self.canvases[fig_num].is_robot_in_canvas(self.robots[i]):
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
            raise TypeError("Input must be a Robot (or subclass) or GraphicalRobot, given {0}".format(type(id)))

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
            for settings in self.canvas_settings:
                # Create the canvas with the given information
                if settings[0]:
                    self.canvases.append(GraphicsCanvas3D(settings[1], settings[2], settings[3], settings[4], settings[5]))
                else:
                    self.canvases.append(GraphicsCanvas2D(settings[1], settings[2], settings[3], settings[4], settings[5]))

    def restart(self):
        """
        state = restart() triggers the external program to close and relaunch
        to the state defined by launch

        """

        super().restart()

        # self.close()
        # self._create_empty_session()
        # for settings in self.canvas_settings:
        #     # Create the canvas with the given information
        #     if settings[0]:
        #         self.canvases.append(GraphicsCanvas3D(settings[1], settings[2], settings[3], settings[4], settings[5]))
        #     else:
        #         self.canvases.append(GraphicsCanvas2D(settings[1], settings[2], settings[3], settings[4], settings[5]))

        # Program on close terminates execution, so just run reset
        self.reset()

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

    def add(self, fig_num, name, dhrobot):
        """
        id = add(robot) adds the robot to the external environment. robot must
        be of an appropriate class. This adds a robot object to a list of
        robots which will act upon the step() method being called.

        :param fig_num: The canvas number to place the robot in
        :type fig_num: `int`
        :param name: The name of the robot
        :type name: `str`
        :param dhrobot: The DHRobot object (if applicable)
        :type dhrobot: class:`roboticstoolbox.robot.DHRobot.DHRobot`, None
        :raises ValueError: Figure number must be between 0 and number of figures created

        """

        super().add()

        # Sanity check input
        if fig_num < 0 or fig_num > len(self.canvases) - 1:
            raise ValueError("Figure number must be between 0 and number of figures created")

        # Add robot to canvas
        self.robots.append(GraphicalRobot(self.canvases[fig_num], name, dhrobot))
        # self.canvases[fig_num].add_robot(self.robots[len(self.robots)-1])

    def remove(self, id, fig_num=0):
        """
        id = remove(robot) removes the robot to the external environment.

        :param id: The Identification of the robot to remove. Can be either the DHLink or GraphicalRobot
        :type id: class:`roboticstoolbox.robot.DHRobot.DHRobot`,
                  class:`roboticstoolbox.backend.VPython.graphics_robot.GraphicalRobot`
        :param fig_num: The canvas index to delete the robot from, defaults to the initial one
        :type fig_num: `int`, optional
        :raises ValueError: Figure number must be between 0 and total number of canvases
        :raises TypeError: Input must be a DHLink or GraphicalRobot
        """

        super().remove()

        if fig_num < 0 or fig_num >= len(self.canvases):
            raise ValueError("Figure number must be between 0 and total number of canvases")

        # If DHLink given
        if isinstance(id, DHLink):
            robot = None
            # Find first occurrence of it that is in the correct canvas
            for i in range(len(self.robots)):
                if self.robots[i].seriallink.equal(id) and self.canvases[fig_num].is_robot_in(self.robots[i]):
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
    # Public non-standard methods
    #
    def record_start(self, fps, scene_num=0):
        """
        Start recording screencaps of a scene
        """
        self._thread_lock.acquire()

        if not self._recording:
            print("VPython Recording...")
            self._recording = True
            self._recording_fps = fps
            # Spawn a thread
            self._recording_thread = threading.Thread(target=self._record_scene, args=(scene_num, fps,))
            self._recording_thread.start()

        self._thread_lock.release()

    def record_stop(self, filename):
        """
        Stop recording screencaps of a scene and combine them into a movie
        """
        #
        self._thread_lock.acquire()
        if self._recording:
            self._recording = False
            print("VPython Recording Stopped...")
            print("VPython Recording Saving... DO NOT EXIT")
        else:
            self._thread_lock.release()
            return
        self._thread_lock.release()

        # Wait for thread to finish
        self._recording_thread.join()

        # Wait a bit longer to ensure downloads complete
        # (Small file sizes so should be quick, but better safe than sorry)
        sleep(3)

        # Get downloads directory
        opsys = platform.system()
        if opsys == 'Windows':  # Windows
            path_in = os.path.join(os.getenv('USERPROFILE'), 'downloads')

        elif opsys == 'Linux' or opsys == 'Darwin':  # Linux / Mac
            path_in = os.path.join(os.getenv('HOME'), 'downloads')

        else:  # Undefined OS
            # lets assume 'HOME' for now
            path_in = os.path.join(os.getenv('HOME'), 'downloads')

        fp_out = filename
        fp_in = path_in + "/vpython_*.png"

        file_format = filename[-4:]
        if file_format[0] == '.':
            file_format=file_format[-3:]

        img, *imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]
        img.save(fp=fp_out, format=file_format, append_images=imgs, save_all=True)

        print("VPython Recording Saved... It is safe to exit")


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

    def _record_scene(self, scene_num, fps):
        """
        Thread-called function to continuously record screenshots
        """
        frame_num = 0
        if fps <= 0:
            raise ValueError("fps must be greater than 0.")
        f = 1 / fps

        self._thread_lock.acquire()
        recording = self._recording
        self._thread_lock.release()

        while recording:
            # Get current time
            t_start = perf_counter()

            # Take screenshot
            filename = "vpython_{:04d}.png".format(frame_num)
            self.canvases[scene_num].take_screenshot(filename)
            frame_num += 1

            # Get current time
            t_stop = perf_counter()

            # Wait for time of frame to finish
            # If saving takes longer than frame frequency, this while is skipped
            while t_stop - t_start < f:
                t_stop = perf_counter()

            self._thread_lock.acquire()
            recording = self._recording
            self._thread_lock.release()
