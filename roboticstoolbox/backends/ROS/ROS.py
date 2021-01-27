#!/usr/bin/env python
"""
@author Jesse Haviland
"""

from roboticstoolbox.backends.Connector import Connector
import roboticstoolbox as rtb
import numpy as np
import spatialmath as sm

import time
import subprocess
import os
# import rospy
# from std_msgs.msg import Float32MultiArray

_ros = None
rospy = None


def _import_ros():     # pragma nocover
    import importlib
    global rospy
    try:
        ros = importlib.import_module('rospy')
    except ImportError:
        print(
            '\nYou must have ROS installed')
        raise


class ROS(Connector):  # pragma nocover
    """

    """
    def __init__(self):
        super(ROS, self).__init__()

        self.robots = []

    #
    #  Basic methods to do with the state of the external program
    #

    def launch(self, roscore=False, ros_master_uri=None, ros_ip=None):
        """
        """

        super().launch()

        # Launch roscore in seperate process
        if roscore:
            print('Launching ROS core\n')
            self.roscore = subprocess.Popen('roscore')

            # Give it some time to launch
            time.sleep(1)

        if ros_master_uri:
            os.environ["ROS_MASTER_URI"] = ros_master_uri

        if ros_ip:
            os.environ["ROS_IP"] = ros_ip

        v = VelPub(rtb.models.Panda())

    def step(self, dt=0.05):
        """
        """

        super().step

    def reset(self):
        """
        """

        super().reset

    def restart(self):
        """
        """

        super().restart

    def close(self):
        """
        """

        super().close()

    def hold(self):
        """
        """

        super().hold()

    #
    #  Methods to interface with the robots created in other environemnts
    #

    def add(
            self, ob):
        """
        """

        super().add()

    def remove(self):
        """
        Remove a robot to the graphical scene

        ``env.remove(robot)`` removes the ``robot`` from the graphical
            environment.
        """

        # TODO - shouldn't this have an id argument? which robot does it remove
        # TODO - it can remove any entity?

        super().remove()


class VelPub:

    def __init__(self, robot):
        rospy.init_node('RTB_VelocityPublisher')
        self.robot = robot
        self.v = np.zeros(robot.n)

        # self.velocity_publisher = rospy.Publisher(
        #     '/joint_velocity',
        #     Float32MultiArray, queue_size=1)

        self.relay()

    # def relay(self):

    #     while True:
    #         data = Float32MultiArray(data=self.robot.q)
    #         self.velocity_publisher.publish(data)
