"""
xArmROSRobot module defines the xArmROSRobot type
xArmROSRobot provides robot-specific callbacks
.. codeauthor:: Gavin Suddreys
.. codeauthor:: Dasun Gunasinghe
"""

import rospy
import actionlib
import roboticstoolbox as rtb
import numpy as np

from armer.robots import ROSRobot

from std_srvs.srv import EmptyRequest, EmptyResponse
from std_srvs.srv import Trigger, TriggerRequest

from armer_msgs.msg import ManipulatorState

from armer_msgs.srv import SetCartesianImpedanceRequest, SetCartesianImpedanceResponse


class xArmROSRobot(ROSRobot):
    def __init__(
        self,
        robot: rtb.robot.Robot,
        controller_name: str = None,
        recover_on_estop: bool = True,
        *args,
        **kwargs,
    ):

        super().__init__(robot, *args, **kwargs)
        self.controller_name = (
            controller_name
            if controller_name
            else self.joint_velocity_topic.split("/")[1]
        )

        self.recover_on_estop = recover_on_estop
        self.last_estop_state = 0

        self.robot_state = None
        self.safety_state = None

        # Max 180 deg/s for each joint as per specifications
        # For use by ARMer NEO (in development)
        self.qdlim = np.array([3.14159, 3.14159, 3.14159, 3.14159, 3.14159, 3.14159])

        # TODO: add xArm specific services for recovery and state
        # self.xarm_state_sub = rospy.Subscriber('/rws/system_states', SystemState, self.xarm_state_cb)

        # Stop/Start-Up/Recovery Callback
        self.recover_cb(EmptyRequest())

    def recover_cb(self, req: EmptyRequest) -> EmptyResponse:  # pylint: disable=no-self-use
        """[summary]
        ROS Service callback:
        Invoke any available error recovery functions on the robot when an error occurs
        :param req: an empty request
        :type req: EmptyRequest
        :return: an empty response
        :rtype: EmptyResponse
        """
        print("Armer_xArm: Recovery Execution...")

        return EmptyResponse()

    def get_state(self):
        state = super().get_state()

        # if self.robot_state:
        #     state.errors |= ManipulatorState.LOCKED if not self.robot_state.motors_on else 0

        # if self.robot_state and self.robot_state.motors_on:
        #     if self.recover_on_estop and self.last_estop_state == 1:
        #         self.recover_cb(EmptyRequest())

        # self.last_estop_state = 1 if not self.robot_state.motors_on else 0

        return state

    def xarm_state_cb(self, msg):
        self.robot_state = msg
