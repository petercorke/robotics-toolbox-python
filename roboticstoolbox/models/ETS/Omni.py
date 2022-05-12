#!/usr/bin/env python

import numpy as np
from roboticstoolbox.robot.ET import ET
from roboticstoolbox.robot.ETS import ETS
from roboticstoolbox.robot.ERobot import ERobot
from roboticstoolbox.robot.Link import Link
import spatialgeometry as sg
import spatialmath as sm


class Omni(ERobot):
    """
    A class an omnidirectional mobile robot.

    :param et_list: List of elementary transforms which represent the robot
        kinematics
    :type et_list: list of etb.robot.et
    :param q_idx: List of indexes within the ets_list which correspond to
        joints
    :type q_idx: list of int
    :param name: Name of the robot
    :type name: str, optional
    :param manufacturer: Manufacturer of the robot
    :type manufacturer: str, optional
    :param base: Location of the base is the world frame
    :type base: float np.ndarray(4,4), optional
    :param tool: Offset of the flange of the robot to the end-effector
    :type tool: float np.ndarray(4,4), optional
    :param qz: The zero joint angle configuration of the robot
    :type qz: float np.ndarray(7,)
    :param qr: The ready state joint angle configuration of the robot
    :type qr: float np.ndarray(7,)

    References: [1] Kinematic Derivatives using the Elementary Transform
        Sequence, J. Haviland and P. Corke
    """

    def __init__(self):

        l, w, h = 0.55, 0.40, 0.35

        b0 = Link(ETS(ET.Rz()), name="base0", parent=None, qlim=[-1000, 1000])

        b1 = Link(ETS(ET.tx()), name="base1", parent=b0, qlim=[-1000, 1000])

        b2 = Link(ETS(ET.ty()), name="base2", parent=b1, qlim=[-1000, 1000])

        g0 = Link(name="gripper", parent=b2)

        b2.geometry = sg.Cuboid(
            [l, w, h], base=sm.SE3(0, 0, h / 2), color=(163, 157, 134)
        )

        elinks = [b0, b1, b2, g0]

        super(Omni, self).__init__(
            elinks,
            name="Omni",
            manufacturer="Jesse",
            keywords=("mobile",),
            gripper_links=g0,
        )

        self.qr = np.array([0, 0, 0])
        self.qz = np.zeros(3)

        self.addconfiguration("qr", self.qr)
        self.addconfiguration("qz", self.qz)


if __name__ == "__main__":  # pragma nocover

    robot = Omni()
    print(robot)
