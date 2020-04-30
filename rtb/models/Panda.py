#!/usr/bin/env python

import numpy as np
from rtb.robot.ets import ets, et
# from rtb.tools.transform import transl, xyzrpy_to_trans


class Panda(ets):
    """
    A class representing the Franka Emika Panda robot arm. ETS taken from [1]
    based on https://frankaemika.github.io/docs/control_parameters.html

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

        deg = np.pi/180
        mm = 1e-3
        d7 = (58.4)*mm

        et_list = [
            et(et.Ttz, 0.333),
            et(et.TRz, i=1),
            et(et.TRx, -90*deg),
            et(et.TRz, i=2),
            et(et.TRx, 90*deg),
            et(et.Ttz, 0.316),
            et(et.TRz, i=3),
            et(et.Ttx, 0.0825),
            et(et.TRx, 90*deg),
            et(et.TRz, i=4),
            et(et.Ttx, -0.0825),
            et(et.TRx, -90*deg),
            et(et.Ttz, 0.384),
            et(et.TRz, i=5),
            et(et.TRx, 90*deg),
            et(et.TRz, i=6),
            et(et.Ttx, 0.088),
            et(et.TRx, 90*deg),
            et(et.Ttz, 0.107),
            et(et.TRz, i=7),
        ]

        q_idx = [1, 3, 6, 9, 13, 15, 19]

        super(Panda, self).__init__(
            et_list,
            q_idx,
            name='Panda',
            manufacturer='Franka Emika',
            tool=np.eye(4))

        # tool = xyzrpy_to_trans(0, 0, d7, 0, 0, -np.pi/4)

        self._qz = np.array([0, 0, 0, 0, 0, 0, 0])
        self._qr = np.array([0, -90, -90, 90, 0, -90, 90]) * deg

    @property
    def qz(self):
        return self._qz

    @property
    def qr(self):
        return self._qr
