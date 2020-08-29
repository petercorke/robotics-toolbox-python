#!/usr/bin/env python

import numpy as np
from ropy.robot.ETS import ETS
from ropy.robot.ET import ET
from ropy.robot.ELink import ELink


class Frankie(ETS):
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
        tool_offset = (103)*mm

        b0 = ELink(
            [ET.TRz()],
            name='base0',
            parent=None
        )

        b1 = ELink(
            [ET.Ttx()],
            name='base1',
            parent=b0
        )

        l0 = ELink(
            [ET.Ttz(0.333), ET.TRz()],
            name='link0',
            parent=b1
        )

        l1 = ELink(
            [ET.TRx(-90*deg), ET.TRz()],
            name='link1',
            parent=l0
        )

        l2 = ELink(
            [ET.TRx(90*deg), ET.Ttz(0.316), ET.TRz()],
            name='link2',
            parent=l1
        )

        l3 = ELink(
            [ET.Ttx(0.0825), ET.TRx(90*deg), ET.TRz()],
            name='link3',
            parent=l2
        )

        l4 = ELink(
            [ET.Ttx(-0.0825), ET.TRx(-90*deg), ET.Ttz(0.384), ET.TRz()],
            name='link4',
            parent=l3
        )

        l5 = ELink(
            [ET.TRx(90*deg), ET.TRz()],
            name='link5',
            parent=l4
        )

        l6 = ELink(
            [ET.Ttx(0.088), ET.TRx(90*deg), ET.Ttz(0.107), ET.TRz()],
            name='link6',
            parent=l5
        )

        ee = ELink(
            [ET.Ttz(tool_offset), ET.TRz(-np.pi/4)],
            name='ee',
            parent=l6
        )

        ETlist = [b0, b1, l0, l1, l2, l3, l4, l5, l6, ee]

        super(Frankie, self).__init__(
            ETlist,
            name='Frankie',
            manufacturer='Franka Emika, Omron')

        self._qz = np.array([0, 0, 0, 0, 0, 0, 0, 0])
        self._qr = np.array([0, 0, -90, -90, 90, 0, -90, 90]) * deg

    @property
    def qz(self):
        return self._qz

    @property
    def qr(self):
        return self._qr
