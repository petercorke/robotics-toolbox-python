#!/usr/bin/env python

import numpy as np
from spatialmath.base import trotz, transl
from ropy.robot.Revolute import Revolute
from ropy.robot.SerialLink import SerialLink


class PandaMDH(SerialLink):
    """
    A class representing the Franka Emika Panda robot arm.

    DH Parameters taken from https://frankaemika.github.io/docs/control_parameters.html

    Attributes:
    --------
        name : string
            Name of the robot
        manufacturer : string
            Manufacturer of the robot
        links : List[n]
            Series of links which define the robot
        base : float np.ndarray(4,4)
            Locaation of the base
        tool : float np.ndarray(4,4)
            Location of the tool
        mdh : int
            1: Pnada is modified D&H
        n : int
            Number of joints in the robot

    Examples
    --------
    >>> panda = Panda()

    See Also
    --------
    ropy.robot.SerialLink : A superclass for arm type robots
    """

    def __init__(self):

        # deg = np.pi/180
        mm = 1e-3
        tool_offset = (103)*mm

        flange = (107)*mm
        # d7 = (58.4)*mm

        L1 = Revolute(a =    0.0, d = 0.333, alpha =      0.0, qlim = np.array([-2.8973, 2.8973]), mdh = 1)
        L2 = Revolute(a =    0.0, d =   0.0, alpha = -np.pi/2, qlim = np.array([-1.7628, 1.7628]), mdh = 1)
        L3 = Revolute(a =    0.0, d = 0.316, alpha =  np.pi/2, qlim = np.array([-2.8973, 2.8973]), mdh = 1)
        L4 = Revolute(a = 0.0825, d =   0.0, alpha =  np.pi/2, qlim = np.array([-3.0718, -0.0698]), mdh = 1)
        L5 = Revolute(a =-0.0825, d = 0.384, alpha = -np.pi/2, qlim = np.array([-2.8973, 2.8973]), mdh = 1)
        L6 = Revolute(a =    0.0, d =   0.0, alpha =  np.pi/2, qlim = np.array([-0.0175, 3.7525]), mdh = 1)
        L7 = Revolute(a =  0.088, d =flange, alpha =  np.pi/2, qlim = np.array([-2.8973, 2.8973]), mdh = 1)

        L = [L1, L2, L3, L4, L5, L6, L7]

        # super(Panda, self).__init__(L, name = 'Panda', manufacturer = 'Franka Emika', tool = transl(0, 0, d7))
        tool = transl(0, 0, tool_offset) @  trotz(-np.pi/4)

        super(PandaMDH, self).__init__(
            L,
            name='Panda',
            manufacturer='Franka Emika',
            tool=tool)

        # tool = xyzrpy_to_trans(0, 0, d7, 0, 0, -np.pi/4)

        self.qz = np.array([0, 0, 0, 0, 0, 0, 0])
        # self.qr = np.array([0, -90, -90, 90, 0, -90, 90]) * deg
        self.qr = np.array([0, -0.3, 0, -2.2, 0, 2.0, np.pi/4])

