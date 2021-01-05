#!/usr/bin/env python

import numpy as np
from spatialmath.base import trotz, transl
from roboticstoolbox import DHRobot, RevoluteMDH


class Panda(DHRobot):
    """
    A class representing the Panda robot arm.

    ``Panda()`` is a class which models a Franka-Emika Panda robot and
    describes its kinematic characteristics using modified DH
    conventions.

    .. runblock:: pycon

        >>> import roboticstoolbox as rtb
        >>> robot = rtb.models.DH.Panda()
        >>> print(robot)

    .. note::
        - SI units of metres are used.
        - The model includes a tool offset.

    :references:
        - https://frankaemika.github.io/docs/control_parameters.html

    .. codeauthor:: Samuel Drew
    .. codeauthor:: Peter Corke
    """

    def __init__(self):

        # deg = np.pi/180
        mm = 1e-3
        tool_offset = (103)*mm

        flange = (107)*mm
        # d7 = (58.4)*mm

        # This Panda model is defined using modified
        # Denavit-Hartenberg parameters
        L = [
                RevoluteMDH(
                    a=0.0,
                    d=0.333,
                    alpha=0.0,
                    qlim=np.array([-2.8973, 2.8973])
                ),

                RevoluteMDH(
                    a=0.0,
                    d=0.0,
                    alpha=-np.pi/2,
                    qlim=np.array([-1.7628, 1.7628])
                ),

                RevoluteMDH(
                    a=0.0,
                    d=0.316,
                    alpha=np.pi/2,
                    qlim=np.array([-2.8973, 2.8973])
                 ),

                RevoluteMDH(
                    a=0.0825,
                    d=0.0,
                    alpha=np.pi/2,
                    qlim=np.array([-3.0718, -0.0698])
                ),

                RevoluteMDH(
                    a=-0.0825,
                    d=0.384,
                    alpha=-np.pi/2,
                    qlim=np.array([-2.8973, 2.8973])
                ),

                RevoluteMDH(
                    a=0.0,
                    d=0.0,
                    alpha=np.pi/2,
                    qlim=np.array([-0.0175, 3.7525])
                ),

                RevoluteMDH(
                    a=0.088,
                    d=flange,
                    alpha=np.pi/2,
                    qlim=np.array([-2.8973, 2.8973])
                )
        ]

        tool = transl(0, 0, tool_offset) @  trotz(-np.pi/4)

        super().__init__(
            L,
            name='Panda',
            manufacturer='Franka Emika',
            meshdir='meshes/FRANKA-EMIKA/Panda',
            tool=tool)

        self.addconfiguration("qz", [0, 0, 0, 0, 0, 0, 0])
        self.addconfiguration("qr", np.r_[0, -0.3, 0, -2.2, 0, 2.0, np.pi/4])


if __name__ == '__main__':   # pragma nocover

    panda = Panda()
    print(panda)

