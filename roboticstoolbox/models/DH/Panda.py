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
        tool_offset = (103) * mm

        flange = (107) * mm
        # d7 = (58.4)*mm

        # This Panda model is defined using modified
        # Denavit-Hartenberg parameters
        L = [
            RevoluteMDH(
                a=0.0,
                d=0.333,
                alpha=0.0,
                qlim=np.array([-2.8973, 2.8973]),
                m=4.970684,
                I=[
                    7.03370e-01,
                    7.06610e-01,
                    9.11700e-03,
                    -1.39000e-04,
                    1.91690e-02,
                    6.77200e-03,
                ],
                G=1,
            ),
            RevoluteMDH(
                a=0.0,
                d=0.0,
                alpha=-np.pi / 2,
                qlim=np.array([-1.7628, 1.7628]),
                m=0.646926,
                I=[
                    7.96200e-03,
                    2.81100e-02,
                    2.59950e-02,
                    -3.92500e-03,
                    7.04000e-04,
                    1.02540e-02,
                ],
                G=1,
            ),
            RevoluteMDH(
                a=0.0,
                d=0.316,
                alpha=np.pi / 2,
                qlim=np.array([-2.8973, 2.8973]),
                m=3.228604,
                I=[
                    3.72420e-02,
                    3.61550e-02,
                    1.08300e-02,
                    -4.76100e-03,
                    -1.28050e-02,
                    -1.13960e-02,
                ],
                G=1,
            ),
            RevoluteMDH(
                a=0.0825,
                d=0.0,
                alpha=np.pi / 2,
                qlim=np.array([-3.0718, -0.0698]),
                m=3.587895,
                I=[
                    2.58530e-02,
                    1.95520e-02,
                    2.83230e-02,
                    7.79600e-03,
                    8.64100e-03,
                    -1.33200e-03,
                ],
                G=1,
            ),
            RevoluteMDH(
                a=-0.0825,
                d=0.384,
                alpha=-np.pi / 2,
                qlim=np.array([-2.8973, 2.8973]),
                m=1.225946,
                I=[
                    3.55490e-02,
                    2.94740e-02,
                    8.62700e-03,
                    -2.11700e-03,
                    2.29000e-04,
                    -4.03700e-03,
                ],
                G=1,
            ),
            RevoluteMDH(
                a=0.0,
                d=0.0,
                alpha=np.pi / 2,
                qlim=np.array([-0.0175, 3.7525]),
                m=1.666555,
                I=[
                    1.96400e-03,
                    4.35400e-03,
                    5.43300e-03,
                    1.09000e-04,
                    3.41000e-04,
                    -1.15800e-03,
                ],
                G=1,
            ),
            RevoluteMDH(
                a=0.088,
                d=flange,
                alpha=np.pi / 2,
                qlim=np.array([-2.8973, 2.8973]),
                m=7.35522e-01,
                I=[
                    1.25160e-02,
                    1.00270e-02,
                    4.81500e-03,
                    -4.28000e-04,
                    -7.41000e-04,
                    -1.19600e-03,
                ],
                G=1,
            ),
        ]

        tool = transl(0, 0, tool_offset) @ trotz(-np.pi / 4)

        super().__init__(
            L,
            name="Panda",
            manufacturer="Franka Emika",
            meshdir="meshes/FRANKA-EMIKA/Panda",
            tool=tool,
        )

        self.qr = np.array([0, -0.3, 0, -2.2, 0, 2.0, np.pi / 4])
        self.qz = np.zeros(7)

        self.addconfiguration("qr", self.qr)
        self.addconfiguration("qz", self.qz)


if __name__ == "__main__":  # pragma nocover

    panda = Panda()
    print(panda)
