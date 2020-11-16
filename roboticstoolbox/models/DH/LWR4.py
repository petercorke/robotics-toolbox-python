#!/usr/bin/env python

import numpy as np
from spatialmath.base import trotz, transl
from roboticstoolbox import DHRobot, RevoluteDH


class LWR4(DHRobot):
    """
    Class that models a LWR-IV manipulator

    ``LWR4()`` is a class which models a Kuka LWR-IV robot and
    describes its kinematic characteristics using standard DH
    conventions.

    .. runblock:: pycon

        >>> import roboticstoolbox as rtb
        >>> robot = rtb.models.DH.LWR4()
        >>> print(robot)

    Defined joint configurations are:

    - qz, zero joint angle configuration, 'L' shaped configuration
    - qr, vertical 'READY' configuration
    - qs, arm is stretched out in the X direction
    - qn, arm is at a nominal non-singular configuration

    .. note:: SI units are used.

    :references:

        - http://www.diag.uniroma1.it/~deluca/rob1_en/09_Exercise_DH_KukaLWR4.pdf

    .. codeauthor:: Peter Corke
    """  # noqa

    def __init__(self):

        # deg = np.pi/180
        mm = 1e-3
        tool_offset = (103) * mm

        flange = 0 * mm
        # d7 = (58.4)*mm

        # This Kuka model is defined using modified
        # Denavit-Hartenberg parameters

        L = [
                RevoluteDH(
                    a=0.0,
                    d=0,
                    alpha=np.pi/2,
                    qlim=np.array([-2.8973, 2.8973])
                ),

                RevoluteDH(
                    a=0.0,
                    d=0.0,
                    alpha=-np.pi/2,
                    qlim=np.array([-1.7628, 1.7628])
                ),

                RevoluteDH(
                    a=0.0,
                    d=0.40,
                    alpha=-np.pi/2,
                    qlim=np.array([-2.8973, 2.8973])
                ),

                RevoluteDH(
                    a=0.0,
                    d=0.0,
                    alpha=np.pi/2,
                    qlim=np.array([-3.0718, -0.0698])
                ),

                RevoluteDH(
                    a=0.0,
                    d=0.39,
                    alpha=np.pi/2,
                    qlim=np.array([-2.8973, 2.8973])
                ),

                RevoluteDH(
                    a=0.0,
                    d=0.0,
                    alpha=-np.pi/2,
                    qlim=np.array([-0.0175, 3.7525])
                ),

                RevoluteDH(
                    a=0.0,
                    d=flange,
                    alpha=0.0,
                    qlim=np.array([-2.8973, 2.8973])
                )
        ]

        tool = transl(0, 0, tool_offset) @  trotz(-np.pi/4)

        super().__init__(
            L,
            name='LWR-IV',
            manufacturer='Kuka',
            tool=tool)

        # tool = xyzrpy_to_trans(0, 0, d7, 0, 0, -np.pi/4)

        self.addconfiguration("qz", [0, 0, 0, 0, 0, 0, 0])


if __name__ == '__main__':   # pragma nocover

    robot = LWR4()
    print(robot)
