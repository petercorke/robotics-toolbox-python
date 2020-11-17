"""
@author: Peter Corke
@author: Samuel Drew
"""

from roboticstoolbox import DHRobot, RevoluteDH
from math import pi
import numpy as np


class IRB140(DHRobot):
    """
    Class that models an ABB  IRB140 manipulator

    ``IRB140()`` is a class which models a Unimation Puma560 robot and
    describes its kinematic and dynamic characteristics using standard DH
    conventions.

    .. runblock:: pycon

        >>> import roboticstoolbox as rtb
        >>> robot = rtb.models.DH.IRB140()
        >>> print(robot)

    Defined joint configurations are:

        - qz, zero joint angle configuration
        - qr, vertical 'READY' configuration
        - qd, lower arm horizontal as per data sheet

    .. note:: SI units of metres are used.

    :references:
        - "IRB 140 data sheet", ABB Robotics.
        - "Utilizing the Functional Work Space Evaluation Tool for Assessing a
          System Design and Reconfiguration Alternatives"
          A. Djuric and R. J. Urbanic
        - https://github.com/4rtur1t0/ARTE/blob/master/robots/ABB/IRB140/parameters.m

    .. codeauthor:: Peter Corke
    """  # noqa
    def __init__(self):
        deg = pi/180

        # robot length values (metres)
        d1 = 0.352
        a1 = 0.070
        a2 = 0.360
        d4 = 0.380
        d6 = 0.065

        # Create Links
        # Updated values form ARTE git. Old values left as comments

        L = [
            RevoluteDH(
                d=d1,
                a=a1,
                alpha=-pi/2,
                m=34655.36e-3,
                r=np.array([27.87, 43.12, -89.03])*1e-3,
                I=np.array([512052539.74, 1361335.88, 51305020.72,
                            1361335.88, 464074688.59, 70335556.04,
                            51305020.72, 70335556.04, 462745526.12])*1e-9,
                qlim=[-180 * deg, 180 * deg]
            ),

            RevoluteDH(
                d=0,
                a=a2,
                alpha=0,
                m=15994.59e-3,
                r=np.array([198.29, 9.73, 92.43])*1e03,
                I=np.array([94817914.40, -3859712.77, 37932017.01,
                            -3859712.77, 328604163.24, -1088970.86,
                            37932017.01, -1088970.86, 277463004.88])*1e-9,
                qlim=[-100 * deg, 100 * deg]
            ),

            RevoluteDH(
                d=0,
                a=0,
                alpha=-pi/2,  # alpha=pi/2,
                m=20862.05e-3,
                r=np.array([-4.56, -79.96, -5.86]),
                I=np.array([500060915.95, -1863252.17, 934875.78,
                            -1863252.17, 75152670.69, -15204130.09,
                            934875.78, -15204130.09, 515424754.34])*1e-9,
                qlim=[-220 * deg, 60 * deg]
            ),

            RevoluteDH(
                d=d4,
                a=0,
                alpha=pi/2,  # alpha=-pi/2,
                qlim=[-200 * deg, 200 * deg]
            ),

            RevoluteDH(
                d=0,
                a=0,
                alpha=-pi/2,  # alpha=pi/2,
                qlim=[-120 * deg, 120 * deg]
            ),

            RevoluteDH(
                d=d6,
                a=0,
                alpha=0,  # alpha=pi/2,
                qlim=[-400 * deg, 400 * deg]
            )
        ]

        super().__init__(
            L,
            # basemesh="ABB/IRB140/link0.stl",
            name='IRB 140',
            manufacturer='ABB',
            meshdir="meshes/ABB/IRB140")

        self.addconfiguration("qz", [0, 0, 0, 0, 0, 0])
        self.addconfiguration("qd", [0, -90*deg, 180*deg, 0, 0, -90*deg])
        self.addconfiguration("qr", [0, -90*deg, 90*deg, 0, 90*deg, -90*deg])


if __name__ == '__main__':   # pragma nocover

    robot = IRB140()
    print(robot)
