
"""
@author: Peter Corke
@author: Samuel Drew
"""

from roboticstoolbox import DHRobot, RevoluteDH, PrismaticDH
from math import pi
import numpy as np


class IRB140(DHRobot):
    """
    IRB140 Create model of ABB IRB 140 manipulator

    IRB140 is a script that creates the workspace variable irb140 which
    describes the kinematic characteristics of an ABB IRB 140 manipulator
    using standard DH conventions.
    Also define the workspace vectors:
      qz         zero joint angle configuration
      qr         vertical 'READY' configuration
      qd         lower arm horizontal as per data sheet

    Reference::
    - "IRB 140 data sheet", ABB Robotics.
    - "Utilizing the Functional Work Space Evaluation Tool for Assessing a
      System Design and Reconfiguration Alternatives"
      A. Djuric and R. J. Urbanic

    Notes::
    - SI units of metres are used.
    - Unlike most other mdl_xxx scripts this one is actually a function that
      behaves like a script and writes to the global workspace.
    """
    def __init__(self):
        deg = pi/180

        # robot length values (metres)
        d1 = 0.352
        a1 = 0.070
        a2 = 0.360
        d4 = 0.380
        d6 = 0.065

        # Create Links
        L1 = RevoluteDH(
            theta=0,
            d=d1,
            a=a1,
            alpha=-pi/2,
            m=34655.36e-3,
            r=np.array([27.87, 43.12, -89.03])*1e-3,
            I=np.array([512052539.74, 1361335.88, 51305020.72,
                        1361335.88, 464074688.59, 70335556.04,
                        51305020.72, 70335556.04, 462745526.12])*1e-9

        L2 = RevoluteDH(
            theta=0,
            d=0,
            a=a2,
            alpha=0,
            m=15994.59e-3,
            r=np.array([198.29, 9.73, 92.43])*1e03,
            I=np.array([94817914.40, -3859712.77, 37932017.01,
                        -3859712.77, 328604163.24, -1088970.86,
                        37932017.01, -1088970.86, 277463004.88])*1e-9
        )

        L3 = RevoluteDH(
            theta=0,
            d=0,
            a=0,
            alpha=pi/2,
            m=20862.05e-3,
            r=np.array([-4.56, -79.96, -5.86]),
            I=np.array([500060915.95, -1863252.17, 934875.78,
                        -1863252.17, 75152670.69, -15204130.09,
                        934875.78, -15204130.09, 515424754.34])*1e-9,
        )

        L4 = RevoluteDH(
            theta=0,
            d=d4,
            a=0,
            alpha=-pi/2
        )

        L5 = RevoluteDH(
            theta=0,
            d=0,
            a=0,
            alpha=pi/2
        )

        L6 = RevoluteDH(
            theta=0,
            d=d6,
            a=0,
            alpha=pi/2
        )

        L = [L1, L2, L3, L4, L5, L6]

        self._qz = np.array([0, 0, 0, 0, 0, 0])

        self._qd = np.array([0, -90*deg, 180*deg, 0, 0, -90*deg])

        self._qr = np.array([0, -90*deg, 90*deg, 0, 90*deg, -90*deg])

        super(Irb140, self).__init__(
            L,
            basemesh="ABB/IRB140/link0.stl",
            name='IRB 140',
            manufacturer='ABB')

    @property
    def qz(self):
        return self._qz

    @property
    def qr(self):
        return self._qr

    @property
    def qd(self):
        return self._qd
