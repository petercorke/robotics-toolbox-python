"""
@author: Peter Corke
@author: Samuel Drew
"""

from roboticstoolbox import DHRobot, RevoluteDH
from math import pi


class KR5(DHRobot):
    '''
    KR5 Create model of Kuka KR5 manipulator

    MDL_KR5 is a script that creates the workspace variable KR5 which
    describes the kinematic characteristics of a Kuka KR5 manipulator using
    standard DH conventions.

    Also define the workspace vectors:
      qk1        nominal working position 1
      qk2        nominal working position 2
      qk3        nominal working position 3

    Notes::
    - SI units of metres are used.
    - Includes an 11.5cm tool in the z-direction

    Author::
    - Gautam Sinha,
      Indian Institute of Technology, Kanpur.

    Define simplest line model for KUKA KR5 robot
    Contain DH parameters for KUKA KR5 robot
    All link lenghts and offsets are measured in cm
    '''
    def __init__(self):

        L1 = RevoluteDH(a=0.18, d=0.4,
                        alpha=pi/2,
                        mesh='KUKA/KR5_arc/link1.stl')
        L2 = RevoluteDH(a=0.6, d=0.135,
                        alpha=pi,
                        mesh='KUKA/KR5_arc/link2.stl')
        L3 = RevoluteDH(a=0.12,
                        d=0.135,
                        alpha=-pi/2,
                        mesh='KUKA/KR5_arc/link3.stl')
        L4 = RevoluteDH(a=0.0,
                        d=0.62,
                        alpha=pi/2,
                        mesh='KUKA/KR5_arc/link4.stl')
        L5 = RevoluteDH(a=0.0,
                        d=0.0,
                        alpha=-pi/2,
                        mesh='KUKA/KR5_arc/link5.stl')
        L6 = RevoluteDH(mesh='KUKA/KR5_arc/link6.stl')

        L = [L1, L2, L3, L4, L5, L6]

        self._qz = [0, 0, 0, 0, 0, 0]

        self._qk1 = [pi/4, pi/3, pi/4, pi/6, pi/4, pi/6]

        self._qk2 = [pi/4, pi/3, pi/6, pi/3, pi/4, pi/6]

        self._qk3 = [pi/6, pi/3, pi/6, pi/3, pi/6, pi/3]

        # Create SerialLink object
        super(KR5, self).__init__(
            L,
            basemesh="KUKA/KR5_arc/link0.stl",
            name='KR5',
            manufacturer='KUKA')

    @property
    def qz(self):
        return self._qz

    @property
    def qk1(self):
        return self._qk1

    @property
    def qk2(self):
        return self._qk2

    @property
    def qk3(self):
        return self._qk3
