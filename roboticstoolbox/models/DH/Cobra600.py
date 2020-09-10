"""
@author: Peter Corke
@author: Samuel Drew
"""

from roboticstoolbox import DHRobot, RevoluteDH, PrismaticDH
from math import pi
import numpy as np


class Cobra600(DHRobot):

    # %MDL_COBRA600 Create model of Adept Cobra 600 manipulator
    # %
    # % MDL_COBRA600 is a script that creates the workspace variable c600 which
    # % describes the kinematic characteristics of the 4-axis Adept Cobra 600
    # % SCARA manipulator using standard DH conventions.
    # %
    # % Also define the workspace vectors:
    # %   qz         zero joint angle configuration
    # %
    # % Notes::
    # % - SI units are used.
    # %
    # % See also SerialRevolute, mdl_puma560akb, mdl_stanford.
    def __init__(self):
        deg = pi/180

        L = [RevoluteDH(d=0.387, a=0.325, qlim=[-50*deg, 50*deg]),
             RevoluteDH(a=0.275, alpha=pi, qlim=[-88*deg, 88*deg]),
             PrismaticDH(qlim=[0, 0.210]),
             RevoluteDH()]

        super(Cobra600, self).__init__(L, name='Cobra600', manufacturer='Adept')

        self._qz = [0, 0, 0, 0]

    @property
    def qz(self):
        return self._qz
