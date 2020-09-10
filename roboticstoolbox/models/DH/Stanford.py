#!/usr/bin/env python
"""
@author: Peter Corke
"""

# Note::
# - SI units are used.
# - Gear ratios not currently known, though reflected armature inertia 
#   is known, so gear ratios are set to 1.
#
# References::
# - Kinematic data from "Modelling, Trajectory calculation and Servoing of 
#   a computer controlled arm".  Stanford AIM-177.  Figure 2.3
# - Dynamic data from "Robot manipulators: mathematics, programming and control"
#   Paul 1981, Tables 6.5, 6.6
# - Dobrotin & Scheinman, "Design of a computer controlled manipulator for
#   robot research", IJCAI, 1973.

# all parameters are in SI units: m, radians, kg, kg.m2, N.m, N.m.s etc.

from roboticstoolbox import DHRobot, RevoluteDH, PrismaticDH
from math import pi
import numpy as np


class Stanford(DHRobot):
    """
    Create model of Stanford arm manipulator

    puma = Puma560() is a script which creates a puma SerialLink object
    describing the kinematic and dynamic characteristics of a Unimation Puma
    560 manipulator using standard DH conventions.

    Also define some joint configurations:
    - qz, zero joint angle configuration, 'L' shaped configuration
    - qr, vertical 'READY' configuration
    - qs, arm is stretched out in the X direction
    - qn, arm is at a nominal non-singular configuration
    """

    def __init__(self):

        deg = pi/180
        inch = 0.0254

        L0 = RevoluteDH(
            d=0.412,          # link length (Dennavit-Hartenberg notation)
            a=0,          # link offset (Dennavit-Hartenberg notation)
            alpha=-pi/2,   # link twist (Dennavit-Hartenberg notation)
            I=[0.276, 0.255, 0.071, 0, 0, 0],  # inertia tensor of link with respect to
                                      # center of mass I = [L_xx, L_yy, L_zz,
                                      # L_xy, L_yz, L_xz]
            r=[0, 0.0175, -0.1105],  # distance of ith origin to center of mass [x,y,z]
                          # in link reference frame
            m=9.29,          # mass of link
            Jm=0.953,    # actuator inertia
            G=1,   # gear ratio
            qlim=[-170*deg, 170*deg])    # minimum and maximum joint angle

        L1 = RevoluteDH(
            d=0.154, a=0., alpha=pi/2,
            I=[0.108, 0.018, 0.100, 0, 0, 0],
            r=[0, -1.054,  0],
            m=5.01, Jm=2.193, G=1,
            qlim=[-170*deg, 170*deg])

        L2 = PrismaticDH(
            theta=-pi/2, a=0.0203, alpha=0,
            I=[2.51, 2.51, 0.006, 0, 0, 0],
            r=[0, 0, -6.447],
            m=4.25, Jm=0.782, G=1,
            qlim=[12*inch, (12+38)*inch])

        L3 = RevoluteDH(
            d=0, a=0, alpha=-pi/2,
            I=[0.002, 0.001, 0.001, 0, 0, 0],
            r=[0, 0.092, -0.054],
            m=1.08, Jm=0.106, G=1,
            qlim=[-170*deg, 170*deg])

        L4 = RevoluteDH(
            d=0, a=0, alpha=pi/2,
            I=[0.003, 0.0004, 0, 0, 0, 0],
            r=[0, 0.566, 0.003], m=0.630,
            Jm=0.097, G=1,
            qlim=[-90*deg, 90*deg])

        L5 = RevoluteDH(
            d=0, a=0, alpha=0,
            I=[0.013, 0.013, 0.0003, 0, 0, 0],
            r=[0, 0, 1.554], m=0.51, Jm=0.020,
            G=1,
            qlim=[-170*deg, 170*deg])

        L = [L0, L1, L2, L3, L4, L5]

        # zero angles, L shaped pose
        self._qz = np.array([0, 0, 0, 0, 0, 0])

        super(Stanford, self).__init__(
            L,
            name="Stanford arm",
            manufacturer="Victor Scheinman")

    @property
    def qz(self):
        return self._qz

if __name__ == '__main__':

    stanford = Stanford()
    print(stanford)