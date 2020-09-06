#!/usr/bin/env python
"""
@author: Peter Corke
@author: Jesse Haviland
"""

# 2/8/95  changed D3 to 150.05mm which is closer to data from Lee, AKB86 and
# Tarn fixed errors in COG for links 2 and 3
# 29/1/91 to agree with data from Armstrong etal.  Due to their use
#  of modified D&H params, some of the offsets Ai, Di are
#  offset, and for links 3-5 swap Y and Z axes.
# 14/2/91 to use Paul's value of link twist (alpha) to be consistant
#  with ARCL.  This is the -ve of Lee's values, which means the
#  zero angle position is a righty for Paul, and lefty for Lee.

# all parameters are in SI units: m, radians, kg, kg.m2, N.m, N.m.s etc.

from ropy import SerialLink
from ropy import Revolute
from math import pi
import numpy as np


class Puma560(SerialLink):
    """
    Create model of Puma 560 manipulator

    puma = Puma560() is a script which creates a puma SerialLink object
    describing the kinematic and dynamic characteristics of a Unimation Puma
    560 manipulator using standard DH conventions.

    Also define some joint configurations:
    - qz, zero joint angle configuration, 'L' shaped configuration
    - qr, vertical 'READY' configuration
    - qs, arm is stretched out in the X direction
    - qn, arm is at a nominal non-singular configuration

    :notes:
        - SI units are used.
        - The model includes armature inertia and gear ratios.
        - The value of m1 is given as 0 here.  Armstrong found no value for it
          and it does not appear in the equation for tau1 after the
          substituion is made to inertia about link frame rather than COG
          frame.
        - Gravity load torque is the motor torque necessary to keep the joint
          static, and is thus -ve of the gravity caused torque.

    :references:
        - "A search for consensus among model parameters reported for the PUMA
          560 robot", P. Corke and B. Armstrong-Helouvry,
          Proc. IEEE Int. Conf. Robotics and Automation, (San Diego),
          pp. 1608-1613, May 1994. (for kinematic and dynamic parameters)
        - "A combined optimization method for solving the inverse kinematics
           problem", Wang & Chen, IEEE Trans. RA 7(4) 1991 pp 489-.
           (for joint angle limits)

    """

    def __init__(self):

        deg = pi/180

        L0 = Revolute(
            d=0,          # link length (Dennavit-Hartenberg notation)
            a=0,          # link offset (Dennavit-Hartenberg notation)
            alpha=pi/2,   # link twist (Dennavit-Hartenberg notation)
            I=[0, 0.35, 0, 0, 0, 0],  # inertia tensor of link with respect to
                                      # center of mass I = [L_xx, L_yy, L_zz,
                                      # L_xy, L_yz, L_xz]
            r=[0, 0, 0],  # distance of ith origin to center of mass [x,y,z]
                          # in link reference frame
            m=0,          # mass of link
            Jm=200e-6,    # actuator inertia
            G=-62.6111,   # gear ratio
            B=1.48e-3,    # actuator viscous friction coefficient (measured
                          # at the motor)
            Tc=[0.395, -0.435],  # actuator Coulomb friction coefficient for
                                 # direction [-,+] (measured at the motor)
            qlim=[-160*deg, 160*deg])    # minimum and maximum joint angle

        L1 = Revolute(
            d=0, a=0.4318, alpha=0,
            I=[0.13, 0.524, 0.539, 0, 0, 0],
            r=[-0.3638, 0.006, 0.2275],
            m=17.4, Jm=200e-6, G=107.815,
            B=.817e-3, Tc=[0.126, -0.071],
            qlim=[-45*deg, 225*deg])

        L2 = Revolute(
            d=0.15005, a=0.0203, alpha=-pi/2,
            I=[0.066, 0.086, 0.0125, 0, 0, 0],
            r=[-0.0203, -0.0141, 0.070],
            m=4.8, Jm=200e-6, G=-53.7063,
            B=1.38e-3, Tc=[0.132, -0.105],
            qlim=[-225*deg, 45*deg])

        L3 = Revolute(
            d=0.4318, a=0, alpha=pi/2,
            I=[1.8e-3, 1.3e-3, 1.8e-3, 0, 0, 0],
            r=[0, 0.019, 0],
            m=0.82, Jm=33e-6, G=76.0364,
            B=71.2e-6, Tc=[11.2e-3, -16.9e-3],
            qlim=[-110*deg, 170*deg])

        L4 = Revolute(
            d=0, a=0, alpha=-pi/2,
            I=[0.3e-3, 0.4e-3, 0.3e-3, 0, 0, 0],
            r=[0, 0, 0], m=0.34,
            Jm=33e-6, G=71.923, B=82.6e-6,
            Tc=[9.26e-3, -14.5e-3],
            qlim=[-100*deg, 100*deg])

        L5 = Revolute(
            d=0, a=0, alpha=0,
            I=[0.15e-3, 0.15e-3, 0.04e-3, 0, 0, 0],
            r=[0, 0, 0.032], m=0.09, Jm=33e-6,
            G=76.686, B=36.7e-6, Tc=[3.96e-3, -10.5e-3],
            qlim=[-266*deg, 266*deg])

        L = [L0, L1, L2, L3, L4, L5]

        # zero angles, L shaped pose
        self._qz = np.array([0, 0, 0, 0, 0, 0])

        # ready pose, arm up
        self._qr = np.array([0, pi/2, -pi/2, 0, 0, 0])

        # straight and horizontal
        self._qs = np.array([0, 0, -pi/2, 0, 0, 0])

        # nominal table top picking pose
        self._qn = np.array([0, pi/4, pi, 0, pi/4, 0])

        super(Puma560, self).__init__(
            L,
            name="Puma 560",
            manufacturer="Unimation")

    @property
    def qz(self):
        return self._qz

    @property
    def qr(self):
        return self._qr

    @property
    def qs(self):
        return self._qs

    @property
    def qn(self):
        return self._qn
