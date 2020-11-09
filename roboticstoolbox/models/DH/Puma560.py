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

# from math import pi
import numpy as np
from roboticstoolbox import DHRobot, RevoluteDH
from spatialmath import SE3


class Puma560(DHRobot):
    """
    Class that models a Puma 560 manipulator

    :param symbolic: use symbolic constants
    :type symbolic: bool

    ``Puma560()`` is a class which models a Unimation Puma560 robot and
    describes its kinematic and dynamic characteristics using standard DH
    conventions.

    .. runblock:: pycon

        >>> import roboticstoolbox as rtb
        >>> robot = rtb.models.DH.Puma560()
        >>> print(robot)

    Defined joint configurations are:

    - qz, zero joint angle configuration, 'L' shaped configuration
    - qr, vertical 'READY' configuration
    - qs, arm is stretched out in the x-direction
    - qn, arm is at a nominal non-singular configuration

    .. note::
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
        - https://github.com/4rtur1t0/ARTE/blob/master/robots/UNIMATE/puma560/parameters.m

    .. codeauthor:: Peter Corke
    """

    def __init__(self, symbolic=False):

        if symbolic:
            import spatialmath.base.symbolic as sym
            zero = sym.zero()
            pi = sym.pi()
        else:
            from math import pi
            zero = 0.0

        deg = pi / 180
        inch = 0.0254

        # base = 0.672      # from mounting surface to shoulder axis

        L = [
            RevoluteDH(
                d=0,  # d=base,  # link length (Dennavit-Hartenberg notation)
                a=0,          # link offset (Dennavit-Hartenberg notation)
                alpha=pi/2,   # link twist (Dennavit-Hartenberg notation)
                I=[0, 0.35, 0, 0, 0, 0],
                # inertia tensor of link with respect to
                # center of mass I = [L_xx, L_yy, L_zz,
                # L_xy, L_yz, L_xz]
                r=[0, 0, 0],
                # distance of ith origin to center of mass [x,y,z]
                # in link reference frame
                m=0,          # mass of link
                Jm=200e-6,    # actuator inertia
                G=-62.6111,   # gear ratio
                B=1.48e-3,    # actuator viscous friction coefficient (measured
                              # at the motor)
                Tc=[0.395, -0.435],
                # actuator Coulomb friction coefficient for
                # direction [-,+] (measured at the motor)
                qlim=[-160*deg, 160*deg]    # minimum and maximum joint angle
            ),

            RevoluteDH(
                d=0, a=0.4318, alpha=zero,
                I=[0.13, 0.524, 0.539, 0, 0, 0],
                r=[-0.3638, 0.006, 0.2275],
                m=17.4,
                Jm=200e-6,
                G=107.815,
                B=.817e-3,
                Tc=[0.126, -0.071],
                qlim=[-110*deg, 110*deg],  # qlim=[-45*deg, 225*deg]
            ),

            RevoluteDH(
                d=0.15005, a=0.0203, alpha=-pi/2,
                I=[0.066, 0.086, 0.0125, 0, 0, 0],
                r=[-0.0203, -0.0141, 0.070],
                m=4.8,
                Jm=200e-6,
                G=-53.7063,
                B=1.38e-3,
                Tc=[0.132, -0.105],
                qlim=[-135*deg, 135*deg]  # qlim=[-225*deg, 45*deg]
            ),

            RevoluteDH(
                d=0.4318, a=0, alpha=pi/2,
                I=[1.8e-3, 1.3e-3, 1.8e-3, 0, 0, 0],
                r=[0, 0.019, 0],
                m=0.82,
                Jm=33e-6,
                G=76.0364,
                B=71.2e-6,
                Tc=[11.2e-3, -16.9e-3],
                qlim=[-266*deg, 266*deg]  # qlim=[-110*deg, 170*deg]
            ),

            RevoluteDH(
                d=0, a=0, alpha=-pi/2,
                I=[0.3e-3, 0.4e-3, 0.3e-3, 0, 0, 0],
                r=[0, 0, 0],
                m=0.34,
                Jm=33e-6,
                G=71.923,
                B=82.6e-6,
                Tc=[9.26e-3, -14.5e-3],
                qlim=[-100*deg, 100*deg]
            ),

            RevoluteDH(
                d=0, a=0, alpha=zero,
                I=[0.15e-3, 0.15e-3, 0.04e-3, 0, 0, 0],
                r=[0, 0, 0.032],
                m=0.09,
                Jm=33e-6,
                G=76.686,
                B=36.7e-6,
                Tc=[3.96e-3, -10.5e-3],
                qlim=[-266*deg, 266*deg]
            )
        ]
        base = SE3(0, 0, 26.45*inch)  # pedestal

        super().__init__(
            L,
            name="Puma 560",
            manufacturer="Unimation",
            keywords=('dynamics', 'symbolic', 'mesh'),
            symbolic=symbolic,
            meshdir="meshes/UNIMATE/puma560",
            base=base
        )

        # zero angles, L shaped pose
        self.addconfiguration("qz", np.array([0, 0, 0, 0, 0, 0]))

        # ready pose, arm up
        self.addconfiguration("qr", np.array([0, pi/2, -pi/2, 0, 0, 0]))

        # straight and horizontal
        self.addconfiguration("qs", np.array([0, 0, -pi/2, 0, 0, 0]))

        # nominal table top picking pose
        self.addconfiguration("qn", np.array([0, pi/4, pi, 0, pi/4, 0]))

    def _ikine(self, T, config):
        # Puma model with shoulder and elbow offsets
        # - Inverse kinematics for a PUMA 560,
        #   Paul and Zhang,
        #   The International Journal of Robotics Research,
        #   Vol. 5, No. 2, Summer 1986, p. 32-44

        # based on MATLAB code by Robert Biro with Gary Von McMurray,
        # GTRI/ATRP/IIMB, Georgia Institute of Technology, 2/13/95

        # config is l|r  u|d  f|n

        a2 = self.links[1].a
        a3 = self.links[2].a
        d1 = self.links[0].d
        d3 = self.links[2].d
        d4 = self.links[3].d

        # The following parameters are extracted from the Homogeneous
        # Transformation as defined in equation 1, p. 34

        Px, Py, Pz = T.t

        theta = np.zeros((3,))

        # Solve for theta[0]
        # r is defined in equation 38, p. 39.
        # theta[0] uses equations 40 and 41, p.39,
        # based on the configuration parameter n1

        r = np.sqrt(Px**2 + Py**2)
        if 'r' in config:
            theta[0] = np.arctan2(Py, Px) + np.arcsin(d3 / r)
        else:
            theta[0] = np.arctan2(Py, Px) + np.pi - np.arcsin(d3 / r)

        # Solve for theta[1]
        # V114 is defined in equation 43, p.39.
        # r is defined in equation 47, p.39.
        # Psi is defined in equation 49, p.40.
        # theta[1] uses equations 50 and 51, p.40, based on the
        # configuration parameter n2
        if 'u' in config:
            n2 = 1
        else:
            n2 = -1

        if 'r' in config:
            n2 = -n2

        V114 = Px * np.cos(theta[0]) + Py * np.sin(theta[0])

        r = np.sqrt(V114**2 + Pz**2)

        Psi = np.arccos(
            (a2**2 - d4**2 - a3**2 + V114**2 + Pz**2)
            / (2.0 * a2 * r))

        if np.isnan(Psi):
            theta = []
        else:
            theta[1] = np.arctan2(Pz, V114) + n2 * Psi

            # Solve for theta[2]
            # theta[2] uses equation 57, p. 40.
            num = np.cos(theta[1]) * V114 + np.sin(theta[1]) * Pz - a2
            den = np.cos(theta[1]) * Pz - np.sin(theta[1]) * V114
            theta[2] = np.arctan2(a3, d4) - np.arctan2(num, den)
        
        return theta

    def ikine_a(self, T, config="lun"):
        return self.ikine_6s(T, config, self._ikine)

if __name__ == '__main__':    # pragma nocover

    puma = Puma560(symbolic=False)
    print(puma)
    T = SE3(0.5, 0.2, 0.5) * SE3.OA([0,0,1], [1,0,0])
    (q, failed, reason) = puma.ikine(T)
    print(failed, q)
    qq = puma.ikine_a(T)
    print(qq)
    print(puma.fkine(qq))