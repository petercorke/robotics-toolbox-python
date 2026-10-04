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
from roboticstoolbox import DHRobot, RevoluteDH, IKSolution, angle_axis
from roboticstoolbox.tools.types import NDArray
from spatialmath import SE3
from spatialmath import base


class Puma560(DHRobot):
    """
    Class that models a Puma 560 manipulator

    :param symbolic: use symbolic constants
    :type symbolic: bool

    ``Puma560()`` is an object which models a Unimation Puma560 robot and
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

    .. warning:: Compared to the MATLAB version of the Toolbox this model
        includes the pedestal, making the z-coordinates 26 inches larger.

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
    """  # noqa

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

        base = 26.45 * inch  # from mounting surface to shoulder axis

        L = [
            RevoluteDH(
                d=base,  # link length (Dennavit-Hartenberg notation)
                a=0,  # link offset (Dennavit-Hartenberg notation)
                alpha=pi / 2,  # link twist (Dennavit-Hartenberg notation)
                I=[0, 0.35, 0, 0, 0, 0],
                # inertia tensor of link with respect to
                # center of mass I = [L_xx, L_yy, L_zz,
                # L_xy, L_yz, L_xz]
                r=[0, 0, 0],
                # distance of ith origin to center of mass [x,y,z]
                # in link reference frame
                m=0,  # mass of link
                Jm=200e-6,  # actuator inertia
                G=-62.6111,  # gear ratio
                B=1.48e-3,  # actuator viscous friction coefficient (measured
                # at the motor)
                Tc=[0.395, -0.435],
                # actuator Coulomb friction coefficient for
                # direction [-,+] (measured at the motor)
                qlim=[-160 * deg, 160 * deg],  # minimum and maximum joint angle
            ),
            RevoluteDH(
                d=0,
                a=0.4318,
                alpha=zero,
                I=[0.13, 0.524, 0.539, 0, 0, 0],
                r=[-0.3638, 0.006, 0.2275],
                m=17.4,
                Jm=200e-6,
                G=107.815,
                B=0.817e-3,
                Tc=[0.126, -0.071],
                qlim=[-110 * deg, 110 * deg],  # qlim=[-45*deg, 225*deg]
            ),
            RevoluteDH(
                d=0.15005,
                a=0.0203,
                alpha=-pi / 2,
                I=[0.066, 0.086, 0.0125, 0, 0, 0],
                r=[-0.0203, -0.0141, 0.070],
                m=4.8,
                Jm=200e-6,
                G=-53.7063,
                B=1.38e-3,
                Tc=[0.132, -0.105],
                qlim=[-135 * deg, 135 * deg],  # qlim=[-225*deg, 45*deg]
            ),
            RevoluteDH(
                d=0.4318,
                a=0,
                alpha=pi / 2,
                I=[1.8e-3, 1.3e-3, 1.8e-3, 0, 0, 0],
                r=[0, 0.019, 0],
                m=0.82,
                Jm=33e-6,
                G=76.0364,
                B=71.2e-6,
                Tc=[11.2e-3, -16.9e-3],
                qlim=[-266 * deg, 266 * deg],  # qlim=[-110*deg, 170*deg]
            ),
            RevoluteDH(
                d=0,
                a=0,
                alpha=-pi / 2,
                I=[0.3e-3, 0.4e-3, 0.3e-3, 0, 0, 0],
                r=[0, 0, 0],
                m=0.34,
                Jm=33e-6,
                G=71.923,
                B=82.6e-6,
                Tc=[9.26e-3, -14.5e-3],
                qlim=[-100 * deg, 100 * deg],
            ),
            RevoluteDH(
                d=0,
                a=0,
                alpha=zero,
                I=[0.15e-3, 0.15e-3, 0.04e-3, 0, 0, 0],
                r=[0, 0, 0.032],
                m=0.09,
                Jm=33e-6,
                G=76.686,
                B=36.7e-6,
                Tc=[3.96e-3, -10.5e-3],
                qlim=[-266 * deg, 266 * deg],
            ),
        ]

        super().__init__(
            L,
            name="Puma 560",
            manufacturer="Unimation",
            keywords=("dynamics", "symbolic", "mesh"),
            symbolic=symbolic,
            meshdir="meshes/UNIMATE/puma560",
        )

        self.qr = np.array([0, pi / 2, -pi / 2, 0, 0, 0])
        self.qz = np.zeros(6)

        # nominal table top picking pose
        self.qn = np.array([0, pi / 4, pi, 0, pi / 4, 0])

        self.addconfiguration("qr", self.qr)
        self.addconfiguration("qz", self.qz)
        self.addconfiguration("qn", self.qn)

        # straight and horizontal
        self.addconfiguration_attr("qs", np.array([0, 0, -pi / 2, 0, 0, 0]))

    def ikine_a(
        self, T: SE3 | NDArray, config: str = "lun", tol: float = 1e-6
    ) -> IKSolution:
        r"""
        Analytic inverse kinematic solution

        :param T: end-effector pose, or a trajectory of poses
        :param config: arm configuration, defaults to "lun"
        :param tol: maximum allowed residual error E, as for the numerical IK
            solvers, used to decide whether a solution reaches the pose
        :returns: an IKSolution containing joint coordinates ``q``, ``success``
            flag, ``residual`` error value, and ``reason`` string if applicable

        ``robot.ikine_a(T, config)`` is the joint coordinates which achieve the
        end-effector pose ``T``.  The configuration string selects the specific
        solution and is a string comprising the following letters:

        ======   ==============================================
        Letter   Meaning
        ======   ==============================================
        l        Choose the left-handed configuration
        r        Choose the right-handed configuration
        u        Choose the elbow up configuration
        d        Choose the elbow down configuration
        n        Choose the wrist not-flipped configuration
        f        Choose the wrist flipped configuration
        ======   ==============================================

        ``T`` is an :class:`SE3`, or a 4x4 array, for a single pose.  If it is an
        :class:`SE3` holding N poses, or an array with shape (N, 4, 4), it is a
        trajectory and each pose is solved independently with the same ``config``.
        The result is the same as for the numerical IK solvers:

        - ``q`` has shape (n,) for a single pose, or (N, n) for a trajectory
        - ``success`` is True only if every pose was solved
        - ``residual`` is the maximum residual over the poses, and is infinite if
          any pose is out of reach
        - ``reason`` is the reason given by the last pose that failed, if any
        - ``iterations`` and ``searches`` are zero, this is not an iterative method

        The joints of a pose that cannot be solved are NaN.  Every solution is
        checked by forward kinematics, so ``success`` is True only if the
        returned ``q`` reaches the pose to within ``tol``.

        This is a hand-written solution for the Puma 560 geometry, in particular
        a zero first link length and a lateral shoulder offset :math:`d_3`.  It
        does not apply to other robots.  Joint limits are not considered.

        .. versionchanged:: 1.5.0
            Accepts an array as well as an :class:`SE3`, returns a single
            :class:`IKSolution` for a trajectory like the numerical solvers (it
            was garbled, and failed for an unreachable pose), reports a pose
            closer to the waist axis than the shoulder offset as out of reach
            (it returned ``success=True`` with NaN joint coordinates), and checks
            the solution by forward kinematics.  It no longer uses
            :meth:`DHRobot.ikine_6s`, which is deprecated.

        :reference:
            - Inverse kinematics for a PUMA 560,
              Paul and Zhang,
              The International Journal of Robotics Research,
              Vol. 5, No. 2, Summer 1986, p. 32-44

        :author: based on MATLAB code by Robert Biro with Gary Von McMurray,
            GTRI/ATRP/IIMB, Georgia Institute of Technology, 2/13/95

        """

        config = self.config_validate(config, ("lr", "ud", "nf"))

        def ik3(robot, T, config="lun"):

            # solve for the first three joints

            a2 = robot.links[1].a
            a3 = robot.links[2].a
            d1 = robot.links[0].d
            d3 = robot.links[2].d
            d4 = robot.links[3].d

            # The following parameters are extracted from the Homogeneous
            # Transformation as defined in equation 1, p. 34

            Px, Py, Pz = T.t
            Pz -= d1  # offset the pedestal height
            theta = np.zeros((3,))

            # Solve for theta[0]
            # r is defined in equation 38, p. 39.
            # theta[0] uses equations 40 and 41, p.39,
            # based on the configuration parameter n1

            r = np.sqrt(Px**2 + Py**2)
            if r < abs(d3) or r == 0:
                # the wrist centre is closer to the waist axis than the
                # shoulder offset, so arcsin(d3 / r) below has no solution
                return "Out of reach"
            if "r" in config:
                theta[0] = np.arctan2(Py, Px) + np.arcsin(d3 / r)
            elif "l" in config:
                theta[0] = np.arctan2(Py, Px) + np.pi - np.arcsin(d3 / r)
            else:
                raise ValueError("bad configuration string")

            # Solve for theta[1]
            # V114 is defined in equation 43, p.39.
            # r is defined in equation 47, p.39.
            # Psi is defined in equation 49, p.40.
            # theta[1] uses equations 50 and 51, p.40, based on the
            # configuration parameter n2
            if "u" in config:
                n2 = 1
            elif "d" in config:
                n2 = -1
            else:
                raise ValueError("bad configuration string")

            if "l" in config:
                n2 = -n2

            V114 = Px * np.cos(theta[0]) + Py * np.sin(theta[0])

            r = np.sqrt(V114**2 + Pz**2)

            with np.errstate(invalid="raise"):
                try:
                    Psi = np.arccos(
                        (a2**2 - d4**2 - a3**2 + V114**2 + Pz**2) / (2.0 * a2 * r)
                    )
                except FloatingPointError:
                    return "Out of reach"

            theta[1] = np.arctan2(Pz, V114) + n2 * Psi

            # Solve for theta[2]
            # theta[2] uses equation 57, p. 40.
            num = np.cos(theta[1]) * V114 + np.sin(theta[1]) * Pz - a2
            den = np.cos(theta[1]) * Pz - np.sin(theta[1]) * V114
            theta[2] = np.arctan2(a3, d4) - np.arctan2(num, den)

            theta = base.angdiff(theta)

            return theta

        # spatialmath cannot build an SE3 from an (N, 4, 4) array, see
        # https://github.com/rai-opensource/spatialmath-python/issues/236
        if isinstance(T, np.ndarray):
            if T.shape != (4, 4) and not (T.ndim == 3 and T.shape[1:] == (4, 4)):
                raise ValueError("T must be an SE3, a 4x4 array or an (N, 4, 4) array")
            T = SE3(list(T) if T.ndim == 3 else T, check=False)

        # undo the base and tool transformations, skipped if they are not set
        # (careful: ``base`` is the spatialmath module in ik3 above)
        T_arm = T
        if not np.array_equal(self.base.A, np.eye(4)):
            T_arm = self.base.inv() * T_arm
        if not np.array_equal(self.tool.A, np.eye(4)):
            T_arm = T_arm * self.tool.inv()

        q = np.full((len(T), self.n), np.nan)
        residual = 0.0
        reason = ""
        success = True

        for k, (Tk, Tk_arm) in enumerate(zip(T, T_arm)):
            # solve for the first three joints
            theta = ik3(self, Tk_arm, config)

            if isinstance(theta, str):
                qk = None
                why = theta
            else:
                # Solve for the wrist rotation.  We need to account for some
                # translations between the first and last 3 joints (d4) and
                # also d6, a6, alpha6 in the final frame.

                # transform of the first 3 joints
                T13 = self.A([0, 2], theta)

                # T = T13 * Tz(d4) * R * Tz(d6) Tx(a5)
                Td4 = SE3(0, 0, self.links[3].d)  # Tz(d4)

                # Tz(d6) Tx(a5) Rx(alpha6)
                Tt = SE3(self.links[5].a, 0, self.links[5].d) * SE3.Rx(
                    self.links[5].alpha
                )

                R = Td4.inv() * T13.inv() * Tk_arm * Tt.inv()

                # the spherical wrist implements Euler angles
                eul = R.eul(flip=True) if "f" in config else R.eul()
                qk = np.r_[theta, eul]
                if self.links[3].alpha > 0:
                    qk[4] = -qk[4]

                # remove the link offset angles
                qk = qk - self.offset
                why = ""

            if qk is None or not np.all(np.isfinite(qk)):
                # out of reach, there is no solution at all
                E = np.inf
                why = why or "Out of reach"
            else:
                # check the solution, success is not the absence of an exception
                e = angle_axis(self.fkine(qk).A, Tk.A)
                E = 0.5 * float(e @ e)
                q[k] = qk
                if E > tol:
                    why = f"Solution does not reach the pose, residual {E:.3g}"

            if E > tol or why:
                success = False
                reason = why
            residual = max(residual, E)

        return IKSolution(
            q=q[0] if len(T) == 1 else q,
            success=success,
            residual=residual,
            reason=reason,
        )


if __name__ == "__main__":  # pragma nocover
    puma = Puma560(symbolic=False)
    print(puma)
    print(puma.dynamics())
    # T = puma.fkine(puma.qn)
    # print(puma.ikine_a(T, 'lu').q)
    # print(puma.ikine_a(T, 'ru').q)
    # print(puma.ikine_a(T, 'ld').q)
    # print(puma.ikine_a(T, 'rd').q)

    # puma.plot(puma.qz)
