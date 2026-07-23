"""
@author: Peter Corke
"""

from roboticstoolbox import DHRobot, RevoluteDH, RevoluteMDH

# from math import pi
from spatialmath import SE3
import numpy as np


class TwoLink(DHRobot):
    """
    Class that models a 2-link robot moving in the vertical plane

    :param symbolic: use symbolic constants
    :param mdh: create a model using modified DH parameters, otherwise standard DH parameters are used
    :param inertia: include link inertias, otherwise they are set to zero

    ``TwoLink()`` is a class which models a 2-link planar robot and
    describes its kinematic and dynamic characteristics using standard DH
    conventions. of a simple planar 2-link mechanism moving in the xz-plane, it experiences gravity loading.
    All mass is concentrated at the joints.

    .. runblock:: pycon

        >>> import roboticstoolbox as rtb
        >>> robot = rtb.models.DH.TwoLink()
        >>> print(robot)

    The parameters values depend on the ``symbolic`` and ``mdh`` parameters

    ===========================================  =================  ==============
    Parameters                                   Numeric values     Symbolic values
    ===========================================  =================  ==============
    link lengths                                 1, 1               a1, a2
    link masses                                  1, 1               m1, m2
    link CoMs in the link frame x-direction DH   -0.5, -0.5         c1, c2
    link CoMs in the link frame x-direction MDH  0.5, 0.5           c1, c2
    gravitational acceleration                   9.8                g
    ===========================================  =================  ==============

    Defined joint configurations are:

        - qz, zero angles, all folded up
        - q1, links are horizontal and vertical respectively
        - q2, links are vertical and horizontal respectively
        - qn, nominal working configuration

    .. note::

        - Robot has only 2 DoF.
        - Motor inertia is 0.
        - Link inertias are 0 unless ``inertia`` is True.
        - Viscous and Coulomb friction is 0.

    :Reference: Based on Fig 3-6 (p73) of Spong and Vidyasagar (1st edition).

    .. codeauthor:: Peter Corke
    """

    def __init__(self, symbolic:bool=False, mdh:bool=False, inertia:bool=False):

        if symbolic:
            import spatialmath.base.symbolic as sym

            zero = sym.zero()
            pi = sym.pi()
            a1, a2 = sym.symbol("a1 a2")  # link lengths # type: ignore
            m1, m2 = sym.symbol("m1 m2")  # link masses # type: ignore
            c1, c2 = sym.symbol("c1 c2")  # link CoMs location relative to link frames# type: ignore
            if inertia:
                r = sym.symbol("r") # link radius, assumed tubular, for inertia calculation # type: ignore
                I1, I2 = _cylinder_inertia_x(m1, r, a1), _cylinder_inertia_x(m2, r, a2)  # moments of inertia about CoM # type: ignore
            else:
                I1, I2 = None, None
            g = sym.symbol("g")
        else:
            from math import pi


        if mdh:
            # create a modified DH model
            if not symbolic:
                zero = 0.0
                a1 = 1 # length of first link
                a2 = 1 # length of second link
                m1 = 1 # mass of first link
                m2 = 1 # mass of second link
                c1 = 0.5 # CoM location of first link relative to first link frame
                c2 = 0.5 # CoM location of second link relative to second link frame
                r = 0.1 # radius of links, assumed tubular, for inertia calculation
                if inertia:
                    I1, I2 = _cylinder_inertia_x(m1, r, a1), _cylinder_inertia_x(m2, r, a2)  # moments of inertia about CoM # type: ignore
                else:
                    I1, I2 = None, None
                g = 9.8

            links = [
                RevoluteMDH(a=0, alpha=zero, m=m1, r=[c1, zero, zero], I=I1),
                RevoluteMDH(a=a1, alpha=zero, m=m2, r=[c2, zero, zero], I=I2),
            ]
            tool = SE3.Tx(a2) # the last link is considered as a tool in this case, so the tool is a translation along x by a2
        else:
            # create a standard DH model
            if not symbolic:
                zero = 0.0
                a1 = 1 # length of first link
                a2 = 1 # length of second link
                m1 = 1 # mass of first link
                m2 = 1 # mass of second link
                c1 = -0.5 # CoM location of first link relative to first link frame
                c2 = -0.5 # CoM location of second link relative to second link frame
                r = 0.1 # radius of links, assumed tubular, for inertia calculation
                if inertia:
                    I1, I2 = _cylinder_inertia_x(m1, r, a1), _cylinder_inertia_x(m2, r, a2)  # moments of inertia about CoM # type: ignore
                else:
                    I1, I2 = None, None
                g = 9.8
            links = [
                RevoluteDH(a=a1, alpha=zero, m=m1, r=[c1, 0, 0], I=I1),
                RevoluteDH(a=a2, alpha=zero, m=m2, r=[c2, 0, 0], I=I2),
            ]
            tool = None

        super().__init__(
            links, symbolic=symbolic, name="2 link", tool=tool, keywords=("planar", "dynamics")
        )

        self.qr = np.array([pi / 6, -pi / 6])
        self.qz = np.zeros(2)

        self.addconfiguration("qr", self.qr)
        self.addconfiguration("qz", self.qz)

        self.addconfiguration_attr("qz", [0, 0])
        self.addconfiguration_attr("q1", [0, pi / 2])
        self.addconfiguration_attr("q2", [pi / 2, -pi / 2])
        self.addconfiguration_attr("qn", [pi / 6, -pi / 6])

        self.base = SE3.Rx(pi / 2)
        self.gravity = [0, 0, g]


def _cylinder_inertia_x(m, r, L):
    ixx = 0.5 * m * r**2
    iyy = (1.0 / 12.0) * m * (3 * r**2 + L**2)
    izz = iyy
    return np.diag([ixx, iyy, izz])

if __name__ == "__main__":  # pragma nocover
    robot = TwoLink(symbolic=True)
    print(robot)
