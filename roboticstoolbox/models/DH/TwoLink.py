"""
@author: Peter Corke
"""

from roboticstoolbox import DHRobot, RevoluteDH
from math import pi
from spatialmath import SE3


class TwoLink(DHRobot):
    """
    Class that models a 2-link robot moving in the vertical plane

    :param symbolic: use symbolic constants
    :type symbolic: bool

    ``TwoLink()`` is a class which models a 2-link planar robot and
    describes its kinematic and dynamic characteristics using standard DH
    conventions. of a simple planar 2-link mechanism moving in the xz-plane, it experiences gravity loading.
    All mass is concentrated at the joints.

    .. runblock:: pycon

        >>> import roboticstoolbox as rtb
        >>> robot = rtb.models.DH.TwoLink()
        >>> print(robot)

    The parameters values depend on the ``symbolic`` parameter

    =======================================  =================  ==============
    Parameters                               Numeric values     Symbolic values
    =======================================  =================  ==============
    link lengths                             1, 1               a1, a2
    link masses                              1, 1               m1, m2
    link CoMs in the link frame x-direction  -0.5, -0.5         c1, c2
    gravitational acceleration               9.8                g
    =======================================  =================  ==============

    Defined joint configurations are:

        - qz, zero angles, all folded up
        - q1, links are horizontal and vertical respectively
        - q2, links are vertical and horizontal respectively
        - qn, nominal working configuration

    .. note::

        - Robot has only 2 DoF.
        - Motor inertia is 0.
        - Link inertias are 0.
        - Viscous and Coulomb friction is 0. 

    :Reference: Based on Fig 3-6 (p73) of Spong and Vidyasagar (1st edition).

    .. codeauthor:: Peter Corke
    """

    def __init__(self, symbolic=False):

        if symbolic:
            import spatialmath.base.symbolic as sym
            zero = sym.zero()
            pi = sym.pi()
            a1, a2 = sym.symbol('a1 a2')
            m1, m2 = sym.symbol('m1 m2')
            c1, c2 = sym.symbol('c1 c2')
            g = sym.symbol('g')
        else:
            from math import pi
            zero = 0.0
            a1 = 1
            a2 = 1
            m1 = 1
            m2 = 1
            c1 = -0.5
            c2 = -0.5
            g = 9.8

        links = [
                RevoluteDH(a=a1, alpha=zero, m=m1, r=[c1, 0, 0]),
                RevoluteDH(a=a2, alpha=zero, m=m2, r=[c2, 0, 0])
            ]

        super().__init__(
            links,
            symbolic=symbolic,
            name='2 link', 
            keywords=('planar', 'dynamics')
        )
        self.addconfiguration("qz", [0, 0])
        self.addconfiguration("q1", [0, pi/2])
        self.addconfiguration("q2", [pi/2, -pi/2])
        self.addconfiguration("qn", [pi/6, -pi/6])

        self.base = SE3.Rx(pi/2)
        self.gravity = [0, 0, g]

if __name__ == '__main__':   # pragma nocover

    robot = TwoLink(symbolic=True)
    print(robot)






