import numpy as np
from roboticstoolbox import DHRobot, RevoluteDH
from spatialmath import SE3


class UR3(DHRobot):
    """
    Class that models a Universal Robotics UR3 manipulator

    :param symbolic: use symbolic constants
    :type symbolic: bool

    ``UR3()`` is an object which models a Unimation Puma560 robot and
    describes its kinematic and dynamic characteristics using standard DH
    conventions.

    .. runblock:: pycon

        >>> import roboticstoolbox as rtb
        >>> robot = rtb.models.DH.UR3()
        >>> print(robot)

    Defined joint configurations are:

    - qz, zero joint angle configuration
    - qr, arm horizontal along x-axis

    .. note::
        - SI units are used.

    :References:

        - `Parameters for calculations of kinematics and dynamics <https://www.universal-robots.com/articles/ur/parameters-for-calculations-of-kinematics-and-dynamics>`_
        
    :sealso: :func:`UR5`, :func:`UR10`


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

        # robot length values (metres)
        a = [0, -0.24365, -0.21325, 0, 0, 0]
        d = [0.1519, 0, 0, 0.11235, 0.08535, 0.0819]

        alpha = [pi/2, zero, zero, pi/2, -pi/2, zero]

        # mass data, no inertia available
        mass = [2, 3.42, 1.26, 0.8, 0.8, 0.35]
        center_of_mass = [
                [0,   -0.02, 0],
                [0.13, 0,    0.1157],
                [0.05, 0,    0.0238],
                [0,    0,    0.01],
                [0,    0,    0.01],
                [0,    0,   -0.02]
            ]
        links = []

        for j in range(6):
            link = RevoluteDH(
                d=d[j],
                a=a[j],
                alpha=alpha[j],
                m=mass[j],
                r=center_of_mass[j],
                G=1
            )
            links.append(link)
    
        super().__init__(
            links,
            name="UR3",
            manufacturer="Universal Robotics",
            keywords=('dynamics', 'symbolic'),
            symbolic=symbolic
        )
    
        # zero angles
        self.addconfiguration("qz", np.array([0, 0, 0, 0, 0, 0]))
        # horizontal along the x-axis
        self.addconfiguration("qr", np.r_[180, 0, 0, 0, 90, 0]*deg)

if __name__ == '__main__':    # pragma nocover

    ur3 = UR3(symbolic=False)
    print(ur3)
    print(ur3.dyntable())
