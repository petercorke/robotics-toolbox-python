import numpy as np
from roboticstoolbox import DHRobot, RevoluteDH
from spatialmath import SE3


class UR10(DHRobot):
    """
    Class that models a Universal Robotics UR10 manipulator

    :param symbolic: use symbolic constants
    :type symbolic: bool

    ``UR10()`` is an object which models a Unimation Puma560 robot and
    describes its kinematic and dynamic characteristics using standard DH
    conventions.

    .. runblock:: pycon

        >>> import roboticstoolbox as rtb
        >>> robot = rtb.models.DH.UR10()
        >>> print(robot)

    Defined joint configurations are:

    - qz, zero joint angle configuration
    - qr, arm horizontal along x-axis

    .. note::
        - SI units are used.

    :References:

        - `Parameters for calculations of kinematics and dynamics <https://www.universal-robots.com/articles/ur/parameters-for-calculations-of-kinematics-and-dynamics>`_

    :sealso: :func:`UR3`, :func:`UR5`


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
        a = [0, -0.612, -0.5723, 0, 0, 0]
        d = [0.1273, 0, 0, 0.163941, 0.1157, 0.0922]

        alpha = [pi / 2, zero, zero, pi / 2, -pi / 2, zero]

        # mass data
        mass = [7.1, 12.7, 4.27, 2.000, 2.000, 0.365]
        center_of_mass = [
            [0.021, 0, 0.027],
            [0.38, 0, 0.158],
            [0.24, 0, 0.068],
            [0.0, 0.007, 0.018],
            [0.0, 0.007, 0.018],
            [0, 0, -0.026],
        ]

        # inertia matrices for each link
        inertia = [
            np.array(
                [[0.0341, 0, -0.0043], [0, 0.0353, 0.0001], [-0.0043, 0.0001, 0.0216]]
            ),
            np.array(
                [[0.0281, 0.0001, -0.0156], [0.0001, 0.7707, 0], [-0.0156, 0, 0.7694]]
            ),
            np.array(
                [[0.0101, 0.0001, 0.0092], [0.0001, 0.3093, 0], [0.0092, 0, 0.3065]]
            ),
            np.array(
                [[0.0030, -0.0000, 0], [-0.0000, 0.0022, -0.0002], [0, -0.0002, 0.0026]]
            ),
            np.array(
                [[0.0030, -0.0000, 0], [-0.0000, 0.0022, -0.0002], [0, -0.0002, 0.0026]]
            ),
            np.array([[0, 0, 0], [0, 0.0004, 0], [0, 0, 0.0003]]),
        ]

        links = []

        for j in range(6):
            link = RevoluteDH(
                d=d[j],
                a=a[j],
                alpha=alpha[j],
                m=mass[j],
                r=center_of_mass[j],
                G=1,
                I=inertia[j],
            )
            links.append(link)

        super().__init__(
            links,
            name="UR10",
            manufacturer="Universal Robotics",
            keywords=("dynamics", "symbolic"),
            symbolic=symbolic,
        )

        self.qr = np.array([180, 0, 0, 0, 90, 0]) * deg
        self.qz = np.zeros(6)

        self.addconfiguration("qr", self.qr)
        self.addconfiguration("qz", self.qz)


if __name__ == "__main__":  # pragma nocover
    ur10 = UR10(symbolic=False)
    print(ur10)
    # print(ur10.dyntable())
