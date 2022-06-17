import numpy as np
from roboticstoolbox import DHRobot, RevoluteDH
from spatialmath import SE3


class UR10e(DHRobot):
    """
    Class that models a Universal Robotics UR10e manipulator

    :param symbolic: use symbolic constants
    :type symbolic: bool

    ``UR10e()`` is an object which models a Universal Robotics UR10e robot and
    describes its kinematic and dynamic characteristics using standard DH
    conventions.

    .. runblock:: pycon

        >>> import roboticstoolbox as rtb
        >>> robot = rtb.models.DH.UR10e()
        >>> print(robot)

    Defined joint configurations are:

    - qz, zero joint angle configuration
    - qr, arm horizontal along x-axis

    .. note::
        - SI units are used.

    :References:

        - `Parameters for calculations of kinematics and dynamics <https://www.universal-robots.com/articles/ur/parameters-for-calculations-of-kinematics-and-dynamics>`_

    :sealso: :func:`UR3e`, :func:`UR5e`

    .. codeauthor:: Meng
    .. sectionauthor:: Peter Corke
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
        a = [0, -0.6127, -0.57155, 0, 0, 0]
        d = [0.1807, 0, 0, 0.17415, 0.11985, 0.11655]

        alpha = [pi / 2, zero, zero, pi / 2, -pi / 2, zero]

        # mass data, no inertia available
        mass = [7.369, 13.051, 3.989, 2.1, 1.98, 0.615]
        center_of_mass = [
            [0.021, 0, 0.027],
            [0.38, 0, 0.158],
            [0.24, 0, 0.068],
            [0.0, 0.007, 0.018],
            [0.0, 0.007, 0.018],
            [0, 0, -0.026],
        ]
        links = []

        for j in range(6):
            link = RevoluteDH(
                d=d[j], a=a[j], alpha=alpha[j], m=mass[j], r=center_of_mass[j], G=1
            )
            links.append(link)

        super().__init__(
            links,
            name="UR10e",
            manufacturer="Universal Robotics",
            keywords=("dynamics", "symbolic"),
            symbolic=symbolic,
        )

        self.qr = np.array([180, 0, 0, 0, 90, 0]) * deg
        self.qz = np.zeros(6)

        self.addconfiguration("qr", self.qr)
        self.addconfiguration("qz", self.qz)


if __name__ == "__main__":  # pragma nocover

    ur10e = UR10e(symbolic=False)
    print(ur10e)
    # print(ur10e.dyntable())
