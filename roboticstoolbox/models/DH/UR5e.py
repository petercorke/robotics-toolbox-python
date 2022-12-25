import numpy as np
from roboticstoolbox import DHRobot, RevoluteDH
from spatialmath import SE3


class UR5e(DHRobot):
    """
    Class that models a Universal Robotics UR5e manipulator

    :param symbolic: use symbolic constants
    :type symbolic: bool

    ``UR5e()`` is an object which models a Universal Robotics UR5e robot and
    describes its kinematic and dynamic characteristics using standard DH
    conventions.

    .. runblock:: pycon

        >>> import roboticstoolbox as rtb
        >>> robot = rtb.models.DH.UR5e()
        >>> print(robot)

    Defined joint configurations are:

    - qz, zero joint angle configuration
    - qr, arm horizontal along x-axis

    .. note::
        - SI units are used.

    :References:

        - `Parameters for calculations of kinematics and dynamics <https://www.universal-robots.com/articles/ur/parameters-for-calculations-of-kinematics-and-dynamics>`_

    :sealso: :func:`UR3e`, :func:`UR10e`

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
        a = [0, -0.425, -0.3922, 0, 0, 0]
        d = [0.1625, 0, 0, 0.1333, 0.0997, 0.0996]

        alpha = [pi / 2, zero, zero, pi / 2, -pi / 2, zero]

        # mass data, no inertia available
        mass = [3.761, 8.058, 2.846, 1.37, 1.3, 0.365]
        center_of_mass = [
            [0, -0.02561, 0.00193],
            [0.2125, 0, 0.11336],
            [0.15, 0, 0.0265],
            [0, -0.0018, 0.01634],
            [0, 0.0018, 0.01634],
            [0, 0, -0.001159],
        ]
        links = []

        for j in range(6):
            link = RevoluteDH(
                d=d[j], a=a[j], alpha=alpha[j], m=mass[j], r=center_of_mass[j], G=1
            )
            links.append(link)

        super().__init__(
            links,
            name="UR5e",
            manufacturer="Universal Robotics",
            keywords=("dynamics", "symbolic"),
            symbolic=symbolic,
        )

        self.qr = np.array([180, 0, 0, 0, 90, 0]) * deg
        self.qz = np.zeros(6)

        self.addconfiguration("qr", self.qr)
        self.addconfiguration("qz", self.qz)


if __name__ == "__main__":  # pragma nocover

    ur5e = UR5e(symbolic=False)
    print(ur5e)
    # print(ur5e.dyntable())
