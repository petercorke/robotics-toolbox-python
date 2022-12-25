import numpy as np
from roboticstoolbox import DHRobot, RevoluteDH
from spatialmath import SE3


class UR3e(DHRobot):
    """
    Class that models a Universal Robotics UR3e manipulator

    :param symbolic: use symbolic constants
    :type symbolic: bool

    ``UR3e()`` is an object which models a Universal Robotics UR3e robot and
    describes its kinematic and dynamic characteristics using standard DH
    conventions.

    .. runblock:: pycon

        >>> import roboticstoolbox as rtb
        >>> robot = rtb.models.DH.UR3e()
        >>> print(robot)

    Defined joint configurations are:

    - qz, zero joint angle configuration
    - qr, arm horizontal along x-axis

    .. note::
        - SI units are used.

    :References:

        - `Parameters for calculations of kinematics and dynamics <https://www.universal-robots.com/articles/ur/parameters-for-calculations-of-kinematics-and-dynamics>`_

    :sealso: :func:`UR5e`, :func:`UR10e`


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
        a = [0, -0.24355, -0.2132, 0, 0, 0]
        d = [0.15185, 0, 0, 0.13105, 0.08535, 0.0921]

        alpha = [pi / 2, zero, zero, pi / 2, -pi / 2, zero]

        # mass data, no inertia available
        mass = [1.98, 3.4445, 1.437, 0.871, 0.805, 0.261]
        center_of_mass = [
            [0, -0.02, 0],
            [0.13, 0, 0.1157],
            [0.05, 0, 0.0238],
            [0, 0, 0.01],
            [0, 0, 0.01],
            [0, 0, -0.02],
        ]
        links = []

        for j in range(6):
            link = RevoluteDH(
                d=d[j], a=a[j], alpha=alpha[j], m=mass[j], r=center_of_mass[j], G=1
            )
            links.append(link)

        super().__init__(
            links,
            name="UR3e",
            manufacturer="Universal Robotics",
            keywords=("dynamics", "symbolic"),
            symbolic=symbolic,
        )

        self.qr = np.array([180, 0, 0, 0, 90, 0]) * deg
        self.qz = np.zeros(6)

        self.addconfiguration("qr", self.qr)
        self.addconfiguration("qz", self.qz)


if __name__ == "__main__":  # pragma nocover

    ur3e = UR3e(symbolic=False)
    print(ur3e)
    # print(ur3e.dyntable())
