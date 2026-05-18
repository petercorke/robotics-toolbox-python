"""
@author: Tassos Natsakis
"""

import numpy as np
from roboticstoolbox import DHRobot, RevoluteMDH
from spatialmath import SE3


class AL5D(DHRobot):
    """
    Class that models a Lynxmotion AL5D manipulator

    :param symbolic: use symbolic constants
    :type symbolic: bool

    ``AL5D()`` is an object which models a Lynxmotion AL5D robot and
    describes its kinematic and dynamic characteristics using modified DH
    conventions.

    .. runblock:: pycon

        >>> import roboticstoolbox as rtb
        >>> robot = rtb.models.DH.AL5D()
        >>> print(robot)

    Defined joint configurations are:

    - qz, zero joint angle configuration

    .. note::
        - SI units are used.

    :References:

        - 'Reference of the robot <http://www.lynxmotion.com/c-130-al5d.aspx>'_

    .. codeauthor:: Tassos Natsakis
    """  # noqa

    def __init__(self, symbolic=False):

        if symbolic:
            import spatialmath.base.symbolic as sym

            # zero = sym.zero()
            pi = sym.pi()
        else:
            from math import pi

            # zero = 0.0

        # robot length values (metres)
        a = [0, 0.002, 0.14679, 0.17751]
        d = [-0.06858, 0, 0, 0]

        alpha = [pi, pi / 2, pi, pi]
        offset = [pi / 2, pi, -0.0427, -0.0427 - pi / 2]

        # mass data as measured
        # mass = [0.187, 0.044, 0.207, 0.081]

        # center of mass as calculated through CAD model
        center_of_mass = [
            [0.01724, -0.00389, 0.00468],
            [0.07084, 0.00000, 0.00190],
            [0.05615, -0.00251, -0.00080],
            [0.04318, 0.00735, -0.00523],
        ]

        # moments of inertia are practically zero
        moments_of_inertia = [
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ]

        joint_limits = [
            [-pi / 2, pi / 2],
            [-pi / 2, pi / 2],
            [-pi / 2, pi / 2],
            [-pi / 2, pi / 2],
        ]

        links = []

        for j in range(3):
            link = RevoluteMDH(
                d=d[j],
                a=a[j],
                alpha=alpha[j],
                offset=offset[j],
                r=center_of_mass[j],
                I=moments_of_inertia[j],
                G=1,
                B=0,
                Tc=[0, 0],
                qlim=joint_limits[j],
            )
            links.append(link)

        tool = SE3(0.07719, 0, 0)

        super().__init__(
            links,
            name="AL5D",
            manufacturer="Lynxmotion",
            keywords=("dynamics", "symbolic"),
            symbolic=symbolic,
            tool=tool,
        )

        # zero angles
        self.addconfiguration("home", np.array([pi / 2, pi / 2, pi / 2, pi / 2]))


if __name__ == "__main__":  # pragma nocover

    al5d = AL5D(symbolic=False)
    print(al5d)
    # print(al5d.dyntable())
