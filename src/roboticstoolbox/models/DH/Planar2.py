"""
@author: Luis Fernando Lara Tobar
@author: Peter Corke
@author: Samuel Drew
"""

from roboticstoolbox import DHRobot, RevoluteDH
import numpy as np


class Planar2(DHRobot):
    """
    Class that models a planar 2-link robot

    ``Planar2()`` is a class which models a 2-link planar robot and
    describes its kinematic characteristics using standard DH
    conventions.

    .. runblock:: pycon

        >>> import roboticstoolbox as rtb
        >>> robot = rtb.models.DH.Planar2()
        >>> print(robot)

    Defined joint configurations are:

        - qz, zero angles, all folded up
        - q1, links are horizontal and vertical respectively
        - q2, links are vertical and horizontal respectively

    .. note::

      - Robot has only 2 DoF.

    .. codeauthor:: Peter Corke
    """

    def __init__(self, symbolic=False):

        if symbolic:
            import spatialmath.base.symbolic as sym

            zero = sym.zero()
            pi = sym.pi()
            a1, a2 = sym.symbol("a1 a2")  # type: ignore
        else:
            from math import pi

            zero = 0.0
            a1 = 1
            a2 = 1

        L = [RevoluteDH(a=a1, alpha=zero), RevoluteDH(a=a2, alpha=zero)]

        super().__init__(L, name="Planar 2 link", keywords=("planar",))

        self.qr = np.array([0, pi / 2])
        self.qz = np.zeros(2)

        self.addconfiguration("qr", self.qr)
        self.addconfiguration("qz", self.qz)

        self.addconfiguration_attr("q1", np.array([0, pi / 2]))
        self.addconfiguration_attr("q2", np.array([pi / 2, -pi / 2]))


if __name__ == "__main__":  # pragma nocover

    robot = Planar2()
    print(robot)
