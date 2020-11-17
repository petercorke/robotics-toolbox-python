"""
@author: Luis Fernando Lara Tobar
@author: Peter Corke
@author: Samuel Drew
"""

from roboticstoolbox import DHRobot, RevoluteDH
from math import pi


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

    def __init__(self):

        L = [
                RevoluteDH(a=1),
                RevoluteDH(a=1)
            ]

        super().__init__(L, name='Planar 2 link', keywords=('planar',))
        self.addconfiguration("qz", [0, 0])
        self.addconfiguration("q1", [0, pi/2])
        self.addconfiguration("q2", [pi/2, -pi/2])


if __name__ == '__main__':   # pragma nocover

    robot = Planar2()
    print(robot)
