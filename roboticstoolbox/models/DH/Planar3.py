"""
@author: Luis Fernando Lara Tobar
@author: Peter Corke
@author: Samuel Drew
"""

from roboticstoolbox import DHRobot, RevoluteDH


class Planar3(DHRobot):
    """
    Class that models a planar 3-link robot

    ``Planar2()`` is a class which models a 3-link planar robot and
    describes its kinematic characteristics using standard DH
    conventions.

    .. runblock:: pycon

        >>> import roboticstoolbox as rtb
        >>> robot = rtb.models.DH.Planar3()
        >>> print(robot)

    Defined joint configurations are:

        - qz, zero angles, all folded up

    .. note::

      - Robot has only 3 DoF.

    .. codeauthor:: Peter Corke
    """

    def __init__(self):

        L = [
                RevoluteDH(a=1),
                RevoluteDH(a=1),
                RevoluteDH(a=1)
            ]

        super().__init__(L, name='Planar 3 link', keywords=('planar',))
        self.addconfiguration("qz", [0, 0, 0])


if __name__ == '__main__':   # pragma nocover

    robot = Planar3()
    print(robot)
