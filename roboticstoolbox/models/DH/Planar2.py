"""
@author: Luis Fernando Lara Tobar
@author: Peter Corke
@author: Samuel Drew
"""

from roboticstoolbox import DHRobot, RevoluteDH


class Planar2(DHRobot):
    """
    Create a planar 2 link robot
    """

    def __init__(self):

        L = [RevoluteDH(a=1),
             RevoluteDH(a=1)]

        super().__init__(L, name='Planar 2 link', keywords=('planar',))
        self.addconfiguration("qz", [0, 0])


if __name__ == '__main__':   # pragma nocover

    robot = Planar2()
    print(robot)
