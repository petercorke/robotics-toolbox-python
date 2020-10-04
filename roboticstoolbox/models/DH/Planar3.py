"""
@author: Luis Fernando Lara Tobar
@author: Peter Corke
@author: Samuel Drew
"""

from roboticstoolbox import DHRobot, RevoluteDH
import numpy as np


class Planar3(DHRobot):
    """
    Create a planar 3 link robot
    """

    def __init__(self):

        L = [RevoluteDH(a=1),
             RevoluteDH(a=1),
             RevoluteDH(a=1)]

        super().__init__(L, name='Planar 3 link', keywords=('planar',))
        self.addconfiguration("qz", [0, 0, 0])

if __name__ == '__main__':

    robot = Planar3()
    print(robot)