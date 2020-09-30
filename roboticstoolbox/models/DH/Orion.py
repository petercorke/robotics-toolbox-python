#!/usr/bin/env python
# Created by: Aditya Dua
# 5 October 2017

from math import pi
import numpy as np
from roboticstoolbox import DHRobot, RevoluteDH

class Orion5(DHRobot):
    def __init__(self, base=None):

        self.qz = np.matrix([[0, 0, 0, 0, 0, 0]])
        self.scale = 0.01
        # Pre-defined Stances
        # TODO

        # Turret, Shoulder, Elbow, Wrist, Claw
        links = [RevoluteDH(d=0, a=0, alpha=0, offset=0),  # Turret
                 RevoluteDH(d=0.53, a=-0.30309, alpha=0, offset=0),  # Shoulder
                 RevoluteDH(d=0, a=-1.70384, alpha=0, offset=0),  # Elbow
                 RevoluteDH(d=0, a=-1.36307, alpha=0, offset=0),  # Wrist
                 RevoluteDH(d=0, a=0, alpha=0, offset=0),
                 RevoluteDH(d=0, a=0, alpha=0, offset=0)]

        super().__init__(
            links,
            name="Orion 5",
            manufacturer="RAWR Robotics")

if __name__ == '__main__':

    orion = Orion5()
    print(orion)
