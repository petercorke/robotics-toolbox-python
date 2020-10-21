#!/usr/bin/env python

import numpy as np
from roboticstoolbox.robot.ERobot import ERobot
import roboticstoolbox as rp
from pathlib import Path
from math import pi


class Puma560(ERobot):

    def __init__(self):

        mpath = Path(rp.__file__).parent
        fpath = mpath / 'models' / 'xacro' / \
            'puma560_description' / 'urdf'
        fname = 'puma560_robot.urdf.xacro'

        args = super(Puma560, self).urdf_to_ets_args(
            (fpath / fname).as_posix())

        super().__init__(
            args[0],
            name=args[1])

        self.manufacturer = "Unimation"
        # self.ee_link = self.ets[9]

        # zero angles, L shaped pose
        self.addconfiguration("qz", np.array([0, 0, 0, 0, 0, 0]))

        # ready pose, arm up
        self.addconfiguration("qr", np.array([0, pi/2, -pi/2, 0, 0, 0]))

        # straight and horizontal
        self.addconfiguration("qs", np.array([0, 0, -pi/2, 0, 0, 0]))

        # nominal table top picking pose
        self.addconfiguration("qn", np.array([0, pi/4, pi, 0, pi/4, 0]))


if __name__ == '__main__':   # pragma nocover

    robot = Puma560()
    print(robot)
