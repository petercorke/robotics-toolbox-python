#!/usr/bin/env python

import numpy as np
from roboticstoolbox.robot.ERobot import ERobot
from pathlib import Path
import roboticstoolbox as rp


class j2n4s300(ERobot):

    def __init__(self):

        mpath = Path(rp.__file__).parent
        fpath = mpath / 'models' / 'xacro' / 'kinova_description' / 'urdf'
        fname = 'j2n4s300_standalone.xacro'

        args = super().urdf_to_ets_args(
            (fpath / fname).as_posix())

        super().__init__(
            args[0],
            name=args[1],
            manufacturer = 'Kinova'
            )

        self.addconfiguration("qz", np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
        self.addconfiguration("qr", np.array([0, 45, 60, 0, 0, 0, 0, 0, 0, 0]) * np.pi/180)


if __name__ == '__main__':

    robot = j2n4s300()
    print(robot)