#!/usr/bin/env python

import numpy as np
import roboticstoolbox as rp
from roboticstoolbox.robot.ERobot import ERobot
from pathlib import Path


class wx200(ERobot):

    def __init__(self):

        mpath = Path(rp.__file__).parent
        fpath = mpath / 'models' / 'xacro' / 'interbotix_descriptions' / 'urdf'
        fname = 'wx200.urdf.xacro'

        args = super().urdf_to_ets_args(
            (fpath / fname).as_posix())

        super().__init__(
                args[0],
                name=args[1],
                manufacturer='Interbotix'
            )

        self.addconfiguration(
            "qz", np.array([0, 0, 0, 0, 0, 0, 0, 0]))
        self.addconfiguration(
            "qr", np.array([0, -0.3, 0, -2.2, 0, 2.0, np.pi/4, 0]))


if __name__ == '__main__':   # pragma nocover

    robot = wx200()
    print(robot)
