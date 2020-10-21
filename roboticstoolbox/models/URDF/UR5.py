#!/usr/bin/env python

import numpy as np
from roboticstoolbox.robot.ERobot import ERobot
from pathlib import Path
import roboticstoolbox as rp


class UR5(ERobot):

    def __init__(self):

        mpath = Path(rp.__file__).parent
        fpath = mpath / 'models' / 'xacro' / 'ur_description' / 'urdf'
        fname = 'ur5_joint_limited_robot.urdf.xacro'

        args = super().urdf_to_ets_args((fpath / fname).as_posix())

        super().__init__(
                args[0],
                name=args[1],
                manufacturer='Universal Robotics'
            )

        self.addconfiguration("qz", np.array([0, 0, 0, 0, 0, 0]))
        self.addconfiguration("qr", np.array([np.pi, 0, 0, 0, np.pi/2, 0]))


if __name__ == '__main__':   # pragma nocover

    robot = UR5()
    print(robot)
