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

        args = super(UR5, self).urdf_to_ets_args((fpath / fname).as_posix())

        super(UR5, self).__init__(
            args[0],
            name=args[1])

        self.manufacturer = 'Universal Robotics'

        self._qz = np.array([0, 0, 0, 0, 0, 0])
        self._qr = np.array([np.pi, 0, 0, 0, np.pi/2, 0])

    @property
    def qz(self):
        return self._qz

    @property
    def qr(self):
        return self._qr
