#!/usr/bin/env python

import numpy as np
from roboticstoolbox.robot.ETS import ETS
from pathlib import Path
import roboticstoolbox as rp


class UR10(ETS):

    def __init__(self):

        mpath = Path(rp.__file__).parent
        fpath = mpath / 'models' / 'xacro' / 'ur_description' / 'urdf'
        fname = 'ur10_joint_limited_robot.urdf.xacro'

        args = super(UR10, self).urdf_to_ets_args((fpath / fname).as_posix())

        super(UR10, self).__init__(
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
