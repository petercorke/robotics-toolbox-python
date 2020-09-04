#!/usr/bin/env python

import numpy as np
from ropy.robot.ETS import ETS
from pathlib import Path
import ropy as rp


class UR5(ETS):

    def __init__(self):

        mpath = Path(rp.__file__).parent
        fpath = mpath / 'models' / 'xacro' / 'ur' / 'urdf'
        fname = 'ur5_joint_limited_robot.urdf.xacro'
        # abspath = fpath.absolute()

        args = super(UR5, self).urdf_to_ets_args((fpath / fname).as_posix())

        super(UR5, self).__init__(
            args[0],
            name=args[1])

        self.manufacturer = 'Universal Robotics'
        # self.ee_link = self.ets[9]

        self._qz = np.array([0, 0, 0, 0, 0, 0])
        self._qr = np.array([np.pi, 0, 0, 0, np.pi/2, 0])

        for link in self.ets:
            for gi in link.geometry:
                if gi.filename[0] != '/':
                    gi.filename = (fpath / gi.filename).as_posix()

    @property
    def qz(self):
        return self._qz

    @property
    def qr(self):
        return self._qr
