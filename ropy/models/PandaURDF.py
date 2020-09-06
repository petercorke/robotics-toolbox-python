#!/usr/bin/env python

import numpy as np
from ropy.robot.ETS import ETS
from pathlib import Path


class PandaURDF(ETS):

    def __init__(self):

        fpath = Path('ropy') / 'models' / 'xacro' / 'panda' / 'robots'
        fname = 'panda_arm_hand.urdf.xacro'
        abspath = fpath.absolute()

        args = super(PandaURDF, self).urdf_to_ets_args(
            (abspath / fname).as_posix())

        super(PandaURDF, self).__init__(
            args[0],
            name=args[1])

        self.manufacturer = 'Franka Emika'
        self.ee_link = self.ets[9]

        self._qz = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
        self._qr = np.array([0, -0.3, 0, -2.2, 0, 2.0, np.pi/4, 0, 0])
        # self._qz = np.array([0, 0, 0, 0, 0, 0, 0])
        # self._qr = np.array([0, -0.3, 0, -2.2, 0, 2.0, np.pi/4])

        for link in self.ets:
            for gi in link.geometry:
                if gi.filename[0] != '/':
                    gi.filename = (abspath / gi.filename).as_posix()
            # print(link.name)
            # print(link.geometry)

    @property
    def qz(self):
        return self._qz

    @property
    def qr(self):
        return self._qr
