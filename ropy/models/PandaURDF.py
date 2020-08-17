#!/usr/bin/env python

import numpy as np
from ropy.robot.ETS import ETS


class PandaURDF(ETS):

    def __init__(self):

        fpath = 'ropy/models/xarco/panda/robots/panda_arm_hand.urdf.xacro'

        args = super(PandaURDF, self).urdf_to_ets_args(fpath)

        super(PandaURDF, self).__init__(
            args[0],
            name=args[1])

        self.manufacturer = 'Franka Emika'
        self.ee_link = self.ets[8]

        self._qz = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
        self._qr = np.array([0, -0.3, 0, -2.2, 0, 2.0, np.pi/4, 0, 0])

    @property
    def qz(self):
        return self._qz

    @property
    def qr(self):
        return self._qr
