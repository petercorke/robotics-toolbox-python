#!/usr/bin/env python

import numpy as np
from roboticstoolbox.robot.ETS import ETS
from pathlib import Path
import roboticstoolbox as rp


class j2n4s300(ETS):

    def __init__(self):

        mpath = Path(rp.__file__).parent
        fpath = mpath / 'models' / 'xacro' / 'kinova_description' / 'urdf'
        fname = 'j2n4s300_standalone.xacro'

        args = super(j2n4s300, self).urdf_to_ets_args(
            (fpath / fname).as_posix())

        super(j2n4s300, self).__init__(
            args[0],
            name=args[1])

        self.manufacturer = 'Kinova'

        self._qz = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self._qr = np.array([0, 45, 60, 0, 0, 0, 0, 0, 0, 0]) * np.pi/180

    @property
    def qz(self):
        return self._qz

    @property
    def qr(self):
        return self._qr
