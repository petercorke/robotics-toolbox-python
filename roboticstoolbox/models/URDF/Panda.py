#!/usr/bin/env python

import numpy as np
import roboticstoolbox as rp
from roboticstoolbox.robot.ERobot import ERobot
from pathlib import Path


class Panda(ERobot):

    def __init__(self):

        mpath = Path(rp.__file__).parent
        fpath = mpath / 'models' / 'xacro' / 'franka_description' / 'robots'
        fname = 'panda_arm_hand.urdf.xacro'

        args = super(Panda, self).urdf_to_ets_args(
            (fpath / fname).as_posix())

        super().__init__(
            args[0],
            name=args[1])

        self.manufacturer = 'Franka Emika'
        self.ee_link = self.ets[9]

        self.qdlim = np.array([
            2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100, 3.0, 3.0])

        self.addconfiguration("qz", np.array(
            [0, 0, 0, 0, 0, 0, 0, 0, 0]))
        self.addconfiguration("qr", np.array(
            [0, -0.3, 0, -2.2, 0, 2.0, np.pi/4, 0, 0]))


if __name__ == '__main__':   # pragma nocover

    robot = Panda()
    print(robot)
