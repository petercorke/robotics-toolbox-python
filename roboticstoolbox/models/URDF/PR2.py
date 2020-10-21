#!/usr/bin/env python

import roboticstoolbox as rp
from roboticstoolbox.robot.ERobot import ERobot
from pathlib import Path


class PR2(ERobot):

    def __init__(self):

        mpath = Path(rp.__file__).parent
        fpath = mpath / 'models' / 'xacro' / 'pr2_description' / 'robots'
        fname = 'pr2.urdf.xacro'
        tld = mpath / 'models' / 'xacro' / 'pr2_description'

        args = super(PR2, self).urdf_to_ets_args(
            (fpath / fname).as_posix(), tld)

        super(PR2, self).__init__(
            args[0],
            name=args[1])

        self.manufacturer = 'Willow Garage'
