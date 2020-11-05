#!/usr/bin/env python

from roboticstoolbox.robot.ERobot import ERobot


class PR2(ERobot):

    def __init__(self):

        args = super().urdf_to_ets_args(
            "pr2_description/robots/pr2.urdf.xacro",
            "pr2_description")

        super(PR2, self).__init__(
            args[0],
            name=args[1])

        self.manufacturer = 'Willow Garage'
