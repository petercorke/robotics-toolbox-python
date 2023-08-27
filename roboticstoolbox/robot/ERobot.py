#!/usr/bin/env python3
"""
@author: Jesse Haviland
"""

from roboticstoolbox.robot.Robot import Robot, Robot2


class ERobot(Robot):
    def __init__(self, *args, **kwargs):

        # warn("ERobot is deprecated, use iscollided instead", FutureWarning)

        super().__init__(*args, **kwargs)


# =========================================================================== #


class ERobot2(Robot2):
    def __init__(self, *args, **kwargs):

        # warn("ERobot2 is deprecated, use iscollided instead", FutureWarning)

        super().__init__(*args, **kwargs)
