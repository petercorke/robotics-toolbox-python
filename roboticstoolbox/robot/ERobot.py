#!/usr/bin/env python3
"""
Created on Tue Apr 24 15:48:52 2020
@author: Jesse Haviland
"""

from roboticstoolbox import Robot, Robot2


class ERobot:
    def __init__(self, *args, **kwargs):

        # warn("ERobot is deprecated, use iscollided instead", FutureWarning)

        return Robot(*args, **kwargs)


# =========================================================================== #


class ERobot2:
    def __init__(self, *args, **kwargs):

        # warn("ERobot2 is deprecated, use iscollided instead", FutureWarning)

        return Robot2(*args, **kwargs)
