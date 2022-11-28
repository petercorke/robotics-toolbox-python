#!/usr/bin/env python3
"""
Created on Tue Apr 24 15:48:52 2020
@author: Jesse Haviland
"""

import roboticstoolbox as rtb


class ERobot:
    def __init__(self, *args, **kwargs):

        # warn("ERobot is deprecated, use iscollided instead", FutureWarning)

        return rtb.Robot(*args, **kwargs)


# =========================================================================== #


class ERobot2:
    def __init__(self, *args, **kwargs):

        # warn("ERobot2 is deprecated, use iscollided instead", FutureWarning)

        return rtb.Robot2(*args, **kwargs)
