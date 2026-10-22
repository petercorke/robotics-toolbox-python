#!/usr/bin/env python3
"""
@author: Jesse Haviland
"""

from __future__ import annotations

from typing import Any

from roboticstoolbox.robot.Robot import Robot, Robot2


class ERobot(Robot):
    """
    Robot defined by an elementary transform sequence

    This is the same as :class:`~roboticstoolbox.robot.Robot.Robot`, which it
    subclasses and can be used in its place, and is kept for compatibility.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:

        # warn("ERobot is deprecated, use iscollided instead", FutureWarning)

        super().__init__(*args, **kwargs)


# =========================================================================== #


class ERobot2(Robot2):
    def __init__(self, *args: Any, **kwargs: Any) -> None:

        # warn("ERobot2 is deprecated, use iscollided instead", FutureWarning)

        super().__init__(*args, **kwargs)
