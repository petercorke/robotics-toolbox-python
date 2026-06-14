#!/usr/bin/env python
"""
@author: Jesse Haviland
"""

from numpy import ndarray, eye, zeros, copy as npcopy
from spatialmath.base import r2q
from abc import ABC
from collections import UserList

# from roboticstoolbox.robot.ETS import ETS
from typing import Type

from spatialgeometry.geom.SceneNode import SceneNode


class SceneGroup(SceneNode, UserList):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __getitem__(self, i: int) -> SceneNode:
        return self._scene_children[i]

    @property
    def data(self):
        return self._scene_children
