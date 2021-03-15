# #!/usr/bin/env python
# """
# @author Jesse Haviland
# """

from roboticstoolbox.backends import Swift
from math import pi
import roboticstoolbox as rtb
from spatialmath import SO3, SE3
import numpy as np
import pathlib
import os

path = os.path.realpath('.')

env = Swift.Swift()
env.launch()

path = rtb.path_to_datafile('data')


g1 = rtb.Box(
    base=SE3(-1, 0, 0.5),
    scale=[0.1, 0.2, 0.3],
    color=[0.9, 0.9, 0.9, 1]
)
# g1.v = [0, 0, 0, 0.4, 0, 0]


env.add(g1)

env.hold()
