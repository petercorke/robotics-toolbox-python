#!/usr/bin/env python
"""
@author Micah Huth
"""

import roboticstoolbox as rtb

from roboticstoolbox.backends.VPython import VPython

env = VPython()
env.launch()


#  PUMA560
puma = rtb.models.DH.Puma560()

env.hold()
