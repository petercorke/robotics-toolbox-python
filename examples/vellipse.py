#!/usr/bin/env python
"""
@author Jesse Haviland
"""

import ropy as rp
# import numpy as np

# Make a panda robot
panda = rp.PandaMDH()

# Init joint to the 'ready' joint angles
panda.q = panda.qr
