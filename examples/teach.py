#!/usr/bin/env python
"""
@author Jesse Haviland
"""

import ropy as rp

# Make a panda robot
panda = rp.PandaMDH()

# Init joint to the 'ready' joint angles
panda.q = panda.qr

# Open a plot with the teach panel
e = panda.teach()
