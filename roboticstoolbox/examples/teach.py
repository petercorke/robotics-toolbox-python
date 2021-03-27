#!/usr/bin/env python
"""
@author Jesse Haviland
"""

import roboticstoolbox as rp

# Make a panda robot
panda = rp.models.DH.Panda()

# Init joint to the 'ready' joint angles
panda.q = panda.qr

# Open a plot with the teach panel
e = panda.teach()
