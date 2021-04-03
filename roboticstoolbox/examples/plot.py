#!/usr/bin/env python
"""
@author Jesse Haviland
"""

import roboticstoolbox as rp
import numpy as np

# Make a panda robot
panda = rp.models.DH.Panda()

# Init joint to the 'ready' joint angles
panda.q = panda.qr

# Make 100 random sets of joint angles
q = np.random.rand(100, 7)

# Plot the joint trajectory with a 50ms delay between configurations
panda.plot(q=q, backend='pyplot', dt=0.050, vellipse=True, fellipse=True)
# panda.plot(q=q, backend='swift', dt=0.050, vellipse=False, fellipse=False)
