#!/usr/bin/env python
"""
@author Jesse Haviland
"""

import ropy as rp
import numpy as np

# Make a panda robot
panda = rp.PandaMDH()

# Init joint to the 'ready' joint angles
panda.q = panda.qr

# Make 100 random sets of joint angles
q = np.random.rand(7, 100)

# Plot the joint trajectory with a 50ms delay between configurations
panda.plot(q=q, dt=50, vellipse=True, fellipse=True)
