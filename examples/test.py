#!/usr/bin/env python
"""
@author Jesse Haviland
"""

import roboticstoolbox as rtb
import spatialmath as sm
import numpy as np
import qpsolvers as qp
import time

# Launch the simulator Swift
env = rtb.backend.Swift()
env.launch()

# Create a Panda robot object
puma = rtb.models.Puma560()

# Set joint angles to ready configuration
puma.q = puma.qr

# Add the puma to the simulator
env.add(puma)
