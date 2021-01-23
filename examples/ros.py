#!/usr/bin/env python
"""
@author Jesse Haviland
"""

import roboticstoolbox as rtb
import spatialmath as sm
import numpy as np
import os

# Launch the simulator Swift
env = rtb.backends.ROS()
env.launch(ros_master_uri='http://localhost:11311')

os.system('rostopic list')

# # Create a Panda robot object
# panda = rtb.models.Panda()

# # Set joint angles to ready configuration
# panda.q = panda.qr

# # Add the Panda to the simulator
# env.add(panda)


