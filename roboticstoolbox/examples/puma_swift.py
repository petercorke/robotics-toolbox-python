#!/usr/bin/env python
"""
@author John Skinner
"""

import swift
import roboticstoolbox as rp
import spatialmath as sm
import numpy as np

env = swift.Swift()
env.launch(realtime=True)

# Create a puma in the default zero pose
puma = rp.models.Puma560()
puma.q = puma.qz
env.add(puma, show_robot=True, show_collision=False)

dt = 0.05
interp_time = 5
wait_time = 2

# Pass through the reference poses one by one.
# This ignores the robot collisions, and may pass through itself
poses = [puma.qz, puma.rd, puma.ru, puma.lu, puma.ld]
for previous, target in zip(poses[:-1], poses[1:]):
    for alpha in np.linspace(0.0, 1.0, int(interp_time / dt)):
        puma.q = previous + alpha * (target - previous)
        env.step(dt)
    for _ in range(int(wait_time / dt)):
        puma.q = target
        env.step(dt)

# Uncomment to stop the browser tab from closing
env.hold()
