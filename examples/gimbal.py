# #!/usr/bin/env python
# """
# @author Jesse Haviland
# """

import roboticstoolbox as rtb
import spatialmath as sm
import numpy as np
import pathlib
import os

path = os.path.realpath('.')

# Launch the simulator Swift
env = rtb.backends.Swift()
env.launch()

path = pathlib.Path(path) / 'roboticstoolbox' / 'data'

g1 = rtb.Mesh(
    filename=str(path / 'gimbal-ring1.stl'),
    base=sm.SE3(0, 0, 1.3),
    color=[34, 143, 201]
)
# g1.v = [0, 0, 0, 0.4, 0, 0]

g2 = rtb.Mesh(
    filename=str(path / 'gimbal-ring2.stl'),
    base=sm.SE3(0, 0, 1.3),
    color=[31, 184, 72]
)
# g2.v = [0, 0, 0, 0.4, 0.0, 0]

g3 = rtb.Mesh(
    filename=str(path / 'gimbal-ring3.stl'),
    base=sm.SE3(0, 0, 1.3),
    color=[240, 103, 103]
)
# g3.v = [0, 0, 0, 0.4, 0, 0]

env.add(g1)
env.add(g2)
env.add(g3)


def set_one(x):
    g1.base = sm.SE3(0, 0, 1.3) * sm.SE3.Rz(np.deg2rad(float(x)))

def set_two(x):
    g2.base = sm.SE3(0, 0, 1.3) * sm.SE3.Ry(np.deg2rad(float(x)))

def set_three(x):
    g3.base = sm.SE3(0, 0, 1.3) * sm.SE3.Rx(np.deg2rad(float(x)))
    # g2.base = sm.SE3(0, 0, 1.3) * sm.SE3.Rz(np.pi/2) * sm.SE3.Ry(-np.deg2rad(float(x)))
    # g2.base = g3.base 


env.add_slider(
    set_one,
    min=-180, max=180,
    step=1, value=0,
    desc='Ring One', unit='&#176;')

env.add_slider(
    set_two,
    min=-180, max=180,
    step=1, value=0,
    desc='Ring Two', unit='&#176;')

env.add_slider(
    set_three,
    min=-180, max=180,
    step=1, value=0,
    desc='Ring Three', unit='&#176;')

while(True):
    env.process_events()
    env.step(0)
