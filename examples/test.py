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
    base=sm.SE3(0, 0, 1.3)
)
# g1.v = [0, 0, 0, 0.4, 0, 0]

g2 = rtb.Mesh(
    filename=str(path / 'gimbal-ring2.stl'),
    base=sm.SE3(0, 0, 1.3) * sm.SE3.Rz(np.pi/2)
)
g2.v = [0, 0, 0, 0.4, 0.0, 0]

g3 = rtb.Mesh(
    filename=str(path / 'gimbal-ring3.stl'),
    base=sm.SE3(0, 0, 1.3)
)
g3.v = [0, 0, 0, 0.4, 0, 0]

env.add(g1)
env.add(g2)
env.add(g3)


panda = rtb.models.Panda()
panda.q = panda.qr
env.add(panda)

j = 0


def set_joint(j, x):
    print(j)
    panda.q[j] = x


for link in panda.links:
    if link.isjoint:

        env.add_slider(
            lambda x, j=j: set_joint(j, x),
            min=link.qlim[0], max=link.qlim[1],
            step=0.01, value=panda.q[j],
            desc='Panda Joint ' + str(j))
        
        j += 1

while(True):
    env.process_events()
    env.step(0.05)
