# #!/usr/bin/env python
# """
# @author Jesse Haviland
# """

from math import pi
import roboticstoolbox as rtb
from spatialmath import SO3, SE3
import numpy as np
import pathlib
import os
import time

path = os.path.realpath('.')

# Launch the simulator Swift
env = rtb.backends.Swift()
env.launch()

path = pathlib.Path(path) / 'rtb-data' / 'rtbdata' / 'data' / 'test-mesh'


# obj_bug = rtb.Mesh(
#     filename=str(path / 'scene.gltf'),
#     scale=[0.001, 0.001, 0.001],
#     base=SE3(0, 0, 0.45) * SE3.Rx(np.pi/2) * SE3.Ry(np.pi/2)
# )

dae = rtb.Mesh(
    filename=str(path / 'walle.dae'),
    base=SE3(0, -1.5, 0)
)

obj = rtb.Mesh(
    filename=str(path / 'walle.obj'),
    base=SE3(0, -0.5, 0)
)

glb = rtb.Mesh(
    filename=str(path / 'walle.glb'),
    base=SE3(0, 0.5, 0) * SE3.Rz(-np.pi/2)
)

ply = rtb.Mesh(
    filename=str(path / 'walle.ply'),
    base=SE3(0, 1.5, 0)
)

# wrl = rtb.Mesh(
#     filename=str(path / 'walle.wrl'),
#     base=SE3(0, 2.0, 0)
# )

pcd = rtb.Mesh(
    filename=str(path / 'pcd.pcd'),
    base=SE3(0, 0, 1.5) * SE3.Rx(np.pi/2) * SE3.Ry(np.pi/2)
)


env.add(dae)
env.add(obj)
env.add(glb)
env.add(ply)
env.add(pcd)

time.sleep(2)

while(True):
    # env.process_events()
    env.step(0)
