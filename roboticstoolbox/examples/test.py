# # #!/usr/bin/env python
# # """
# # @author Jesse Haviland
# # """

# from roboticstoolbox.backends.Swift import Swift
# import roboticstoolbox as rtb
# import spatialmath as sm
# import numpy as np

# env = Swift()
# env.launch()


# b1 = rtb.Box(
#     scale=[0.60, 1.1, 0.02],
#     base=sm.SE3(2.9, 0, 0.20))

# b2 = rtb.Box(
#     scale=[0.60, 1.1, 0.02],
#     base=sm.SE3(2.9, 0, 0.60))

# b3 = rtb.Box(
#     scale=[0.60, 1.1, 0.02],
#     base=sm.SE3(2.9, 0, 1.00))

# b4 = rtb.Box(
#     scale=[0.60, 1.1, 0.02],
#     base=sm.SE3(2.9, 0, 1.40))

# b5 = rtb.Box(
#     scale=[0.60, 0.02, 1.40],
#     base=sm.SE3(2.9, 0.55, 0.7))

# b6 = rtb.Box(
#     scale=[0.60, 0.02, 1.40],
#     base=sm.SE3(2.9, -0.55, 0.7))

# # Front panel
# t1 = rtb.Box(
#     scale=[0.02, 1.6, 0.6],
#     base=sm.SE3(2.6, 1.4, 0.3)
# )

# # Left side
# t2 = rtb.Box(
#     scale=[0.6, 0.02, 0.6],
#     base=sm.SE3(2.9, 2.2, 0.3)
# )

# # Right side
# t3 = rtb.Box(
#     scale=[0.6, 0.02, 0.6],
#     base=sm.SE3(2.9, 0.6, 0.3)
# )

# # Back panel
# t4 = rtb.Box(
#     scale=[0.02, 1.6, 0.6],
#     base=sm.SE3(3.2, 1.4, 0.3)
# )

# # Bench top
# t5 = rtb.Box(
#     scale=[0.62, 1.12, 0.02],
#     base=sm.SE3(2.9, 1.15, 0.6)
# )

# # Sink bottom
# t6 = rtb.Box(
#     scale=[0.6, 0.5, 0.02],
#     base=sm.SE3(2.9, 1.95, 0.3)
# )

# # Sink Right
# t7 = rtb.Box(
#     scale=[0.6, 0.02, 0.6],
#     base=sm.SE3(2.9, 1.7, 0.3)
# )

# # Bench Obstacle
# t8 = rtb.Box(
#     scale=[0.6, 0.02, 0.3],
#     base=sm.SE3(2.9, 1.1, 0.75)
# )

# collisions = [
#     b1, b2, b3, b4, b5, b6,
#     t1, t2, t3, t4, t5, t6,
#     t7, t8
# ]

# for col in collisions:
#     env.add(col)


# r = rtb.models.Frankie()
# r.q = r.qr
# env.add(r)

# for i in range(1000):
#     r.qd = [0.0, 0.4, 0, 0, 0, 0, 0, 0, 0]
#     env.step(0.05, render=False)

#     if r.collided(b2):
#         print("collided you muppet")

#     r.base = r.links[2]._fk * sm.SE3.Tz(-0.38)
#     r.q[:2] = 0

# env.hold()


# from math import pi
# import roboticstoolbox as rtb
# from spatialmath import SO3, SE3
# import numpy as np
# import pathlib
# import os
# import time

# path = os.path.realpath('.')

# # Launch the simulator Swift
# env = rtb.backends.Swift()
# env.launch()

# path = pathlib.Path(path) / 'rtb-data' / 'rtbdata' / 'data' / 'test-mesh'


# # obj_bug = rtb.Mesh(
# #     filename=str(path / 'scene.gltf'),
# #     scale=[0.001, 0.001, 0.001],
# #     base=SE3(0, 0, 0.45) * SE3.Rx(np.pi/2) * SE3.Ry(np.pi/2)
# # )

# dae = rtb.Mesh(
#     filename=str(path / 'walle.dae'),
#     base=SE3(0, -1.5, 0)
# )

# obj = rtb.Mesh(
#     filename=str(path / 'walle.obj'),
#     base=SE3(0, -0.5, 0)
# )

# glb = rtb.Mesh(
#     filename=str(path / 'walle.glb'),
#     base=SE3(0, 0.5, 0) * SE3.Rz(-np.pi/2)
# )

# ply = rtb.Mesh(
#     filename=str(path / 'walle.ply'),
#     base=SE3(0, 1.5, 0)
# )

# # wrl = rtb.Mesh(
# #     filename=str(path / 'walle.wrl'),
# #     base=SE3(0, 2.0, 0)
# # )

# pcd = rtb.Mesh(
#     filename=str(path / 'pcd.pcd'),
#     base=SE3(0, 0, 1.5) * SE3.Rx(np.pi/2) * SE3.Ry(np.pi/2)
# )


# env.add(dae)
# env.add(obj)
# env.add(glb)
# env.add(ply)
# env.add(pcd)

# time.sleep(2)


# while(True):
#     # env.process_events()
#     env.step(0)
