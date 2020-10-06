#!/usr/bin/env python
"""
@author Jesse Haviland
"""

# import roboticstoolbox as rp
# import spatialmath as sm
# import numpy as np
# import fcl
# import pybullet as p

# r = rp.models.Panda()
# r.q = r.qr

# r.scollision()



# # obj1 = fcl.Box(1, 1, 1)
# # co1 = fcl.CollisionObject(obj1, fcl.Transform())
# # obj2 = fcl.Box(1, 1, 1)
# # co2 = fcl.CollisionObject(obj2, fcl.Transform())

# t1 = sm.SE3() * sm.SE3.Rx(-1.2) * sm.SE3.Ry(0.8)
# # t2 = sm.SE3(10, 10, 10)
# t2 = sm.SE3(10, 2.7, 5.8) * sm.SE3.Rx(1.2) * sm.SE3.Ry(0.2) * sm.SE3.Rz(2.2)

# obj1 = rp.Shape.Cylinder(radius=0.5, length=1, base=t1)
# obj2 = rp.Shape.Cylinder(radius=0.5, length=1, base=t2)
# # t2 = sm.SE3(0, 10, 0) * sm.SE3.Rx(1.5)

# # tf1 = fcl.Transform(t1.R, t1.t)
# # co1.setTransform(tf1)
# # tf2 = fcl.Transform(t2.R, t2.t)
# # co2.setTransform(tf2)

# request = fcl.DistanceRequest()
# result = fcl.DistanceResult()
# ret = fcl.distance(obj1.co, obj2.co, request, result)
# print(ret)
# print(result.nearest_points)
# np1 = result.nearest_points[0]
# np2 = result.nearest_points[1]

# aTp1 = (t1 * sm.SE3(np1)).t
# bTp2 = (t2 * sm.SE3(np2)).t

# p1Tp2 = bTp2 - aTp1

# d = np.linalg.norm(p1Tp2)
# print(d)

import pybullet as p
import time
import math
import pybullet_data



cid = p.connect(p.SHARED_MEMORY)
if (cid < 0):
  p.connect(p.DIRECT)

# p.setAdditionalSearchPath(pybullet_data.getDataPath())
# p.setPhysicsEngineParameter(numSolverIterations=10)
# p.setTimeStep(1. / 120.)

# logId = p.startStateLogging(p.STATE_LOGGING_PROFILE_TIMINGS, "visualShapeBench.json")

#useMaximalCoordinates is much faster then the default reduced coordinates (Featherstone)
# p.loadURDF("plane100.urdf", useMaximalCoordinates=True)

#disable rendering during creation.
# p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
# p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

#disable tinyrenderer, software (CPU) renderer, we don't use it here
# p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 0)


c1 = p.createCollisionShape(shapeType=p.GEOM_SPHERE, radius=0.1)
c2 = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=[1, 0.5, 0.1])

b1 = p.createMultiBody(baseMass=1,
                    baseInertialFramePosition=[0, 0, 0],
                    baseCollisionShapeIndex=c1,
                    basePosition=[0, 0, 0],
                    useMaximalCoordinates=True)

# p.resetBasePositionAndOrientation(b1, [0, 0, 1], [1, 0, 0, 0])

b2 = p.createMultiBody(baseMass=1,
                    baseInertialFramePosition=[0, 0, 0],
                    baseCollisionShapeIndex=c2,
                    basePosition=[0, 0, 1],
                    useMaximalCoordinates=True)

print(p.getClosestPoints(b1, b2, 2))

# p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
# p.stopStateLogging(logId)
# p.setGravity(0, 0, -10)
# p.setRealTimeSimulation(1)

# colors = [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1], [1, 1, 1, 1]]
# currentColor = 0

# while (1):
#   time.sleep(1./240.)
#   print(p.getClosestPoints(b1, b2, 2))