#!/usr/bin/env python
"""
@author Jesse Haviland
"""

import roboticstoolbox as rp
import spatialmath as sm
import numpy as np
import fcl

# r = rp.models.Panda()
# r.q = r.qr

# r.scollision()



obj1 = fcl.Box(1, 1, 1)
co1 = fcl.CollisionObject(obj1, fcl.Transform())
obj2 = fcl.Box(1, 1, 1)
co2 = fcl.CollisionObject(obj2, fcl.Transform())

t1 = sm.SE3()
t2 = sm.SE3(10, 2.7, 5.8) * sm.SE3.Rx(1.2) * sm.SE3.Ry(0.2) * sm.SE3.Rz(2.2)
# t2 = sm.SE3(0, 10, 0) * sm.SE3.Rx(1.5)

tf1 = fcl.Transform(t1.R, t1.t)
co1.setTransform(tf1)
tf2 = fcl.Transform(t2.R, t2.t)
co2.setTransform(tf2)

request = fcl.DistanceRequest()
result = fcl.DistanceResult()
ret = fcl.distance(co1, co2, request, result)
print(ret)
print(result.nearest_points)
np1 = result.nearest_points[0]
np2 = result.nearest_points[1]

aTp1 = (t1 * sm.SE3(np1)).t
bTp2 = (t2 * sm.SE3(np2)).t

p1Tp2 = bTp2 - aTp1

d = np.linalg.norm(p1Tp2)
print(d)
