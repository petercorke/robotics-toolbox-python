import numpy as np
import roboticstoolbox as rtb
import spatialmath as sm


puma = rtb.models.DH.Puma560()
q = puma.qn  # type: ignore

# m1 = puma.manipulability(q, method="asada")

# print(m1)

print(puma.inertia(q))
