import numpy as np
import roboticstoolbox as rtb
import spatialmath as sm


r = rtb.models.Panda()
q = r.qr

m1 = r.jacobm(q)
m2 = r.jacobm(q, axes="trans")
m3 = r.jacobm(q, axes="rot")

print(m1.__repr__())
print(m2.__repr__())
print(m3.__repr__())
