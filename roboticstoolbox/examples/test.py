import numpy as np
import roboticstoolbox as rtb
import spatialmath as sm
import fknm


r = rtb.models.Panda()
r.q = r.qr

Tep = r.fkine(r.q) * sm.SE3.Rx(0.1) * sm.SE3.Tx(0.2)

a = fknm.IK(r.ets()._fknm, Tep)

print(a)

