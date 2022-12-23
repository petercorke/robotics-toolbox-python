import numpy as np
import roboticstoolbox as rtb
import spatialmath as sm


l1 = rtb.Link(ets=rtb.ETS(rtb.ET.Ry()), m=1, r=[0.5, 0, 0], name="l1")
l2 = rtb.Link(
    ets=rtb.ETS(rtb.ET.tx(1)) * rtb.ET.Ry(), m=1, r=[0.5, 0, 0], parent=l1, name="l2"
)
r = rtb.Robot([l1, l2], name="simple 2 link")
q = np.zeros(r.n)

m1 = r.manipulability(q, method="asada")

# print(m1)

# for l in r:
#     print(l.jindex)

print(r.inertia(q))
