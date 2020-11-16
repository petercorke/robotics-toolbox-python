import roboticstoolbox as rtb
from spatialmath import *   # lgtm [py/polluting-import]


p560 = rtb.models.DH.Puma560()
print(p560)
tw, T0 = p560.twists(p560.qn)
print(tw)
qt = rtb.tools.trajectory.jtraj(p560.qr, p560.qz, 50)
p560.plot(qt.q.T, dt=100, movie='puma_sitting.gif')
