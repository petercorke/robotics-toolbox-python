from roboticstoolbox import *
from spatialmath import SO2
import numpy as np
# puma = models.DH.Puma560()

# traj = jtraj(puma.qz, puma.qr, 100)
# traj.plot(block=True)

via = SO2(30, unit="deg") * np.array([[-1, 1, 1, -1, -1], [1, 1, -1, -1, 1]])
print(via)
traj0 = mstraj(via.T, dt=0.2, tacc=0.5, qdmax=[2, 1])
xplot(traj0.q[:, 0], traj0.q[:, 1], color="red")
traj0.plot(block=True)