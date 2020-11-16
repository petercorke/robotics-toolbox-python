import roboticstoolbox as rtb
from spatialmath import *   # lgtm [py/polluting-import]


#robot = rtb.models.URDF.Puma560()
robot = rtb.models.DH.Puma560()
print(robot)
print(robot.fkine(robot.qz))
# tw, T0 = p560.twists(p560.qn)
# print(tw)
qt = rtb.tools.trajectory.jtraj(robot.qz, robot.qr, 200)
robot.plot(qt.q.T, dt=100, block=True) #movie='puma_sitting.gif')
#swift = rtb.backend.Swift()
#swift.launch()
#swift.add(robot)
#for q in qt.q:
	#robot.q = q
	#swift.step()
