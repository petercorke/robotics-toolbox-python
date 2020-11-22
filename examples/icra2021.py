from math import pi
import numpy as np
from spatialmath.base import *
from spatialmath import *
import roboticstoolbox as rtb

T = transl(0.5, 0.0, 0.0) \
@ rpy2tr(0.1, 0.2, 0.3, order='xyz') \
@ trotx(-90, 'deg')
print(T)

T = SE3(0.5, 0.0, 0.0) \
* SE3.RPY([0.1, 0.2, 0.3], order='xyz') \
* SE3.Rx(-90, unit='deg')
print(T)
      
T.eul()
T.R

T.plot(color='red', label='2')

UnitQuaternion.Rx(0.3)
UnitQuaternion.AngVec(0.3, [1, 0, 0])
R = SE3.Rx(np.linspace(0, pi/2, num=100))
len(R)
robot = rtb.DHRobot(
[
rtb.RevoluteDH(d=d1, a=a1, alpha=-pi/2), 
rtb.RevoluteDH(a=a2), 
rtb.RevoluteDH(alpha=pi/2),
], name="my IRB140"
)

puma = rtb.models.DH.Puma560()
T = puma.fkine([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
q, *_ = puma.ikine(T)
puma.plot(q)
puma.ikine_a(T, config="lun")
e = ET.tz(l1) * ET.rz() * ET.ty(l2) * ET.ry() \
* ET.tz(l3) * ET.tx(l6) * ET.ty(l4) * ET.ry() \
* ET.tz(l5) * ET.rz() * ET.ry() * ET.rz()
# robot = SerialLink(ETS String)
# panda = Panda()
T = robot.fkine(panda.qz)
panda.plot(qz)
q = panda.ikine(T)
traj = rtb.jtraj(puma.qz, puma.qr, 100)
rtb.qplot(traj.j)
t = np.arange(0, 2, 0.010)
T0 = SE3(0.6, -0.5, 0.0)
T1 = SE3(0.4, 0.5, 0.2)
Ts = ctraj(T0, T1, t)
len(Ts)
qs, *_ = puma.ikine(Ts)
qs.shape

import roboticstoolbox.base.symbolics as sym
phi, theta, psi = sym.symbol('phi, theta, psi')
rpy2r(phi, theta, psi)

q = sym.symbol("q_:6") # q = (q_1, q_2, ... q_5)
T = puma.fkine(q)
puma = rtb.models.DH.Puma560(symbolic=True)
T = puma.fkine(q)
T.t[0]

J = puma.jacob0(q)
J = puma.jacobe(q)
J = puma.jacob0(puma.qr)
np.linalg.matrix_rank(J)

jsingu(J)
H = puma.hessian0(q)
H = puma.hessiane(q)
m = puma.manipulability(q)
m = puma.manipulability(q, "asada")
J = puma.manipulability(q, axes="trans")
Jm = puma.manipm(q, J, H)
tau = puma.rne(puma.qn, np.zeros((6,)), np.zeos((6,)))
J = puma.inertia(q)
C = puma.coriolis(q, qd)
g = puma.gravload(q)
qdd = puma.accel(q, tau, qd)
q = puma.fdyn(q, ...)
# puma.plot(q)
# pyplot = roboticstoolbox.backends.PyPlot()
# pyplot.launch()
# pyplot.add(puma)
# puma.q = q
# puma.step()