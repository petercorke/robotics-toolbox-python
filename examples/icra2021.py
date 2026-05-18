# -*- coding: utf-8 -*-
# # Not your grandmother’s toolbox– the Robotics Toolbox reinvented for python
# ### Peter Corke and Jesse Haviland
#
# This is the code for the examples in the paper published at ICRA2021.
#

from swift import Swift
import spatialmath.base.symbolic as sym
from roboticstoolbox import ETS as ET
from roboticstoolbox import *
from spatialmath import *
from spatialgeometry import *
from math import pi
import numpy as np

# # III.SPATIAL MATHEMATICS

from spatialmath.base import *

T = transl(0.5, 0.0, 0.0) @ rpy2tr(0.1, 0.2, 0.3, order="xyz") @ trotx(-90, "deg")
print(T)

T = SE3(0.5, 0.0, 0.0) * SE3.RPY([0.1, 0.2, 0.3], order="xyz") * SE3.Rx(-90, unit="deg")
print(T)

T.eul()

T.R

T.plot(color="red", label="2")

UnitQuaternion.Rx(0.3)
UnitQuaternion.AngVec(0.3, [1, 0, 0])

R = SE3.Rx(np.linspace(0, pi / 2, num=100))
len(R)

# # IV. ROBOTICS TOOLBOX
# ## A. Robot models

# robot length values (metres)
d1 = 0.352
a1 = 0.070
a2 = 0.360
d4 = 0.380
d6 = 0.065

robot = DHRobot(
    [
        RevoluteDH(d=d1, a=a1, alpha=-pi / 2),
        RevoluteDH(a=a2),
        RevoluteDH(alpha=pi / 2),
    ],
    name="my IRB140",
)

print(robot)

puma = models.DH.Puma560()
T = puma.fkine([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
sol = puma.ikine_LM(T)
print(sol)

puma.plot(sol.q, block=False)

puma.ikine_a(T, config="lun")


# Puma dimensions (m), see RVC2 Fig. 7.4 for details
l1 = 0.672
l2 = -0.2337
l3 = 0.4318
l4 = 0.0203
l5 = 0.0837
l6 = 0.4318

e = (
    ET.tz(l1)
    * ET.rz()
    * ET.ty(l2)
    * ET.ry()
    * ET.tz(l3)
    * ET.tx(l4)
    * ET.ty(l5)
    * ET.ry()
    * ET.tz(l6)
    * ET.rz()
    * ET.ry()
    * ET.rz()
)

robot = ERobot(e)
print(robot)

panda = models.URDF.Panda()
print(panda)


# ## B. Trajectories

traj = jtraj(puma.qz, puma.qr, 100)
qplot(traj.q)

t = np.arange(0, 2, 0.010)
T0 = SE3(0.6, -0.5, 0.3)
T1 = SE3(0.4, 0.5, 0.2)
Ts = ctraj(T0, T1, t)
len(Ts)
sol = puma.ikine_LM(Ts)
sol.q.shape

# ## C. Symbolic manipulation

phi, theta, psi = sym.symbol("φ, ϴ, ψ")
rpy2r(phi, theta, psi)

q = sym.symbol("q_:6")  # q = (q_1, q_2, ... q_5)
T = puma.fkine(q)

puma = models.DH.Puma560(symbolic=True)
T = puma.fkine(q)
T.t[0]

puma = models.DH.Puma560(symbolic=False)
J = puma.jacob0(puma.qn)
J = puma.jacobe(puma.qn)

# ## D. Differential kinematics

J = puma.jacob0(puma.qr)
np.linalg.matrix_rank(J)

jsingu(J)

H = panda.hessian0(panda.qz)
H.shape

puma.manipulability(puma.qn)
puma.manipulability(puma.qn, method="asada")

puma.manipulability(puma.qn, axes="trans")

panda.jacobm(panda.qr)


# ## E. Dynamics

tau = puma.rne(puma.qn, np.zeros((6,)), np.zeros((6,)))
J = puma.inertia(puma.qn)
C = puma.coriolis(puma.qn, 0.1 * np.ones((6,)))
g = puma.gravload(puma.qn)
qdd = puma.accel(puma.qn, tau, np.zeros((6,)))


# # V. NEW CAPABILITY
# ## B. Collision checking


obstacle = Box([1, 1, 1], base=SE3(1, 0, 0))
iscollision0 = panda.collided(panda.q, obstacle)  # boolean
iscollision1 = panda.links[0].collided(obstacle)

d, p1, p2 = panda.closest_point(panda.q, obstacle)
print(d, p1, p2)
d, p1, p2 = panda.links[0].closest_point(obstacle)
print(d, p1, p2)

# ## C. Interfaces

panda.plot(panda.qr, block=False)

backend = Swift()
backend.launch()  # create graphical world
backend.add(panda)  # add robot to the world
panda.q = panda.qr  # update the robot
backend.step()  # display the world


#
