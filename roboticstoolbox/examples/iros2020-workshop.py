import numpy as np

import roboticstoolbox as rtb
from spatialmath.base import *
from spatialmath import SE3, Twist3
puma = rtb.models.DH.Puma560()
# print(puma)
# #puma.plot(puma.qz)

# q = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
# T = puma.fkine(q)
# q = puma.ikine(T)[0]
# print(q)
# # q = puma.ikine_sym(T)

# puma.maniplty(q)

# J = puma.jacob0(q)
# print(J)


# rtb.tools.models()

# import roboticstoolbox.ETS3 as ETS
# ets = ETS.tx() * ETS.rx() * ETS.tx() * …

# robot = Robot(ets)
# np.set_printoptions(
# linewidth=100, formatter={
# 'float': lambda x: f"{x:8.4g}" if x > 1e-10 else f"{0:8.4g}"})

q = puma.qn
qd = np.zeros((6,))
qdd = np.zeros((6,))
tau = np.zeros((6,))
qdd = puma.accel(q, qd, tau)
print(qdd)

# M = puma.inertia(q)
# print(M)
# C = puma.coriolis(q, qd)
# print(C)
# g = puma.gravload(q)
# print(g)

# links = [
# ELink(name=“l1”, parent=“base”, joint=“rx”, jindex=0, transform=SE3(
# RevoluteDH
# robot = DHRobot(…

# a = SE2
# a

# b = SO3
# b
# a * b

# skew([1, 2, 3])
# skewa([1, 2, 3, 4, 5, 6])

# T = SE3.Rand()
# tw = Twist3(T)
# tw
# tw.v
# tw.w
# tw.pitch()
# tw.pole()

# tw.exp(0)
# tw.exp(1)
# tw.exp([0, 1])

# Twist3.R([1, 0, 0], [0, 0, 0])
# Twist3.P([0, 1, 0])

# line = tw.line()
# type(line)

# sm.geom3.plotvol([-5 5 -5 5 -5 5])
# line.plot('k:')


# S,T0 = puma.twists()

# qn = np.random.rand((7,))
# S.exp(puma.qn)

# S.exp(qn).prod() * T0

# puma.fkine(qn)


# axisLines = tw.line()
# print(axisLines)
# puma.plot(qn)
# axisLines.plot('k:')
