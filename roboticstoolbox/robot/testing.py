from PoERobot import PoERobot, PoERevolute, PoEPrismatic
from spatialmath import SE3
from spatialmath.base import trnorm
from roboticstoolbox import Robot, ETS, ET
import numpy as np
np.set_printoptions(2)


# Book example
link1 = PoERevolute([0, 0, 1], [0, 0, 0])
link2 = PoERevolute([0, 0, 1], [0.5, 0, 0])
TE0 = SE3.Tx(2)

pr = PoERobot([link1, link2], TE0)

ets = pr.ets()
er = Robot(ets)

q = [0, 0]
q = [np.pi/2, -np.pi/2]
print("RR robot - book example")
print(er.fkine(q))
print(np.isclose(pr.fkine(q), er.fkine(q)))
print("--- jacob0:")
print(np.isclose(pr.jacob0(q), er.jacob0(q)))
print("--- jacobe:")
print(np.isclose(pr.jacobe(q), er.jacobe(q)))

print("----------")
print("")
print("----------")


# RRPR
link1 = PoERevolute([0, 0, 1], [0, 0, 0])
link2 = PoERevolute([0, 1, 0], [0, 0, 0.2])
link3 = PoEPrismatic([0, 1, 0])
link4 = PoERevolute([0, -1, 0], [0.2, 0, 0.5])
TE0 = SE3(np.array([[1, 0, 0, 0.3], [0, 0, -1, 0], [0, 1, 0, 0.5], [0, 0, 0, 1]]))

pr = PoERobot([link1, link2, link3, link4], TE0)
er = Robot(pr.ets())

q = [0, 0, 0, 0]
q = [np.pi/7, -np.pi/5, 0.3, -np.pi/3]

print("RRPR robot")
print(er.fkine(q))
print(np.isclose(pr.fkine(q), er.fkine(q)))
print("--- jacob0:")
print(np.isclose(pr.jacob0(q), er.jacob0(q)))
print("--- jacobe:")
print(np.isclose(pr.jacobe(q), er.jacobe(q)))

print("----------")
print("")
print("----------")


# 3RP (arbitrary structure)
link1 = PoERevolute([0, 0, 1], [0, 0, 0])

w = [-0.635, 0.495, 0.592]
w = w / np.linalg.norm(w)
p = [-0.152, -0.023, -0.144]
link2 = PoERevolute(w, p)

w = [-0.280, 0.790, 0.544]
w = w / np.linalg.norm(w)
p = [-0.300, -0.003, -0.150]
link3 = PoERevolute(w, p)

w = [-0.280, 0.790, 0.544]
w = w / np.linalg.norm(w)
link4 = PoEPrismatic(w)

TE0 = np.array([[0.2535, -0.5986, 0.7599, 0.2938], [-0.8063, 0.3032, 0.5078, -0.0005749], [-0.5344, -0.7414, -0.4058, 0.08402], [0, 0, 0, 1]])
TE0 = SE3(trnorm(TE0))

pr = PoERobot([link1, link2, link3, link4], TE0)
er = Robot(pr.ets())

q = [0, 0, 0, 0]
q = [np.pi/2, np.pi/7, np.pi/5, 0.58]
q = [1, -2, -1, 0.58]

print("3RP robot - arbitrary structure")
print(er.fkine(q))
print(np.isclose(pr.fkine(q), er.fkine(q)))
print("--- jacob0:")
print(np.isclose(pr.jacob0(q), er.jacob0(q)))
print("--- jacobe:")
print(np.isclose(pr.jacobe(q), er.jacobe(q)))

