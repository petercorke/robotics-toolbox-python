from roboticstoolbox import Bicycle, RandomPath, VehiclePolygon, VehicleIcon
from spatialmath import *   # lgtm [py/polluting-import]
from math import pi

dim = 10

# v = VehiclePolygon()
anim = VehicleIcon('greycar', scale=2)

veh = Bicycle(
    animation=anim,
    control=RandomPath(
        dim=dim),
    dim=dim,
    verbose=False)
print(veh)

# odo = veh.step(1, 0.3)
# print(odo)

# print(veh.x)

# print(veh.f([0, 0, 0], odo))

# def control(v, t, x):
#     goal = (6,6)
#     goal_heading = atan2(goal[1]-x[1], goal[0]-x[0])
#     d_heading = base.angdiff(goal_heading, x[2])
#     v.stopif(base.norm(x[0:2] - goal) < 0.1)

#     return (1, d_heading)

p = veh.run(1000, plot=True)
print(p)
