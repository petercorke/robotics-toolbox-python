import numpy as np
from spatialmath import SE3
import roboticstoolbox as rtb
import timeit
from ansitable import ANSITable, Column

# change for the robot IK under test, must set:
#  * robot, the DHRobot object
#  * T, the end-effector pose
#  * q0, the initial joint angles for solution

robot = rtb.models.DH.Puma560()

q = robot.qn
qd = np.random.rand(len(q))
z = np.zeros((len(q),))

# setup to run timeit
setup = """
from __main__ import robot, T, q0
"""
N = 1000

# setup results table
table = ANSITable(
    Column("Operation", headalign="^", colalign="<"),
    Column("Time (us)", headalign="^", fmt="{:.1f}"),
    border="thick",
)


def measure(statement):
    global table

    t = timeit.timeit(stmt=statement, setup=setup, number=N, globals=globals())

    table.row(statement, t / N * 1e6)


# ------------------------------------------------------------------------- #

setup = """
from __main__ import robot, q, qd, z
"""
measure("robot.jacobe(q)")
J = robot.jacobe(q)
measure("np.linalg.inv(J)")
measure("np.linalg.pinv(J)")
measure("robot.rne(q, qd, qd)")
# measure('robot.rne_python(q, qd, qd)')

table.rule()

measure("robot.gravload(q)")
measure("robot.coriolis(q, qd)")
measure("robot.inertia(q)")
measure("robot.accel(q, qd, z)")

table.rule()

# measure('robot.gravload_x(q)')
# measure('robot.coriolis_x(q, qd)')
# measure('robot.inertia_x(q)')
# measure('robot.accel_x(q, qd, z)')

# pretty print the results
table.print()
# print(table.markdown())
