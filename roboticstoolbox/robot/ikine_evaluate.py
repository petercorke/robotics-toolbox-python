import numpy as np
from spatialmath import SE3
import roboticstoolbox as rtb
import timeit
from ansitable import ANSITable, Column

# change for the robot IK under test, must set:
#  * robot, the DHRobot object
#  * T, the end-effector pose
#  * q0, the initial joint angles for solution

# Puma robot case
# robot = rtb.models.DH.Puma560()
# q = robot.qn
# q0 = robot.qz
# T = robot.fkine(q)

# Panda robot case
robot = rtb.models.DH.Panda()
q0 = robot.qr
T = SE3(0.8, 0.2, 0.1) * SE3.OA([0, 1, 0], [0, 0, -1])
# this pose does quite badly, it's from the RTB README.md

# build the list of IK methods to test
ikfuncs = [ 
    robot.ikine_LM,  # Levenberg-Marquadt
    robot.ikine_LMS, # Levenberg-Marquadt (Sugihara)
    robot.ikine_unc, #numerical solution with no constraints 
    robot.ikine_con, # numerical solution with constraints
    robot.ikine_min  # numerical solution 2 with constraints
]
if "ikine_a" in robot:
    ikfuncs.append(robot.ikine_a)    # analytic solution

# setup to run timeit
setup = '''
from __main__ import robot, T, q0
'''
N = 10

# setup results table
table = ANSITable(
    Column("Operation", headalign="^"),
    Column("Time (μs)", headalign="^", fmt="{:.2f}"),
    Column("Error", headalign="^", fmt="{:.3g}"),
    border="thick")

# test the IK methods
for ik in ikfuncs:
    print('Testing:', ik.__name__)
    
    # test the method, don't pass q0 to the analytic function
    if ik.__name__ == "ikine_a":
        sol = ik(T)
        statement = f"sol = robot.{ik.__name__}(T)"
    else:
        sol = ik(T, q0=q0)
        statement = f"sol = robot.{ik.__name__}(T, q0=q0)"

    # print error message if there is one
    if not sol.success:
        print('  failed:', sol.reason)

    # evalute the error
    err = np.linalg.norm(T - robot.fkine(sol.q))
    print('  error', err)

    # evaluate the execution time
    t = timeit.timeit(stmt=statement, setup=setup, number=N)

    # add it to the output table
    table.row(ik.__name__, t/N*1e6, err)

# pretty print the results     
table.print()
