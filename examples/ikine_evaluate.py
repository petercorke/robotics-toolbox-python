import numpy as np
from spatialmath import SE3
import roboticstoolbox as rtb
import timeit
from ansitable import ANSITable, Column

# change for the robot IK under test, must set:
#  * robot, the DHRobot object
#  * T, the end-effector pose
#  * q0, the initial joint angles for solution

example = 'puma'  # 'panda'

if example == 'puma':
    # Puma robot case
    robot = rtb.models.DH.Puma560()
    q = robot.qn
    q0 = robot.qz
    T = robot.fkine(q)
elif example == 'panda':
    # Panda robot case
    robot = rtb.models.DH.Panda()
    T = SE3(0.7, 0.2, 0.1) * SE3.OA([0, 1, 0], [0, 0, -1])
    q0 = robot.qz


# build the list of IK methods to test
# it's a tuple:
#  - the method to execute
#  - the name for the results table
#  - the statement to execute for timeit
ikfuncs = [ 
    (robot.ikine_LM,  # Levenberg-Marquadt
        "ikine_LM",
        "sol = robot.ikine_LM(T, q0)"
    ),
    (robot.ikine_LMS, # Levenberg-Marquadt (Sugihara)
        "ikine_LMS",
        "sol = robot.ikine_LMS(T, q0)"
    ),
    (robot.ikine_min, #numerical solution with no constraints
        "ikine_min(qlim=False)",
        "sol = robot.ikine_min(T, q0)"
    ),
    (lambda T, q0: robot.ikine_min(T, q0, qlim=True), #numerical solution with constraints
        "ikine_min(qlim=True)",
        "sol = robot.ikine_min(T, q0, qlim=True)"
    ),
    # (robot.ikine_mmc, #numerical solution with no constraints
    #     "ikine_min(qlim=False)",
    #     "sol = robot.ikine_min(T, q0)"
    # ),
]
if hasattr(robot, "ikine_a"):
    a =  (robot.ikine_a, # analytic solution
        "ikine_a",
        "sol = robot.ikine_a(T)"
    )
    ikfuncs.insert(0, a)    

# setup to run timeit
setup = '''
from __main__ import robot, T, q0
'''
N = 10

# setup results table
table = ANSITable(
    Column("Operation", headalign="^", colalign='<'),
    Column("Time (ms)", headalign="^", fmt="{:.2g}"),
    Column("Error", headalign="^", fmt="{:.3g}"),
    border="thick")

# test the IK methods
for ik in ikfuncs:
    print('Testing:', ik[1])
    
    # test the method, don't pass q0 to the analytic function
    if ik[1] == "ikine_a":
        sol = ik[0](T)
    else:
        sol = ik[0](T, q0=q0)

    # print error message if there is one
    if not sol.success:
        print('  failed:', sol.reason)

    # evalute the error
    err = np.linalg.norm(T - robot.fkine(sol.q))
    print('  error', err)

    if N > 0:
        # evaluate the execution time
        t = timeit.timeit(stmt=ik[2], setup=setup, number=N)
    else:
        t = 0

    # add it to the output table
    table.row(f"`{ik[1]}`", t/N*1e3, err)

# pretty print the results     
table.print()
print(table.markdown())