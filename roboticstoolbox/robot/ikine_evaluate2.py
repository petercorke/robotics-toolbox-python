import numpy as np
from spatialmath import SE3
import roboticstoolbox as rtb
import timeit
from ansitable import ANSITable, Column
import traceback

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

solvers = [
        'Nelder-Mead',
        'Powell',
        'CG',
        'BFGS',
        'Newton-CG',  ## Jacobian is required
        'L-BFGS-B',
        'TNC',
        'COBYLA',
        'SLSQP',
        'trust-constr',
        'dogleg',
        'trust-ncg',
        'trust-exact',
        'trust-krylov',
    ]


# setup to run timeit
setup = '''
from __main__ import robot, T, q0
'''
N = 10

# setup results table
table = ANSITable(
    Column("Solver", headalign="^", colalign='<'),
    Column("Time (ms)", headalign="^", fmt="{:.2g}", colalign='>'),
    Column("Error", headalign="^", fmt="{:.3g}", colalign='>'),
    border="thick")

# test the IK methods
for solver in solvers:
    print('Testing:', solver)
    
    # test the method, don't pass q0 to the analytic function
    try:
        sol = robot.ikine_min(T, q0=q0, qlim=True, method=solver)
    except Exception as e:
        print('***', solver, ' failed')
        print(e)
        continue

    # print error message if there is one
    if not sol.success:
        print('  failed:', sol.reason)

    # evalute the error
    err = np.linalg.norm(T - robot.fkine(sol.q))
    print('  error', err)

    if N > 0:
        # evaluate the execution time
        t = timeit.timeit(stmt=f"robot.ikine_min(T, q0=q0, qlim=True, method='{solver}')", setup=setup, number=N)
    else:
        t = 0

    # add it to the output table
    table.row(f"`{solver}`", t/N*1e3, err)

# pretty print the results     
table.print()
print(table.markdown())
