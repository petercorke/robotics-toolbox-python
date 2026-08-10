#!/usr/bin/env python
"""Benchmark the fast, C-only numerical IK solvers (ik_LM's three variants)
over a batch of random reachable poses.
"""

import time
import sys

import numpy as np
from ansitable import ANSITable, Column

import roboticstoolbox as rtb

from _cpu_info import cpu_info

# Our robot and ETS
robot = rtb.models.Panda()
ets = robot.ets()

### Experiment parameters
# Number of problems to solve
nproblems = 10_000

# Cartesion DoF priority matrix
mask = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

# random valid q values which will define Tep
q_rand = ets.random_q(nproblems)

# Our desired end-effector poses
Tep = np.zeros((nproblems, 4, 4))

for i in range(nproblems):
    Tep[i] = ets.eval(q_rand[i])

# Maximum iterations allowed in a search
ilimit = 30

# Maximum searches allowed per problem
slimit = 100

# Solution tolerance
tol = 1e-6


solvers = [
    lambda Tep: ets.ik_NR(
        Tep,
        q0=None,
        ilimit=ilimit,
        slimit=slimit,
        tol=tol,
        joint_limits=False,
        mask=mask,
        pinv=True,
        pinv_damping=0.0,
    ),
    lambda Tep: ets.ik_GN(
        Tep,
        q0=None,
        ilimit=ilimit,
        slimit=slimit,
        tol=tol,
        joint_limits=False,
        mask=mask,
        pinv=False,
        pinv_damping=0.2,
    ),
    lambda Tep: ets.ik_LM(
        Tep,
        q0=None,
        ilimit=ilimit,
        slimit=slimit,
        tol=tol,
        joint_limits=True,
        mask=mask,
        k=0.1,
        method="chan",
    ),
    lambda Tep: ets.ik_LM(
        Tep,
        q0=None,
        ilimit=ilimit,
        slimit=slimit,
        tol=tol,
        joint_limits=True,
        mask=mask,
        k=1e-4,
        method="wampler",
    ),
    lambda Tep: ets.ik_LM(
        Tep,
        q0=None,
        ilimit=ilimit,
        slimit=slimit,
        tol=tol,
        joint_limits=True,
        mask=mask,
        k=0.1,
        method="sugihara",
    ),
]

times: list[float] = []

solver_names = [
    "Newton Raphson",
    "Gauss Newton",
    "LM Chan",
    "LM Wampler",
    "LM Sugihara",
]

print(f"\nNumerical Inverse Kinematics Methods benchmark:\n  * running on {cpu_info()},\n  * robot is {robot.name} with {robot.n} DoF,\n  * for a batch of {nproblems} random configurations.\n\nTime per IK solution:\n")

for solver in solvers:
    print(".", file=sys.stdout, end="", flush=True) # show activity

    start = time.time()

    for i in range(nproblems):
        solver(Tep[i])

    total_time = time.time() - start
    times.append(total_time)


print("\r", end="") # clear the progress line

table = ANSITable(
    Column("Method", colalign="<", headalign="^"),
    Column("Time (μs)", fmt="{:.1f}", headalign="^"),
    border="thin",
)

for name, t in zip(solver_names, times):
    table.row(
        name,
        (t / nproblems) * 1e6,
    )

table.print()
