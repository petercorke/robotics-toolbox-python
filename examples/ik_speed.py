#!/usr/bin/env python
"""Benchmark numerical IK solver speed three ways: the fast, C-only ik_XX
solvers; the pure-Python ikine_XX solvers backed by the C-accelerated ETS
(fkine/jacobian evaluated in C++); and ikine_XX backed by the pure-Python
ETS fallback (fkine/jacobian evaluated in Python too, as in a pure-Python
wheel/Pyodide build).
"""

import time
import sys

import numpy as np
from ansitable import ANSITable, Column

import roboticstoolbox as rtb
import roboticstoolbox.ets.fknm as fknm

from _cpu_info import cpu_info

# Our robot and ETS
robot = rtb.models.Panda()
ets = robot.ets()

### Experiment parameters
# Number of problems to solve for the C-accelerated columns (ik_XX and
# ikine_XX with the C++ ETS)
nproblems = 10_000

# The pure-Python ETS fallback is interpreted at every iteration, not just
# the outer solver loop -- 10,000 problems would take far too long, so use
# a much smaller sample for that column alone.
nproblems_slow = 1_000

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


ik_solvers = [
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

# ikine_XX equivalents -- same settings as above, minus pinv_damping (which
# ikine_NR/ikine_GN don't accept; only the C-only ik_NR/ik_GN do)
ikine_solvers = [
    lambda Tep: ets.ikine_NR(
        Tep,
        q0=None,
        ilimit=ilimit,
        slimit=slimit,
        tol=tol,
        joint_limits=False,
        mask=mask,
        pinv=True,
    ),
    lambda Tep: ets.ikine_GN(
        Tep,
        q0=None,
        ilimit=ilimit,
        slimit=slimit,
        tol=tol,
        joint_limits=False,
        mask=mask,
        pinv=False,
    ),
    lambda Tep: ets.ikine_LM(
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
    lambda Tep: ets.ikine_LM(
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
    lambda Tep: ets.ikine_LM(
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

solver_names = [
    "Newton Raphson",
    "Gauss Newton",
    "LM Chan",
    "LM Wampler",
    "LM Sugihara",
]

print(
    f"\nNumerical Inverse Kinematics Methods benchmark:\n"
    f"  * running on {cpu_info()},\n"
    f"  * robot is {robot.name} with {robot.n} DoF,\n"
    f"  * ik_XX and ikine_XX (C++ ETS) columns use {nproblems} random configurations,\n"
    f"  * ikine_XX (pure Python ETS) column uses {nproblems_slow} (too slow at {nproblems}).\n"
    f"\nTime per IK solution:\n"
)

ik_times: list[float] = []
ikine_cpp_times: list[float] = []
ikine_py_times: list[float] = []

for solver in ik_solvers:
    print(".", file=sys.stdout, end="", flush=True)  # show activity

    start = time.time()
    for i in range(nproblems):
        solver(Tep[i])
    ik_times.append(time.time() - start)

for solver in ikine_solvers:
    print(".", file=sys.stdout, end="", flush=True)

    start = time.time()
    for i in range(nproblems):
        solver(Tep[i])
    ikine_cpp_times.append(time.time() - start)

# Force the pure-Python ETS fallback: a fresh ETS starts with no C++ handle
# built, and robot.ets() returns a cached instance, so dirty this one's
# cache while _C_AVAILABLE is patched off to make it rebuild as pure-Python.
fknm._C_AVAILABLE = False
ets._fknm_stale = True

for solver in ikine_solvers:
    print(".", file=sys.stdout, end="", flush=True)

    start = time.time()
    for i in range(nproblems_slow):
        solver(Tep[i])
    ikine_py_times.append(time.time() - start)

fknm._C_AVAILABLE = True
ets._fknm_stale = True

print("\r", end="")  # clear the progress line

table = ANSITable(
    Column("Method", colalign="<", headalign="^"),
    Column("ik_XX (μs)", fmt="{:.1f}", headalign="^"),
    Column("ikine_XX, C++ ETS (μs)", fmt="{:.1f}", headalign="^"),
    Column("ikine_XX, pure Python ETS (μs)", fmt="{:.1f}", headalign="^"),
    border="thin",
)

for name, ik_t, ikine_cpp_t, ikine_py_t in zip(
    solver_names, ik_times, ikine_cpp_times, ikine_py_times
):
    table.row(
        name,
        (ik_t / nproblems) * 1e6,
        (ikine_cpp_t / nproblems) * 1e6,
        (ikine_py_t / nproblems_slow) * 1e6,
    )

table.print()
