#!/usr/bin/env python
"""Benchmark the three RNE implementations -- C extension (rne), pure Python
(rne_python), and the generic ETS/Featherstone implementation (Robot.rne) --
for both a single pose and a 1000-row trajectory.

Uses rtb.models.DH.Panda() (7-DOF, mdh=True, real dynamics) rather than
Puma560: Robot.rne()'s Featherstone recursion structurally requires
joint-last ETS segments, true for MDH DHRobots, false for standard DH (see
rne.md issue 6) -- it simply doesn't apply to Puma560 at all (asserts
instead of silently miscomputing). Panda's DH variant is mdh=True, so all
three implementations genuinely apply here and can be compared fairly, on
the same robot, apples-to-apples.

The trajectory is N *distinct*, randomly-generated poses, not a tiled/
repeated single pose -- a uniform trajectory can't distinguish "this
implementation correctly processes N different rows" from "it silently
computes the same row N times" (relevant here since the C path now loops
the whole trajectory inside a single C++ call rather than once per row --
see rne.md plan step 7 and tests/test_fknm_fallback.py's
TestRneTrajectoryVaryingRows for the correctness side of this).

Riffs off ik_speed.py's ANSITable reporting and rne_compare.py's timing
harness.
"""

import os
import platform
import subprocess
import timeit

import numpy as np
from ansitable import ANSITable, Column

import roboticstoolbox as rtb
from roboticstoolbox.robot.Robot import Robot as RobotBase


def cpu_info() -> str:
    """Best-effort, portable one-line CPU description (name + core count,
    clock speed when the OS actually exposes one). No new hard dependency:
    psutil is used for clock speed only if already installed, and skipped
    otherwise -- some platforms (e.g. Apple Silicon) don't expose a single
    meaningful clock speed at all, so a missing/nonsensical reading is
    silently omitted rather than shown.
    """
    system = platform.system()
    name = None

    if system == "Darwin":
        try:
            name = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip()
        except Exception:
            pass
    elif system == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.lower().startswith("model name"):
                        name = line.split(":", 1)[1].strip()
                        break
        except Exception:
            pass
    elif system == "Windows":
        name = platform.processor() or None

    if not name:
        name = platform.processor() or platform.machine() or "unknown CPU"

    cores = os.cpu_count() or "?"
    info = f"{name} ({cores} cores)"

    try:
        import psutil

        freq = psutil.cpu_freq()
        # Sanity floor: real clock speeds are in the hundreds-to-thousands
        # of MHz; some platforms (Apple Silicon via psutil) report bogus
        # single-digit values instead of raising.
        if freq and freq.max and freq.max > 100:
            info += f", {freq.max:.0f} MHz"
    except Exception:
        pass

    return info


robot = rtb.models.DH.Panda()
n = robot.n
rng = np.random.default_rng(0)

print(f"CPU: {cpu_info()}")
print(f"Robot: {robot.name}  (n={n}, mdh={bool(robot.mdh)})")
print()

# ---------------------------------------------------------------------------
# Correctness check first -- a timing comparison between implementations
# that don't actually agree with each other would be misleading.
#
# NB: this is only a *pairwise* (relative) check -- necessary but not
# sufficient, since all three could share a bug and still agree with each
# other. It's a sanity guard for this benchmark (catches a stale build or
# wrong branch before trusting the timing numbers below), not a proof of
# correctness. That proof lives in tests/test_fknm_fallback.py's
# TestTwoLinkAbsoluteGroundTruth (Spong et al.'s closed-form solution) and
# examples/rne_dh_convention_check.py's Lagrangian-identity check -- both
# independent of all three RNE implementations here.
# ---------------------------------------------------------------------------

q1 = rng.uniform(-np.pi, np.pi, n)
qd1 = rng.uniform(-2.0, 2.0, n)
qdd1 = rng.uniform(-2.0, 2.0, n)

tau_c = robot.rne(q1, qd1, qdd1)
tau_py = robot.rne_python(q1, qd1, qdd1)
tau_base = RobotBase.rne(robot, q1, qd1, qdd1)

print("Correctness (single random pose):")
print(f"  max|C - rne_python|         = {np.max(np.abs(tau_c - tau_py)):.3e}")
print(f"  max|C - Robot.rne|          = {np.max(np.abs(tau_c - tau_base)):.3e}")
print(f"  max|rne_python - Robot.rne| = {np.max(np.abs(tau_py - tau_base)):.3e}")
print()

# ---------------------------------------------------------------------------
# Timing: single pose (1-off) and a 1000-row trajectory
# ---------------------------------------------------------------------------

N = 1000
Q = rng.uniform(-np.pi, np.pi, (N, n))
QD = rng.uniform(-2.0, 2.0, (N, n))
QDD = rng.uniform(-2.0, 2.0, (N, n))


def bench(fn, repeat=7):
    return min(timeit.repeat(fn, number=1, repeat=repeat))


implementations = [
    ("C extension (rne)", lambda q, qd, qdd: robot.rne(q, qd, qdd)),
    ("pure Python (rne_python)", lambda q, qd, qdd: robot.rne_python(q, qd, qdd)),
    ("generic Robot.rne", lambda q, qd, qdd: RobotBase.rne(robot, q, qd, qdd)),
]

results = []
for name, fn in implementations:
    t_single = bench(lambda fn=fn: fn(q1, qd1, qdd1))
    t_traj = bench(lambda fn=fn: fn(Q, QD, QDD))
    results.append((name, t_single, t_traj))

t_c_traj = results[0][2]

table = ANSITable(
    Column("Method", colalign="<", headalign="^"),
    Column("1 pose (us)", fmt="{:.2f}", headalign="^"),
    Column(f"{N}-row traj (ms)", fmt="{:.3f}", headalign="^"),
    Column("us/row", fmt="{:.2f}", headalign="^"),
    Column("speedup vs C (traj)", fmt="{:.1f}x", headalign="^"),
    border="thin",
)

for name, t_single, t_traj in results:
    table.row(
        name,
        t_single * 1e6,
        t_traj * 1e3,
        t_traj / N * 1e6,
        t_traj / t_c_traj,
    )

print(f"Timing over a {N}-row trajectory (distinct random poses, not tiled):")
table.print()
