#!/usr/bin/env python
"""Compare the three RNE implementations on Puma560: C extension, rne_python,
and the generic Robot.rne (spatialmath Spatial* object based).

Robot.rne is invoked as the *unbound* base-class method so that DHRobot's
override doesn't shadow it -- Puma560 (a DHRobot) never calls Robot.rne
through normal dispatch.

Historical note: this originally treated Robot.rne() as just "the third
implementation" and compared it directly against Puma560. Since then (see
rne.md issue 6), Robot.rne() turned out to structurally require joint-last
ETS segments -- true for MDH DHRobots, false for Puma560 (standard DH) --
and now asserts on the incompatible case instead of silently returning a
wrong answer. So Robot.rne(puma, ...) below is *expected* to raise, and
this script demonstrates that rather than crashing on it. For a 3-way
comparison on a robot where all three genuinely apply, see rne_speed.py
(uses rtb.models.DH.Panda(), which is mdh=True).
"""

import timeit

import numpy as np
import numpy.testing as nt

import roboticstoolbox as rtb
from roboticstoolbox.robot.Robot import Robot as RobotBase

puma = rtb.models.DH.Puma560()
qn = puma.qn
qd1 = np.full(6, 0.1)
qdd1 = np.full(6, 0.1)

print("=" * 70)
print("Single pose (q=qn, qd=qdd=[0.1]*6)")
print("=" * 70)

tau_c = puma.rne(qn, qd1, qdd1)
tau_py = puma.rne_python(qn, qd1, qdd1)

print("C extension (rne):        ", tau_c)
print("pure Python (rne_python): ", tau_py)
print("max|C - rne_python|   =", np.max(np.abs(tau_c - tau_py)))

print()
print("Robot.rne on Puma560 (standard DH) -- expected to be rejected:")
try:
    RobotBase.rne(puma, qn, qd1, qdd1)
except AssertionError as e:
    print(f"  correctly rejected: {e}")
else:
    print("  ERROR: expected AssertionError, got a result instead")

print()
print("=" * 70)
print("Timing: trajectory of N=1000 identical rows (q=qn, qd=qdd=[0.1]*6)")
print("=" * 70)

N = 1000
Q = np.tile(qn, (N, 1))
QD = np.tile(qd1, (N, 1))
QDD = np.tile(qdd1, (N, 1))


def bench(label, fn, repeat=5):
    t = min(timeit.repeat(fn, number=1, repeat=repeat))
    print(f"{label:28s} {t*1e3:9.3f} ms   ({t/N*1e6:7.2f} us/row)")
    return t


t_c = bench("C extension (rne)", lambda: puma.rne(Q, QD, QDD))
t_py = bench("pure Python (rne_python)", lambda: puma.rne_python(Q, QD, QDD))

print()
print(f"speedup rne_python -> C:  {t_py/t_c:6.1f}x")
print("(Robot.rne skipped here -- Puma560 is standard DH, see note above."
      " For a full 3-way timing comparison, see rne_speed.py.)")

print()
print("Per-row C-call count check (does rne() batch the trajectory into one")
print("C call, or call frne() once per row from Python?)")
print("Historical note: this used to report N -- frne() was called once per")
print("trajectory row from a Python loop in DHRobot.rne(). Fixed (rne.md")
print("plan step 7): the whole trajectory is now looped over inside a")
print("single C++ call. Should report 1 below.")
import sys

# roboticstoolbox/robot/__init__.py does `from .DHRobot import DHRobot`, which
# rebinds the `DHRobot` attribute of the `roboticstoolbox.robot` package to the
# *class*, shadowing the submodule of the same name -- so
# `import roboticstoolbox.robot.DHRobot as X` silently gives the class, not
# the module, and patching X.frne is a no-op. sys.modules[...] is the only
# reliable way to get the real module object (see test_fknm_fallback.py).
DHRobot_module = sys.modules["roboticstoolbox.robot.DHRobot"]

orig_frne = DHRobot_module.frne
calls = {"n": 0}


def counting_frne(*args, **kwargs):
    calls["n"] += 1
    return orig_frne(*args, **kwargs)


DHRobot_module.frne = counting_frne
puma.rne(Q, QD, QDD)
print(f"frne() C calls for a {N}-row trajectory: {calls['n']}")
DHRobot_module.frne = orig_frne
