"""Planar simulation uses the robot's control mode and seconds for time."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as nt
import pytest

import roboticstoolbox as rtb
from roboticstoolbox.backends.PyPlot import PyPlot2


@pytest.fixture
def env():
    backend = PyPlot2()
    backend.launch()
    yield backend
    backend.close()
    plt.close("all")


def arm():
    robot = rtb.Robot2(rtb.ET2.R() * rtb.ET2.tx(1) * rtb.ET2.R() * rtb.ET2.tx(0.7))
    robot.q = [0.2, -0.3]
    robot.qd = [0.4, -0.2]
    return robot


def endpoint(q):
    """Independent two-link forward kinematics for the rendered line."""
    return [
        np.cos(q[0]) + 0.7 * np.cos(q[0] + q[1]),
        np.sin(q[0]) + 0.7 * np.sin(q[0] + q[1]),
    ]


def drawn_endpoint(handle):
    line = handle.links[-1]
    return [line.get_xdata()[-1], line.get_ydata()[-1]]


@pytest.mark.parametrize("dt", [0.001, 0.01, 0.05])
def test_velocity_step_uses_seconds_and_updates_drawing(env, dt):
    robot = arm()
    initial = robot.q.copy()
    velocity = robot.qd.copy()
    handle = env.add(robot, eeframe=False)

    # The default mode is velocity control. Exercise the public step method,
    # including real Matplotlib drawing, rather than its private integrator.
    env.step(dt)
    expected = initial + velocity * dt
    nt.assert_allclose(robot.q, expected, atol=1e-14, rtol=0)
    nt.assert_allclose(robot.qd, velocity, atol=0, rtol=0)
    nt.assert_allclose(drawn_endpoint(handle), endpoint(expected), atol=1e-13)


def test_repeated_steps_and_velocity_changes(env):
    robot = arm()
    handle = env.add(robot, eeframe=False)
    expected = robot.q.copy()

    for dt, velocity in [(0.01, [0.4, -0.2]), (0.02, [-0.1, 0.5]), (0.005, [0, 0])]:
        robot.qd = velocity
        env.step(dt)
        expected += np.array(velocity) * dt
        nt.assert_allclose(robot.q, expected, atol=1e-14, rtol=0)
        nt.assert_allclose(drawn_endpoint(handle), endpoint(expected), atol=1e-13)


@pytest.mark.parametrize("mode", ["p", "v", "a"])
@pytest.mark.parametrize("readonly", [False, True])
def test_control_modes_and_readonly_plots(env, mode, readonly):
    robot = arm()
    robot.control_mode = mode
    initial = robot.q.copy()
    handle = env.add(robot, readonly=readonly, eeframe=False)

    env.step(0.01)
    expected = initial + robot.qd * 0.01 if mode == "v" and not readonly else initial
    nt.assert_allclose(robot.q, expected, atol=1e-14, rtol=0)
    nt.assert_allclose(drawn_endpoint(handle), endpoint(expected), atol=1e-13)


def test_pinned_readonly_drawing_and_live_robot_share_a_scene(env):
    robot = arm()
    initial = robot.q.copy()
    pinned_pose = [-0.7, 0.4]
    pinned = env.add(robot, q=pinned_pose, readonly=True, eeframe=False)
    live = env.add(robot, eeframe=False)

    env.step(0.01)
    expected = initial + robot.qd * 0.01
    nt.assert_allclose(robot.q, expected, atol=1e-14, rtol=0)
    nt.assert_allclose(drawn_endpoint(live), endpoint(expected), atol=1e-13)
    nt.assert_allclose(drawn_endpoint(pinned), endpoint(pinned_pose), atol=1e-13)


def test_each_robot_uses_its_own_control_mode(env):
    moving, held = arm(), arm()
    moving.control_mode, held.control_mode = "v", "p"
    moving.qd = [-0.3, 0.6]
    held.q = [-0.8, 0.4]
    initial, held_pose = moving.q.copy(), held.q.copy()
    live = env.add(moving, eeframe=False)
    fixed = env.add(held, eeframe=False)

    env.step(0.02)
    expected = initial + moving.qd * 0.02
    nt.assert_allclose(moving.q, expected, atol=1e-14, rtol=0)
    nt.assert_allclose(held.q, held_pose, atol=0, rtol=0)
    nt.assert_allclose(drawn_endpoint(live), endpoint(expected), atol=1e-13)
    nt.assert_allclose(drawn_endpoint(fixed), endpoint(held_pose), atol=1e-13)
