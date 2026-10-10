"""Regression tests for payload capacity with arbitrary joint counts."""

import numpy as np
import numpy.testing as npt
import pytest

from roboticstoolbox.robot.Dynamics import DynamicsMixin


class _DummyRobot:
    paycap = DynamicsMixin.paycap

    def __init__(self, n):
        self.n = n
        self.q = np.zeros(n)
        self.calls = []

    def gravload(self, q):
        return np.zeros(self.n)

    def pay(self, w, q, frame):
        self.calls.append((np.copy(w), np.copy(q), frame))
        return np.arange(1, self.n + 1) * w[0]


def _limits(n):
    return np.column_stack((np.full(n, 20.0), np.full(n, -10.0)))


@pytest.mark.parametrize("n", [3, 6, 7])
def test_paycap_joint_count(n):
    robot = _DummyRobot(n)
    capacity, limiting_joint = robot.paycap(
        [2.0, 0, 0, 0, 0, 0], _limits(n), frame=0
    )
    npt.assert_allclose(capacity, 20.0 / np.arange(1, n + 1))
    npt.assert_equal(capacity.shape, (n,))
    npt.assert_equal(limiting_joint, n - 1)
    npt.assert_allclose(robot.calls[0][0], [1, 0, 0, 0, 0, 0])
    npt.assert_equal(robot.calls[0][2], 0)


def test_paycap_negative_wrench_uses_minimum_torque_limit():
    robot = _DummyRobot(7)
    capacity, limiting_joint = robot.paycap(
        [-10.0, 0, 0, 0, 0, 0], _limits(7)
    )
    npt.assert_allclose(capacity, 10.0 / np.arange(1, 8))
    npt.assert_equal(limiting_joint, 6)


def test_paycap_trajectory_uses_each_wrench_direction():
    robot = _DummyRobot(7)
    q = np.stack([np.zeros(7), np.ones(7)])
    wrench = np.array([[2.0, 0, 0, 0, 0, 0], [-10.0, 0, 0, 0, 0, 0]])
    capacities, joints = robot.paycap(wrench, _limits(7), q=q)
    npt.assert_equal(capacities.shape, (2, 7))
    npt.assert_allclose(capacities[0], 20.0 / np.arange(1, 8))
    npt.assert_allclose(capacities[1], 10.0 / np.arange(1, 8))
    npt.assert_array_equal(joints, [6, 6])
    npt.assert_allclose([call[1] for call in robot.calls], q)


def test_paycap_trajectory_row_mismatch():
    robot = _DummyRobot(7)
    with pytest.raises(ValueError, match="same number of rows"):
        robot.paycap(np.ones((1, 6)), _limits(7), q=np.zeros((2, 7)))


def test_paycap_rejects_zero_wrench_direction():
    robot = _DummyRobot(7)
    with pytest.raises(ValueError, match="nonzero wrench"):
        robot.paycap(np.zeros(6), _limits(7))


def test_paycap_joint_unaffected_by_wrench_is_unbounded():
    robot = _DummyRobot(3)
    robot.pay = lambda w, q, frame: np.array([1.0, 0.0, 2.0])
    limits = _limits(3)
    limits[1, 1] = 0.0

    with np.errstate(divide="raise", invalid="raise"):
        capacities, joint = robot.paycap([1.0, 0, 0, 0, 0, 0], limits)

    npt.assert_allclose(capacities[[0, 2]], [20.0, 10.0])
    npt.assert_equal(np.isinf(capacities[1]), True)
    npt.assert_equal(joint, 2)
