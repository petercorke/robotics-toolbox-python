"""Tests for payload capacity, ``paycap``, and the payload torque, ``pay``.

The first group uses a stand-in robot to test the arithmetic and the shapes,
for any number of joints.  The second uses a planar two link arm whose
gravity and payload torques are easy to work out by hand:

- both links 1 m long, the mass of each link is 1 kg at its midpoint
- gravity is along -y, the arm is in the x-y plane
- the weight of a payload is a force along -y, applied at the tip
"""

import numpy as np
import numpy.testing as npt
import pytest

import roboticstoolbox as rtb
from roboticstoolbox import DHRobot, RevoluteDH
from roboticstoolbox.robot.Dynamics import DynamicsMixin

G = 9.81


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
    wmax, joint = robot.paycap([2.0, 0, 0, 0, 0, 0], _limits(n), frame=0)
    # the capacity is a scalar: the smallest of 20 / (1 .. n), at joint n - 1
    npt.assert_allclose(wmax, 20.0 / n)
    npt.assert_equal(np.ndim(wmax), 0)
    npt.assert_equal(joint, n - 1)
    # only the direction of the wrench is used
    npt.assert_allclose(robot.calls[0][0], [1, 0, 0, 0, 0, 0])
    npt.assert_equal(robot.calls[0][2], 0)


def test_paycap_negative_wrench_uses_minimum_torque_limit():
    robot = _DummyRobot(7)
    wmax, joint = robot.paycap([-10.0, 0, 0, 0, 0, 0], _limits(7))
    npt.assert_allclose(wmax, 10.0 / 7)
    npt.assert_equal(joint, 6)


def test_paycap_trajectory_uses_each_wrench_direction():
    robot = _DummyRobot(7)
    q = np.stack([np.zeros(7), np.ones(7)])
    wrench = np.array([[2.0, 0, 0, 0, 0, 0], [-10.0, 0, 0, 0, 0, 0]])
    wmax, joints = robot.paycap(wrench, _limits(7), q=q)
    npt.assert_equal(wmax.shape, (2,))
    npt.assert_allclose(wmax, [20.0 / 7, 10.0 / 7])
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


def test_paycap_joint_unaffected_by_wrench_does_not_limit():
    robot = _DummyRobot(3)
    robot.pay = lambda w, q, frame: np.array([1.0, 0.0, 2.0])
    limits = _limits(3)
    limits[1, 1] = 0.0

    with np.errstate(divide="raise", invalid="raise"):
        wmax, joint = robot.paycap([1.0, 0, 0, 0, 0, 0], limits)

    # joints 0 and 2 give 20 and 10, joint 1 is not loaded so is unbounded
    npt.assert_allclose(wmax, 10.0)
    npt.assert_equal(joint, 2)


# ------------------------------------------------------------------------- #
# a planar two link arm, results worked out by hand


@pytest.fixture
def arm():
    links = [
        RevoluteDH(a=1, m=1, r=[-0.5, 0, 0], I=np.zeros(3)),
        RevoluteDH(a=1, m=1, r=[-0.5, 0, 0], I=np.zeros(3)),
    ]
    return DHRobot(links, gravity=[0, -G, 0], name="planar2")


# limits in the order (maximum, minimum) for each joint
LIMITS = np.array([[30.0, -30.0], [10.0, -10.0]])
WEIGHT = [0, -1, 0, 0, 0, 0]  # a weight, force along -y


def test_arm_gravity_torque(arm):
    # stretched along x: link 1 at x=0.5 and link 2 at x=1.5 load joint 1,
    # only link 2, 0.5 m beyond it, loads joint 2
    npt.assert_allclose(arm.gravload([0, 0]), [G * 2.0, G * 0.5])


def test_pay_is_the_torque_that_supports_the_payload(arm):
    # a 1 N weight at the tip, 2 m and 1 m from the joints, needs [2, 1] N m
    # of holding torque, the same sign as the torque to support gravity
    npt.assert_allclose(arm.pay(WEIGHT, q=[0, 0], frame=0), [2.0, 1.0])
    # and an upward force needs the opposite
    npt.assert_allclose(arm.pay([0, 1, 0, 0, 0, 0], q=[0, 0], frame=0), [-2.0, -1.0])


def test_paycap_planar_arm(arm):
    # holding torque for the payload is [2, 1] per N, gravity takes [2g, 0.5g]
    #   joint 1: (30 - 2g) / 2 = 5.19      joint 2: (10 - 0.5g) / 1 = 5.095
    wmax, joint = arm.paycap(WEIGHT, LIMITS, q=[0, 0], frame=0)
    npt.assert_allclose(wmax, 10 - 0.5 * G)
    npt.assert_equal(joint, 1)


def test_paycap_magnitude_of_wrench_is_ignored(arm):
    wmax, _ = arm.paycap([0, -7, 0, 0, 0, 0], LIMITS, q=[0, 0], frame=0)
    npt.assert_allclose(wmax, 10 - 0.5 * G)


def test_paycap_upward_force_uses_the_minimum_limits(arm):
    # tauP = [-2, -1]: (-30 - 2g) / -2 = 24.8   (-10 - 0.5g) / -1 = 14.9
    wmax, joint = arm.paycap([0, 1, 0, 0, 0, 0], LIMITS, q=[0, 0], frame=0)
    npt.assert_allclose(wmax, 10 + 0.5 * G)
    npt.assert_equal(joint, 1)


def test_paycap_is_negative_if_the_arm_cannot_support_itself(arm):
    # joint 1 is limited to 10 N m but gravity alone needs 2g = 19.62
    limits = np.array([[10.0, -10.0], [10.0, -10.0]])
    wmax, joint = arm.paycap(WEIGHT, limits, q=[0, 0], frame=0)
    npt.assert_allclose(wmax, (10 - 2 * G) / 2)
    assert wmax < 0
    npt.assert_equal(joint, 0)


def test_paycap_in_either_frame_agrees(arm):
    # elbow up: tip at (1, 1), the end-effector x axis is along base +y, so a
    # weight along base -y is along -x of the end-effector frame.
    # Hand: gravity [1.5g, 0], payload [1, 0], capacity (30 - 1.5g) / 1 at
    # joint 1, and joint 2 is not loaded at all
    q = [0, np.pi / 2]
    w0, j0 = arm.paycap([0, -1, 0, 0, 0, 0], LIMITS, q=q, frame=0)
    w1, j1 = arm.paycap([-1, 0, 0, 0, 0, 0], LIMITS, q=q, frame=1)
    npt.assert_allclose(w0, 30 - 1.5 * G)
    npt.assert_allclose(w1, w0)
    npt.assert_equal((j0, j1), (0, 0))


def test_paycap_payload_along_the_arm_is_unlimited(arm):
    # pointing straight up, a force along the arm gives no joint torque and
    # gravity gives none either
    wmax, _ = arm.paycap(WEIGHT, LIMITS, q=[np.pi / 2, 0], frame=0)
    assert np.isinf(wmax)


def test_paycap_trajectory_planar_arm(arm):
    q = np.array([[0.0, 0.0], [0.0, np.pi / 2]])
    w = np.array([[0, -1, 0, 0, 0, 0], [0, -1, 0, 0, 0, 0]])
    wmax, joint = arm.paycap(w, LIMITS, q=q, frame=0)
    npt.assert_allclose(wmax, [10 - 0.5 * G, 30 - 1.5 * G])
    npt.assert_array_equal(joint, [1, 0])


def test_paycap_on_a_real_model():
    # no hand value, but it must run on a six joint model with dynamics and
    # give a sensible scalar and joint index
    puma = rtb.models.DH.Puma560()
    tauR = np.column_stack((np.full(6, 100.0), np.full(6, -100.0)))
    wmax, joint = puma.paycap([0, 0, -1, 0, 0, 0], tauR, q=puma.qn, frame=0)
    assert np.ndim(wmax) == 0
    assert 0 <= joint < 6
    assert wmax > 0
