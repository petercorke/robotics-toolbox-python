"""
Payload capacity: the largest multiple of a wrench that the joint torque limits allow.

The reference is physics that does not use the method: a real mass on the last link
changes the gravity load, computed by the recursive Newton-Euler dynamics, and the
payload capacity must be the mass at which a joint reaches its torque limit.
"""

import copy

import numpy as np
import numpy.testing as nt
import pytest
from spatialmath import SE3

import roboticstoolbox as rtb

G = 9.81
W_DOWN = np.array([0, 0, -G, 0, 0, 0])  # the weight of a 1 kg mass, applied to the robot


@pytest.fixture(scope="module", params=["puma", "panda"])
def robot(request):
    return rtb.models.DH.Puma560() if request.param == "puma" else rtb.models.DH.Panda()


def with_mass(robot, m):
    """A copy of the robot with a real mass at the tool point."""
    r = copy.deepcopy(robot)
    r.payload(m, robot.tool.t)
    return r


def within(tau, tau_min, tau_max, tol=1e-9):
    return bool(np.all(tau <= tau_max + tol) and np.all(tau >= tau_min - tol))


def limits(n, rng, symmetric):
    tau_max = rng.uniform(20, 80, n)
    tau_min = -tau_max if symmetric else -rng.uniform(5, 30, n)
    return tau_min, tau_max


class TestAgainstRealMass:
    @pytest.mark.parametrize("symmetric", [True, False])
    def test_the_capacity_is_the_mass_that_reaches_a_limit(self, robot, symmetric):
        rng = np.random.default_rng(1)
        checked = 0
        for _ in range(30):
            q = robot.random_q()
            tau_min, tau_max = limits(robot.n, rng, symmetric)
            cap = robot.payload_capacity(q, np.vstack([tau_min, tau_max]), W_DOWN)[0]
            if np.any(np.isnan(cap)):
                continue  # gravity alone exceeds a limit
            s = np.min(cap)  # kg, since W_DOWN is the weight of 1 kg
            assert within(with_mass(robot, 0.999 * s).gravload(q), tau_min, tau_max)
            assert not within(with_mass(robot, 1.01 * s).gravload(q), tau_min, tau_max)
            checked += 1
        assert checked > 10

    def test_the_torque_that_a_mass_adds_is_minus_J_transpose_w(self, robot):
        # the sign that the capacity relies on
        for _ in range(5):
            q = robot.random_q()
            added = with_mass(robot, 1.0).gravload(q) - robot.gravload(q)
            nt.assert_allclose(added, -robot.jacob0(q).T @ W_DOWN, atol=1e-9)

    def test_asymmetric_limits_are_used_the_right_way_round(self):
        # one joint, gravity holds it at +5; a downward payload pushes the torque on up
        puma = rtb.models.DH.Puma560()
        q = puma.qn
        tau_g = puma.gravload(q)
        tau_p = -puma.jacob0(q).T @ W_DOWN
        j = int(np.argmax(np.abs(tau_p)))
        tau_max = np.full(6, 1e3)
        tau_min = np.full(6, -1e3)
        tau_max[j] = tau_g[j] + 2.0 * abs(tau_p[j]) if tau_p[j] > 0 else 1e3
        tau_min[j] = tau_g[j] - 2.0 * abs(tau_p[j]) if tau_p[j] < 0 else -1e3
        cap = puma.payload_capacity(q, np.vstack([tau_min, tau_max]), W_DOWN)[0]
        assert cap[j] == pytest.approx(2.0)


class TestResult:
    def test_values(self):
        puma = rtb.models.DH.Puma560()
        q = puma.qn
        cap = puma.payload_capacity(q, [56, 97, 52, 10, 10, 10], W_DOWN)
        assert cap.shape == (1, 6)
        tau_g = puma.gravload(q)
        tau_p = -puma.jacob0(q).T @ W_DOWN
        tau_max = np.array([56, 97, 52, 10, 10, 10.0])
        bound = np.where(tau_p > 0, tau_max, -tau_max)
        expect = np.where(np.abs(tau_p) > 1e-10, (bound - tau_g) / tau_p, np.inf)
        nt.assert_allclose(cap[0], expect)

    def test_a_joint_that_the_wrench_does_not_load_is_inf(self):
        puma = rtb.models.DH.Puma560()
        # a force along z and a moment about x never loads joint 1 (rotation about z) at qz
        cap = puma.payload_capacity(puma.qz, [56, 97, 52, 10, 10, 10], [0, 0, -G, 0, 0, 0])
        assert np.isinf(cap[0, 0])

    def test_gravity_alone_beyond_a_limit_is_nan(self):
        puma = rtb.models.DH.Puma560()
        q = puma.qn
        tau_g = puma.gravload(q)
        j = int(np.argmax(np.abs(tau_g)))
        tau_max = np.full(6, 1e3)
        tau_max[j] = abs(tau_g[j]) / 2
        cap = puma.payload_capacity(q, tau_max, W_DOWN)
        assert np.isnan(cap[0, j])
        assert not np.any(np.isnan(np.delete(cap[0], j)))

    def test_a_trajectory(self):
        puma = rtb.models.DH.Puma560()
        qt = np.array([puma.qn, puma.qr, puma.qs])
        cap = puma.payload_capacity(qt, [56, 97, 52, 10, 10, 10], W_DOWN)
        assert cap.shape == (3, 6)
        for k in range(3):
            nt.assert_allclose(
                cap[k], puma.payload_capacity(qt[k], [56, 97, 52, 10, 10, 10], W_DOWN)[0]
            )

    def test_a_single_configuration_may_be_a_vector(self):
        puma = rtb.models.DH.Puma560()
        cap = puma.payload_capacity(puma.qn, [56, 97, 52, 10, 10, 10], W_DOWN)
        assert cap.shape == (1, 6)

    def test_the_forms_of_the_limits(self):
        puma = rtb.models.DH.Puma560()
        tmax = np.array([56, 97, 52, 10, 10, 10.0])
        q = puma.qn
        a = puma.payload_capacity(q, tmax, W_DOWN)
        b = puma.payload_capacity(q, tmax.reshape(1, 6), W_DOWN)
        c = puma.payload_capacity(q, np.vstack([-tmax, tmax]), W_DOWN)
        nt.assert_allclose(a, b)
        nt.assert_allclose(a, c)
        # a smaller minimum changes the joints that the payload loads towards it
        tau_p = -puma.jacob0(q).T @ W_DOWN
        d = puma.payload_capacity(q, np.vstack([-0.5 * tmax, tmax]), W_DOWN)
        towards_max = tau_p > 0
        nt.assert_allclose(d[0, towards_max], a[0, towards_max])
        assert np.all(d[0, ~towards_max] <= a[0, ~towards_max] + 1e-12) or np.any(
            np.isnan(d[0, ~towards_max])
        )

    def test_wrong_shapes(self):
        puma = rtb.models.DH.Puma560()
        with pytest.raises(ValueError):
            puma.payload_capacity(puma.qn, [1, 2, 3], W_DOWN)  # 3 limits for 6 joints
        with pytest.raises(ValueError):
            puma.payload_capacity(puma.qn, [56, 97, 52, 10, 10, 10], [0, 0, 1])
        with pytest.raises(ValueError):
            puma.payload_capacity(np.zeros((2, 5)), [56, 97, 52, 10, 10, 10], W_DOWN)


class TestFrame:
    def test_the_default_frame_is_the_base(self):
        puma = rtb.models.DH.Puma560()
        tmax = [56, 97, 52, 10, 10, 10]
        nt.assert_allclose(
            puma.payload_capacity(puma.qn, tmax, W_DOWN),
            puma.payload_capacity(puma.qn, tmax, W_DOWN, frame="base"),
        )

    def test_a_wrench_in_the_end_effector_frame_is_the_same_wrench_in_the_base(self):
        puma = rtb.models.DH.Puma560()
        puma.tool = SE3.Tz(0.05)
        tmax = [56, 97, 52, 10, 10, 10]
        q = puma.qn
        w_e = np.array([1.0, -2.0, 3.0, 0.1, -0.2, 0.3])  # in the end-effector frame
        R = puma.fkine(q).R
        w_0 = np.r_[R @ w_e[:3], R @ w_e[3:]]  # the same wrench in the base frame
        nt.assert_allclose(
            puma.payload_capacity(q, tmax, w_e, frame="ee"),
            puma.payload_capacity(q, tmax, w_0, frame="base"),
            rtol=1e-9,
            equal_nan=True,
        )

    def test_the_end_effector_frame_makes_a_difference(self):
        puma = rtb.models.DH.Puma560()
        tmax = [56, 97, 52, 10, 10, 10]
        base = puma.payload_capacity(puma.qn, tmax, W_DOWN, frame="base")
        ee = puma.payload_capacity(puma.qn, tmax, W_DOWN, frame="ee")
        assert not np.allclose(base, ee, equal_nan=True)

    @pytest.mark.parametrize("frame", [0, 1, 2, "world", "end", "tool", "", None])
    def test_a_bad_frame(self, frame):
        puma = rtb.models.DH.Puma560()
        with pytest.raises(ValueError, match="frame"):
            puma.payload_capacity(puma.qn, [56, 97, 52, 10, 10, 10], W_DOWN, frame=frame)
