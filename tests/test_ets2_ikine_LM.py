"""
Tests for ETS2.ikine_LM, the planar numerical inverse kinematics solver.
"""

import numpy as np
import numpy.testing as nt
import pytest
from spatialmath import SE2

from roboticstoolbox.ets.ET2 import R, tx
from roboticstoolbox.robot.IK import IKSolution


def two_link():
    return R() * tx(1) * R() * tx(1)


def three_link():
    return R() * tx(1) * R() * tx(1) * R() * tx(0.5)


def pose_error(ets, q, Tep):
    Te = np.asarray(ets.fkine(q).A, dtype=float)
    Tep = np.asarray(Tep, dtype=float)
    d = np.arctan2(Tep[1, 0], Tep[0, 0]) - np.arctan2(Te[1, 0], Te[0, 0])
    return np.array(
        [Tep[0, 2] - Te[0, 2], Tep[1, 2] - Te[1, 2], np.arctan2(np.sin(d), np.cos(d))]
    )


def test_two_link_reaches_pose():
    ets = two_link()
    Tep = ets.fkine([0.3, 0.7])
    sol = ets.ikine_LM(Tep, seed=0, tol=1e-12)
    assert isinstance(sol, IKSolution)
    assert sol.success
    assert sol.reason == "Success"
    assert sol.q.shape == (2,)
    nt.assert_allclose(pose_error(ets, sol.q, Tep.A), 0, atol=1e-5)


def test_accepts_ndarray_target():
    ets = three_link()
    Tep = ets.fkine([0.2, -0.4, 0.9]).A
    sol = ets.ikine_LM(Tep, seed=1, tol=1e-12)
    assert sol.success
    nt.assert_allclose(pose_error(ets, sol.q, Tep), 0, atol=1e-5)


def test_initial_guess_converges_in_one_search():
    ets = three_link()
    q_true = np.array([0.2, -0.4, 0.9])
    Tep = ets.fkine(q_true)
    sol = ets.ikine_LM(Tep, q0=q_true + 0.05, seed=3, tol=1e-12)
    assert sol.success
    assert sol.searches == 1
    nt.assert_allclose(sol.q, q_true, atol=1e-4)


def test_initial_guess_matrix():
    ets = two_link()
    Tep = ets.fkine([0.3, 0.7])
    sol = ets.ikine_LM(
        Tep, q0=np.array([[2.0, 2.0], [0.3, 0.7]]), ilimit=1, slimit=2, tol=1e-12
    )
    assert sol.success
    assert sol.searches == 2


def test_mask_position_only():
    # a 2-link arm can't reach an arbitrary (x, y, θ), so only position
    ets = two_link()
    Tep = SE2(1.2, 0.8, 2.5)
    assert not ets.ikine_LM(Tep, seed=0, slimit=10).success
    sol = ets.ikine_LM(Tep, mask=[1, 1, 0], seed=0, tol=1e-12)
    assert sol.success
    nt.assert_allclose(ets.fkine(sol.q).t, [1.2, 0.8], atol=1e-5)


def test_joint_limits():
    ets = R() * tx(1) * R(qlim=[-1, 1]) * tx(1)
    Tep = ets.fkine([0.2, 2.0])  # q2 = 2.0 is outside its limit
    sol = ets.ikine_LM(Tep, seed=0, slimit=20)
    assert not sol.success
    assert "violates joint limits" in sol.reason
    sol = ets.ikine_LM(Tep, seed=0, joint_limits=False, tol=1e-12)
    assert sol.success
    nt.assert_allclose(pose_error(ets, sol.q, Tep.A), 0, atol=1e-5)


def test_unreachable_target():
    ets = two_link()
    sol = ets.ikine_LM(SE2(5, 0, 0), mask=[1, 1, 0], seed=0, ilimit=10, slimit=5)
    assert not sol.success
    assert sol.searches == 5
    assert sol.reason.startswith("iteration and search limit reached")


def test_seed_is_reproducible():
    ets = three_link()
    Tep = ets.fkine([1.0, 1.0, -0.5])
    a = ets.ikine_LM(Tep, seed=7)
    b = ets.ikine_LM(Tep, seed=7)
    nt.assert_array_equal(a.q, b.q)
    assert a.iterations == b.iterations


def test_trajectory():
    ets = three_link()
    qs = np.array([[0.1, 0.2, 0.3], [0.4, -0.2, 0.1], [-0.3, 0.5, 0.2]])
    Tep = ets.fkine(qs)
    assert len(Tep) == 3
    sol = ets.ikine_LM(Tep, seed=0, tol=1e-12)
    assert sol.success
    assert sol.q.shape == (3, 3)
    for q, T in zip(sol.q, Tep):
        nt.assert_allclose(pose_error(ets, q, T.A), 0, atol=1e-5)

    sol2 = ets.ikine_LM(np.array([T.A for T in Tep]), seed=0, tol=1e-12)
    assert sol2.success
    assert sol2.q.shape == (3, 3)


def test_bad_arguments():
    ets = two_link()
    Tep = ets.fkine([0.3, 0.7])
    with pytest.raises(ValueError):
        ets.ikine_LM(Tep, mask=[1, 1])
    with pytest.raises(ValueError):
        ets.ikine_LM(np.eye(4))
    with pytest.raises(ValueError):
        ets.ikine_LM(Tep, q0=[0.1, 0.2, 0.3])
