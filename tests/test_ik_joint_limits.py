"""Joint-coordinate semantics shared by the numerical IK solvers."""

import numpy as np
import numpy.testing as nt
import pytest

import roboticstoolbox as rtb
from roboticstoolbox.ets import fknm
from roboticstoolbox.robot.IK import IKSolution
from tests import skip_no_qp

skip_no_c = pytest.mark.skipif(
    not fknm._C_AVAILABLE, reason="compiled IK extension is not available"
)
PYTHON_SOLVERS = [
    pytest.param("ikine_LM", {}, id="python-LM"),
    pytest.param("ikine_GN", {}, id="python-GN"),
    pytest.param("ikine_NR", {}, id="python-NR"),
    pytest.param("ikine_QP", {}, marks=skip_no_qp, id="python-QP"),
]
C_SOLVERS = [
    pytest.param("ik_LM", {"method": "chan"}, marks=skip_no_c, id="C-LM-chan"),
    pytest.param("ik_LM", {"method": "wampler"}, marks=skip_no_c, id="C-LM-wampler"),
    pytest.param("ik_LM", {"method": "sugihara"}, marks=skip_no_c, id="C-LM-sugihara"),
    pytest.param("ik_GN", {}, marks=skip_no_c, id="C-GN"),
    pytest.param("ik_NR", {}, marks=skip_no_c, id="C-NR"),
]
SOLVERS = PYTHON_SOLVERS + C_SOLVERS
REPRESENTATIVE_SOLVERS = [PYTHON_SOLVERS[0], C_SOLVERS[0]]


def _mixed_ets(
    prismatic_limit: float = 10.0,
    revolute_limits: tuple[float, float] = (-np.pi, np.pi),
    jindices: list[int] | None = None,
) -> rtb.ETS:
    indices = [None] * 6 if jindices is None else jindices
    return rtb.ETS(
        [
            rtb.ET.Rz(0.2),
            rtb.ET.tx(
                qlim=[-prismatic_limit, prismatic_limit],
                flip=True,
                jindex=indices[0],
            ),
            rtb.ET.ty(0.15),
            rtb.ET.ty(qlim=[-10, 10], jindex=indices[1]),
            rtb.ET.tz(qlim=[-10, 10], jindex=indices[2]),
            rtb.ET.Rx(qlim=revolute_limits, jindex=indices[3]),
            rtb.ET.tx(0.25),
            rtb.ET.Ry(qlim=[-np.pi, np.pi], jindex=indices[4]),
            rtb.ET.Rz(qlim=[-np.pi, np.pi], flip=True, jindex=indices[5]),
        ]
    )


def _solve(
    ets: rtb.ETS,
    method: str,
    options: dict[str, str],
    q_target: np.ndarray,
    q0: np.ndarray | None = None,
    **kwargs: float,
) -> IKSolution:
    initial = q_target.copy() if q0 is None else q0.copy()
    initial_before = initial.copy()
    parameters: dict[str, object] = {
        "q0": initial,
        "ilimit": 1,
        "slimit": 1,
        "tol": 1e-12,
    }
    parameters.update(options)
    parameters.update(kwargs)
    solution = getattr(ets, method)(ets.fkine(q_target), **parameters)
    nt.assert_array_equal(initial, initial_before)
    return solution


def _assert_valid_solution(
    ets: rtb.ETS, solution: IKSolution, q_target: np.ndarray
) -> None:
    assert solution.success, solution.reason
    nt.assert_allclose(ets.fkine(solution.q).A, ets.fkine(q_target).A, atol=2e-6)
    assert np.all(solution.q >= ets.qlim[0])
    assert np.all(solution.q <= ets.qlim[1])


@pytest.mark.parametrize("method, options", SOLVERS)
@pytest.mark.parametrize("displacement", [4.0, -4.0], ids=["positive", "negative"])
def test_ik_preserves_prismatic_coordinates(
    method: str, options: dict[str, str], displacement: float
) -> None:
    # Static ETs must not shift the joint numbering; flipped joints still store
    # their own coordinate, rather than a signed or angularly wrapped surrogate.
    ets = _mixed_ets()
    q = np.array([displacement, 0.3, -0.2, 0.4, 0.5, 0.6])
    solution = _solve(ets, method, options, q)
    _assert_valid_solution(ets, solution, q)
    nt.assert_array_equal(solution.q[:3], q[:3])


@pytest.mark.parametrize("method, options", SOLVERS)
def test_ik_preserves_valid_multi_turn_revolute_coordinates(
    method: str, options: dict[str, str]
) -> None:
    ets = _mixed_ets(revolute_limits=(-4 * np.pi, 4 * np.pi))
    q = np.array([0.4, 0.3, -0.2, 2 * np.pi + 0.4, 0.5, 0.6])
    solution = _solve(ets, method, options, q)
    _assert_valid_solution(ets, solution, q)
    nt.assert_allclose(solution.q, q, atol=1e-12)


@pytest.mark.parametrize("method, options", REPRESENTATIVE_SOLVERS)
@pytest.mark.parametrize(
    "limits, angle, expected",
    [
        pytest.param((3.5, 5.5), 4.0, 4.0, id="valid-offset-range"),
        pytest.param((3.5, 5.5), 4.0 + 2 * np.pi, 4.0, id="equivalent-offset-range"),
        pytest.param((-5.5, -3.5), -4.0 - 2 * np.pi, -4.0, id="negative-offset-range"),
        pytest.param(
            (-np.pi, np.pi), -4 * np.pi + 0.4, 0.4, id="negative-multiple-turns"
        ),
        pytest.param((-np.pi, np.pi), np.pi, np.pi, id="positive-boundary"),
        pytest.param((-np.pi, np.pi), -np.pi, -np.pi, id="negative-boundary"),
        pytest.param((0.1, 1.0), 0.1 + 2 * np.pi, 0.1, id="lower-boundary-turn"),
        pytest.param((0.2, 0.3), 0.3 + 2 * np.pi, 0.3, id="upper-boundary-turn"),
        pytest.param((-1.0, -0.1), -0.1 - 2 * np.pi, -0.1, id="negative-boundary-turn"),
    ],
)
def test_ik_selects_equivalent_revolute_coordinate_in_limits(
    method: str,
    options: dict[str, str],
    limits: tuple[float, float],
    angle: float,
    expected: float,
) -> None:
    ets = _mixed_ets(revolute_limits=limits)
    q = np.array([0.4, 0.3, -0.2, angle, 0.5, 0.6])
    solution = _solve(ets, method, options, q)
    _assert_valid_solution(ets, solution, q)
    nt.assert_allclose(solution.q[3], expected, atol=1e-12)


@pytest.mark.parametrize("method, options", SOLVERS)
def test_ik_does_not_wrap_invalid_prismatic_coordinate_into_limits(
    method: str, options: dict[str, str]
) -> None:
    ets = _mixed_ets(prismatic_limit=1.0)
    q = np.array([2 * np.pi + 0.4, 0.3, -0.2, 0.4, 0.5, 0.6])
    rejected = _solve(ets, method, options, q, joint_limits=True)
    assert not rejected.success

    # Disabling limit rejection allows this same pose, but must not change the
    # achieved translation by treating the coordinate as a periodic angle.
    accepted = _solve(ets, method, options, q, joint_limits=False)
    assert accepted.success
    nt.assert_allclose(ets.fkine(accepted.q).A, ets.fkine(q).A, atol=1e-12)
    nt.assert_array_equal(accepted.q[:3], q[:3])


@pytest.mark.parametrize("method, options", REPRESENTATIVE_SOLVERS)
@pytest.mark.parametrize(
    "angle", [1.0, np.nextafter(0.2, -np.inf), np.nextafter(0.3, np.inf)]
)
def test_ik_rejects_revolute_pose_without_equivalent_coordinate_in_limits(
    method: str, options: dict[str, str], angle: float
) -> None:
    ets = _mixed_ets(revolute_limits=(0.2, 0.3))
    q = np.array([0.4, 0.3, -0.2, angle, 0.5, 0.6])
    solution = _solve(ets, method, options, q)
    assert not solution.success


@pytest.mark.parametrize("method, options", SOLVERS)
def test_ik_preserves_translation_after_solver_iterations(
    method: str, options: dict[str, str]
) -> None:
    ets = _mixed_ets()
    q = np.array([4.0, 0.3, -0.2, 0.4, 0.5, 0.6])
    q0 = q + np.array([0.2, -0.1, 0.1, 0.02, -0.02, 0.01])
    solution = _solve(ets, method, options, q, q0=q0, ilimit=100)
    _assert_valid_solution(ets, solution, q)
    assert solution.iterations > 0
    nt.assert_allclose(solution.q[:3], q[:3], atol=2e-6)


@pytest.mark.parametrize("method, options", PYTHON_SOLVERS)
def test_ik_joint_types_follow_sparse_joint_indices(
    method: str, options: dict[str, str]
) -> None:
    # The solver works internally with a padded global coordinate vector, while
    # public q0 and IKSolution.q contain only this branch's active joints.
    ets = _mixed_ets(jindices=[1, 3, 5, 7, 9, 11])
    q = np.array([4.0, 0.3, -0.2, 0.4, 0.5, 0.6])
    solution = _solve(ets, method, options, q)
    _assert_valid_solution(ets, solution, q)
    assert solution.q.shape == (6,)
    nt.assert_allclose(solution.q, q, atol=1e-12)

    limited = _mixed_ets(prismatic_limit=1.0, jindices=[1, 3, 5, 7, 9, 11])
    invalid_q = q.copy()
    invalid_q[0] = 2 * np.pi + 0.4
    rejected = _solve(limited, method, options, invalid_q)
    assert not rejected.success


@pytest.mark.parametrize("method, options", PYTHON_SOLVERS)
def test_ik_trajectory_retains_prismatic_coordinates(
    method: str, options: dict[str, str]
) -> None:
    ets = _mixed_ets()
    qs = np.array([[4.0, 0.3, -0.2, 0.4, 0.5, 0.6], [4.1, 0.2, -0.1, 0.4, 0.5, 0.6]])
    targets = ets.fkine(qs)
    solution = getattr(ets, method)(
        targets, q0=qs[0], ilimit=100, slimit=1, tol=1e-12, **options
    )
    assert solution.success, solution.reason
    assert solution.q.shape == (2, 6)
    for q, target in zip(solution.q, targets):
        nt.assert_allclose(ets.fkine(q).A, target.A, atol=2e-6)
    nt.assert_allclose(solution.q[:, :3], qs[:, :3], atol=2e-6)
