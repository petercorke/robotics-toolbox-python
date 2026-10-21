"""The orientation of the line of sight of Robot.vision_collision_damper."""

import numpy as np
import numpy.testing as nt
import pytest

from roboticstoolbox.robot.Robot import _rotation_between_vectors

Z = np.array([0.0, 0.0, 1.0])


def check(a, b):
    """The rotation is proper and takes the direction of a to the direction of b."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    T = _rotation_between_vectors(a, b)
    R = T.R

    nt.assert_allclose(R.T @ R, np.eye(3), atol=1e-12)
    assert np.linalg.det(R) == pytest.approx(1.0)
    nt.assert_allclose(T.t, np.zeros(3), atol=1e-12)  # a pure rotation
    nt.assert_allclose(R @ (a / np.linalg.norm(a)), b / np.linalg.norm(b), atol=1e-9)
    return R


@pytest.mark.parametrize("scale", [1.0, 2.0, 1e-3, 7.5])
def test_parallel_is_the_identity(scale):
    R = check(Z, scale * Z)
    nt.assert_allclose(R, np.eye(3), atol=1e-12)


@pytest.mark.parametrize("v", [[0.3, 0.4, 0.5], [1.0, -2.0, 3.0], [1e-3, 2e-3, 3e-3]])
def test_parallel_up_to_rounding(v):
    # a is twice b, but their normalized cross product need not be exactly zero
    v = np.array(v)
    R = check(2 * v, v)
    nt.assert_allclose(R, np.eye(3), atol=1e-9)


@pytest.mark.parametrize(
    "b", [[0.0, 0.0, -1.0], [0.0, 0.0, -3.0], [0.0, 0.0, -1e-3]]
)
def test_antiparallel_is_a_half_turn(b):
    R = check(Z, b)
    # a half turn: it is its own inverse, and has trace -1
    nt.assert_allclose(R @ R, np.eye(3), atol=1e-12)
    assert np.trace(R) == pytest.approx(-1.0)


@pytest.mark.parametrize(
    "a", [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.3, 0.4, 0.5], [-1.0, 2.0, -3.0]]
)
def test_antiparallel_in_any_direction(a):
    a = np.array(a)
    R = check(a, -2.5 * a)
    nt.assert_allclose(R @ R, np.eye(3), atol=1e-12)


@pytest.mark.parametrize("b", [[1.0, 2.0, 3.0], [1.0, 0.0, 0.0], [-1.0, -1.0, 0.1]])
def test_generic(b):
    check(Z, b)


def test_random_pairs():
    rng = np.random.default_rng(0)
    for _ in range(200):
        check(rng.normal(size=3), rng.normal(size=3))


def test_a_camera_directly_above_and_below_the_target():
    # the line of sight is parallel, then antiparallel, to the cylinder's axis
    target = np.array([0.5, 0.0, 0.2])
    for camera in (np.array([0.5, 0.0, 1.0]), np.array([0.5, 0.0, -0.6])):
        check(Z, camera - target)


def test_no_warnings():
    # arccos(1.0000000000000002) is NaN with a RuntimeWarning
    import warnings

    a = np.array([0.1, 0.2, 0.3])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check(a, a * (1 + 3e-16))
        check(a, a)
