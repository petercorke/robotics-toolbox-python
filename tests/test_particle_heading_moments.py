"""Circular heading moments in native particle-filter simulation steps."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.stats import circmean, circstd
from spatialmath import base

import roboticstoolbox as rtb


def _particle_filter(
    particles: np.ndarray, *, observations: bool = False, seed: int = 7
) -> rtb.ParticleFilter:
    """Build a stationary native vehicle, map, sensor and particle filter."""
    robot = rtb.Bicycle(workspace=10)
    robot.control = (0.0, 0.0)
    landmarks = rtb.LandmarkMap(np.array([[3.0, 4.0], [-3.0, -2.0]]).T, workspace=10)
    sensor = rtb.RangeBearingSensor(
        robot,
        landmarks,
        covar=np.zeros((2, 2)),
        fail=[] if observations else [(0, 100)],
        seed=seed,
    )
    particle_filter = rtb.ParticleFilter(
        robot,
        sensor,
        R=np.zeros((3, 3)),
        L=np.diag([0.2, 0.2]),
        nparticles=len(particles),
        seed=seed,
        animate=False,
    )
    particle_filter._init()
    particle_filter.x = particles.copy()
    return particle_filter


@pytest.mark.parametrize("centre", [0.0, np.pi - 0.02, -np.pi + 0.02])
@pytest.mark.parametrize("copies", [1, 3, 50])
def test_heading_moments_survive_wrap_and_particle_replication(
    centre: float, copies: int
) -> None:
    """A tight heading cloud has the same centre/spread across the angle cut."""
    offsets = np.array([-0.3, -0.1, 0.1, 0.3])
    particles = np.c_[[0.0, 2.0, 4.0, 6.0], [0.0, 1.0, 2.0, 3.0], centre + offsets]
    particles = np.tile(particles, (copies, 1))
    particle_filter = _particle_filter(particles)

    particle_filter._step()

    step = particle_filter.history[-1]
    assert_allclose(base.angdiff(step.xest[2], centre), 0.0, atol=1e-14)
    expected_spread = np.sqrt(-2 * np.log((np.cos(0.1) + np.cos(0.3)) / 2))
    assert_allclose(step.std[2], expected_spread, atol=1e-14)
    assert_allclose(step.xest[:2], [3.0, 1.5])
    assert_allclose(step.std[:2], [np.sqrt(5.0), np.sqrt(1.25)])
    assert_allclose(particle_filter.get_std()[-1], step.std)
    np.testing.assert_equal(np.all(np.abs(particle_filter.x[:, 2]) <= np.pi), True)


def test_single_particle_has_zero_heading_spread() -> None:
    """A one-particle cloud has finite zero population dispersion."""
    particle_filter = _particle_filter(np.array([[2.0, -1.0, 3.0]]))

    particle_filter._step()

    assert_allclose(particle_filter.history[-1].xest, [2.0, -1.0, 3.0])
    # Roundoff in the unit resultant is amplified by the square root.
    assert_allclose(
        particle_filter.history[-1].std,
        np.zeros(3),
        atol=2 * np.sqrt(np.finfo(float).eps),
    )


@pytest.mark.parametrize(
    "heading", [-1.715309588860027, 0.9613273519984773, 1.1812388377497625]
)
@pytest.mark.parametrize("copies", [3, 50, 1000])
def test_identical_headings_have_zero_spread(heading: float, copies: int) -> None:
    """Replicated point masses have zero spread despite resultant roundoff."""
    particles = np.tile([2.0, -1.0, heading], (copies, 1))
    particle_filter = _particle_filter(particles)

    particle_filter._step()

    step = particle_filter.history[-1]
    assert_allclose(base.angdiff(step.xest[2], heading), 0.0, atol=1e-14)
    assert_allclose(step.std, np.zeros(3))


@pytest.mark.parametrize("observations", [False, True])
def test_heading_moments_use_the_current_resampled_cloud(
    observations: bool,
) -> None:
    """Population moments describe the current cloud after native resampling."""
    particles = np.c_[
        np.linspace(-0.2, 0.2, 12),
        np.linspace(0.1, -0.1, 12),
        np.linspace(np.pi - 0.3, np.pi + 0.1, 12),
    ]
    particle_filter = _particle_filter(particles, observations=observations)

    particle_filter._step()

    cloud = particle_filter.x
    expected_heading = circmean(cloud[:, 2], low=-np.pi, high=np.pi)
    expected_spread = circstd(cloud[:, 2], low=-np.pi, high=np.pi)
    step = particle_filter.history[-1]
    assert_allclose(base.angdiff(step.xest[2], expected_heading), 0.0, atol=1e-14)
    assert_allclose(step.std[2], expected_spread)
    assert_allclose(step.xest[:2], cloud[:, :2].mean(axis=0))
    assert_allclose(step.std[:2], cloud[:, :2].std(axis=0))
    if observations:
        np.testing.assert_equal(np.any(step.weights != step.weights[0]), True)


def test_opposing_headings_have_large_circular_dispersion() -> None:
    """Nearly zero resultants have an undefined centre and unbounded spread."""
    particle_filter = _particle_filter(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, np.pi]]))

    particle_filter._step()

    step = particle_filter.history[-1]
    np.testing.assert_equal(np.isfinite(step.xest).all(), True)
    assert_allclose(step.std[:2], np.zeros(2))
    np.testing.assert_equal(step.std[2] > np.pi, True)
