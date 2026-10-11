"""Circular heading moments in native particle-filter simulation steps."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.stats import circmean
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
    assert_allclose(step.std[2], np.sqrt(0.05), atol=1e-14)
    assert_allclose(step.xest[:2], [3.0, 1.5])
    assert_allclose(step.std[:2], [np.sqrt(5.0), np.sqrt(1.25)])
    assert_allclose(particle_filter.get_std()[-1], step.std)
    assert np.all(np.abs(particle_filter.x[:, 2]) <= np.pi)


def test_single_particle_has_zero_heading_spread() -> None:
    """A one-particle cloud has finite zero population dispersion."""
    particle_filter = _particle_filter(np.array([[2.0, -1.0, 3.0]]))

    particle_filter._step()

    assert_allclose(particle_filter.history[-1].xest, [2.0, -1.0, 3.0])
    assert_allclose(particle_filter.history[-1].std, np.zeros(3))


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
    residuals = np.angle(np.exp(1j * (cloud[:, 2] - expected_heading)))
    step = particle_filter.history[-1]
    assert_allclose(base.angdiff(step.xest[2], expected_heading), 0.0, atol=1e-14)
    assert_allclose(step.std[2], np.sqrt(np.mean(residuals**2)))
    assert_allclose(step.xest[:2], cloud[:, :2].mean(axis=0))
    assert_allclose(step.std[:2], cloud[:, :2].std(axis=0))
    if observations:
        assert np.any(step.weights != step.weights[0])


def test_symmetric_resultant_has_finite_dispersion() -> None:
    """An undefined circular centre still uses the library's representative."""
    particle_filter = _particle_filter(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, np.pi]]))

    particle_filter._step()

    step = particle_filter.history[-1]
    assert np.isfinite(step.xest).all()
    assert np.isfinite(step.std).all()
    assert 0 <= step.std[2] <= np.pi
