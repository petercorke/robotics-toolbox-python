"""Quintic boundary conditions across trajectory time scales

The reference is always an independent integer-coefficient polynomial in
normalized time, so these tests share no code with the implementation.
"""

from types import SimpleNamespace

import numpy as np
import numpy.testing as nt
import pytest
from numpy.polynomial import Polynomial

from roboticstoolbox import jtraj, mtraj, quintic, quintic_func

DURATIONS = [0.001, 0.01, 1.0, 10.0, 1000.0, 1e6, -1.0, -1000.0]
SAMPLES = np.array([0.0, 0.125, 0.5, 0.875, 1.0])


@pytest.mark.parametrize("method", ["lstsq", "hermite"])
@pytest.mark.parametrize("duration", DURATIONS)
@pytest.mark.parametrize("moving_endpoints", [False, True])
def test_quintic_func_matches_polynomial(
    method: str, duration: float, moving_endpoints: bool
) -> None:
    # q(tau) from 1 to 2 with zero acceleration at both ends
    polynomial = Polynomial(
        [1, 1, 0, -4, 7, -3] if moving_endpoints else [1, 0, 0, 10, -15, 6]
    )
    # velocities are given in position units per time unit, not per tau
    qd0 = polynomial.deriv()(0) / duration
    qdf = polynomial.deriv()(1) / duration
    actual = quintic_func(1, 2, duration, qd0=qd0, qdf=qdf, method=method)(
        SAMPLES * duration
    )
    for order, values in enumerate(actual):
        nt.assert_allclose(
            values * duration**order,
            polynomial.deriv(order)(SAMPLES),
            rtol=1e-9,
            atol=1e-9,
        )


@pytest.mark.parametrize("duration", [0.001, 1.0, 1000.0])
def test_quintic_func_physical_endpoint_velocities(duration: float) -> None:
    # Hermite basis polynomials in tau, endpoint velocities in time units
    position = Polynomial([0, 0, 0, 10, -15, 6])
    initial_velocity = Polynomial([0, 1, 0, -6, 8, -3])
    final_velocity = Polynomial([0, 0, 0, -4, 7, -3])
    for method in ("lstsq", "hermite"):
        actual = quintic_func(-2, 3, duration, qd0=0.2, qdf=-0.3, method=method)(
            SAMPLES * duration
        )
        for order, values in enumerate(actual):
            reference = (
                5 * position.deriv(order)(SAMPLES)
                + (0.2 * duration) * initial_velocity.deriv(order)(SAMPLES)
                - (0.3 * duration) * final_velocity.deriv(order)(SAMPLES)
            )
            if order == 0:
                reference -= 2
            nt.assert_allclose(values * duration**order, reference, atol=1e-9)
        nt.assert_allclose(actual[0][[0, -1]], [-2, 3], atol=1e-9)
        nt.assert_allclose(actual[1][[0, -1]], [0.2, -0.3], atol=1e-9)
        nt.assert_allclose(actual[2][[0, -1]], [0, 0], atol=1e-6)


def test_quintic_func_long_duration_regression() -> None:
    # PR #736: the unscaled least-squares system started this near zero
    f = quintic_func(1, 2, 1000)
    nt.assert_allclose([f(0)[0], f(500)[0], f(1000)[0]], [1, 1.5, 2], atol=1e-12)
    nt.assert_allclose(quintic(1, 2, 1000).q[[0, -1]], [1, 2], atol=1e-12)


def test_quintic_func_methods_agree() -> None:
    rng = np.random.default_rng(0)
    for duration in (0.001, 0.5, 7.0, 1000.0, 1e5):
        q0, qf, qd0, qdf = rng.uniform(-5, 5, 4)
        t = np.linspace(0, duration, 25)
        a = quintic_func(q0, qf, duration, qd0, qdf, method="lstsq")(t)
        b = quintic_func(q0, qf, duration, qd0, qdf, method="hermite")(t)
        for order, (x, y) in enumerate(zip(a, b)):
            nt.assert_allclose(
                x * duration**order, y * duration**order, rtol=1e-8, atol=1e-8
            )


@pytest.mark.parametrize("duration", [0, np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("method", ["lstsq", "hermite"])
def test_quintic_func_bad_duration(duration: float, method: str) -> None:
    with pytest.raises(ValueError, match="finite and non-zero"):
        quintic_func(1, 2, duration, method=method)


def test_quintic_func_bad_arguments() -> None:
    with pytest.raises(ValueError, match="unknown method"):
        quintic_func(1, 2, 1, method="nope")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must be scalars"):
        quintic_func([1, 2], 3, 1)


@pytest.mark.parametrize("samples", [0.5, [0, 0.5, 1], [[0, 0.5], [0.75, 1]]])
def test_quintic_func_sample_shapes(samples: object) -> None:
    for values in quintic_func(1, 2, 1)(samples):
        nt.assert_equal(np.shape(values), np.shape(samples))


@pytest.mark.parametrize("steps", [11, 1000])
def test_quintic_integer_samples(steps: int) -> None:
    trajectory = quintic(1, 2, steps)
    samples = np.arange(steps) / (steps - 1)
    polynomial = Polynomial([1, 0, 0, 10, -15, 6])

    nt.assert_array_equal(trajectory.t, np.arange(steps))
    nt.assert_equal(not trajectory.istime, True)
    nt.assert_allclose(trajectory.q, polynomial(samples), atol=1e-12)
    nt.assert_allclose(
        trajectory.qd * (steps - 1), polynomial.deriv()(samples), atol=1e-12
    )
    nt.assert_allclose(
        trajectory.qdd * (steps - 1) ** 2, polynomial.deriv(2)(samples), atol=1e-12
    )


@pytest.mark.parametrize("duration", [0.001, 1.0, 1000.0])
@pytest.mark.parametrize("origin", [-5.0, 0.0, 5.0])
def test_quintic_sampled_time_origin(duration: float, origin: float) -> None:
    times = origin + duration * SAMPLES
    elapsed = times[-1] - times[0]
    polynomial = Polynomial([1, 1, 0, -4, 7, -3])
    trajectory = quintic(1, 2, times, qd0=1 / elapsed, qdf=2 / elapsed)

    nt.assert_array_equal(trajectory.t, times)
    nt.assert_equal(trajectory.istime, True)
    tau = (times - times[0]) / elapsed
    for values, reference, scale in (
        (trajectory.q, polynomial, 1),
        (trajectory.qd, polynomial.deriv(), elapsed),
        (trajectory.qdd, polynomial.deriv(2), elapsed**2),
    ):
        nt.assert_allclose(values * scale, reference(tau), rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("duration", [0.001, 1.0, 1000.0])
def test_quintic_multi_axis_matches_jtraj(duration: float) -> None:
    times = np.linspace(0, duration, 11)
    initial = np.array([1, -2, 3])
    final = np.array([2, 4, 3])
    actual = mtraj(quintic, initial, final, times)
    expected = jtraj(initial, final, times)

    nt.assert_equal(actual.q.shape, (11, 3))
    nt.assert_allclose(actual.q, expected.q, atol=1e-9)
    nt.assert_allclose(actual.qd * duration, expected.qd * duration, atol=1e-9)
    nt.assert_allclose(actual.qdd * duration**2, expected.qdd * duration**2, atol=1e-9)


def test_quintic_needs_two_samples() -> None:
    with pytest.raises(ValueError, match="at least two"):
        quintic(1, 2, 1)
    with pytest.raises(TypeError, match="bad argument for time"):
        quintic(1, 2, "not time")  # type: ignore[arg-type]


def test_bdsim_quintic_block_long_duration() -> None:
    from roboticstoolbox.blocks.arm import Traj

    block = Traj(y0=1, yf=2, T=1000, time=True, traj="quintic")
    block.start(SimpleNamespace(T=1000))
    nt.assert_allclose(block.output(0.0, [], []), [[1], [0], [0]], atol=1e-9)
    nt.assert_allclose(block.output(500.0, [], []), [[1.5], [0.001875], [0]], atol=1e-9)
