"""Quintic boundary conditions across trajectory time scales."""

from collections.abc import Callable
from decimal import Decimal, localcontext

import numpy as np
import numpy.testing as nt
import pytest
from numpy.polynomial import Polynomial
from numpy.typing import ArrayLike

from roboticstoolbox import jtraj, mtraj, quintic, quintic_func


@pytest.mark.parametrize("duration", [0.001, 0.01, 1.0, 10.0, 999.0, 1000.0, 1e6])
@pytest.mark.parametrize("moving_endpoints", [False, True])
def test_quintic_func_analytic_polynomial(
    duration: float, moving_endpoints: bool
) -> None:
    # These fixed polynomials independently specify all six boundary conditions.
    # Both have p(0)=1, p(1)=2, p''(0)=p''(1)=0. The second has p'(0)=1,
    # p'(1)=2; the first has zero endpoint velocities.
    polynomial = Polynomial(
        [1, 1, 0, -4, 7, -3] if moving_endpoints else [1, 0, 0, 10, -15, 6]
    )
    velocity = polynomial.deriv()
    acceleration = velocity.deriv()
    samples = np.array([0.0, 0.125, 0.25, 0.5, 0.875, 1.0])
    function = quintic_func(
        1, 2, duration, qd0=velocity(0) / duration, qdf=velocity(1) / duration
    )
    position, speed, accel = function(samples * duration)

    nt.assert_allclose(position, polynomial(samples), rtol=1e-12, atol=1e-12)
    nt.assert_allclose(speed * duration, velocity(samples), rtol=1e-12, atol=1e-12)
    nt.assert_allclose(
        accel * duration**2, acceleration(samples), rtol=1e-12, atol=1e-12
    )
    nt.assert_allclose(position[[0, -1]], [1, 2], atol=1e-12)
    nt.assert_allclose(
        speed[[0, -1]] * duration, [velocity(0), velocity(1)], atol=1e-12
    )
    nt.assert_allclose(accel[[0, -1]] * duration**2, [0, 0], atol=1e-12)


@pytest.mark.parametrize("duration", [0.001, 1.0, 1000.0])
def test_quintic_func_physical_endpoint_velocities(duration: float) -> None:
    # Independent Hermite basis functions, with fixed velocities in time units.
    position_basis = Polynomial([0, 0, 0, 10, -15, 6])
    initial_velocity_basis = Polynomial([0, 1, 0, -6, 8, -3])
    final_velocity_basis = Polynomial([0, 0, 0, -4, 7, -3])
    samples = np.array([0.0, 0.125, 0.5, 0.875, 1.0])
    actual = quintic_func(-2, 3, duration, qd0=0.2, qdf=-0.3)(samples * duration)
    for order, values in enumerate(actual):
        # Evaluate each integer-coefficient basis before scaling so exact zero
        # endpoint derivatives are not replaced by oracle cancellation noise.
        reference = (
            5 * position_basis.deriv(order)(samples)
            + (0.2 * duration) * initial_velocity_basis.deriv(order)(samples)
            - (0.3 * duration) * final_velocity_basis.deriv(order)(samples)
        )
        if order == 0:
            reference -= 2
        nt.assert_allclose(
            values * duration**order,
            reference,
            rtol=1e-12,
            atol=1e-12,
        )
    nt.assert_allclose(actual[0][[0, -1]], [-2, 3], atol=1e-12)
    nt.assert_allclose(actual[1][[0, -1]], [0.2, -0.3], atol=1e-12)
    nt.assert_allclose(actual[2][[0, -1]], [0, 0], atol=1e-9)


@pytest.mark.parametrize("steps", [11, 1000])
def test_quintic_integer_samples(steps: int) -> None:
    trajectory = quintic(1, 2, steps)
    samples = np.arange(steps) / (steps - 1)
    polynomial = Polynomial([1, 0, 0, 10, -15, 6])

    nt.assert_array_equal(trajectory.t, np.arange(steps))
    nt.assert_equal(not trajectory.istime, True)
    nt.assert_equal(
        trajectory.q.shape == trajectory.qd.shape == trajectory.qdd.shape == (steps,),
        True,
    )
    nt.assert_allclose(trajectory.q, polynomial(samples), rtol=1e-12, atol=1e-12)
    nt.assert_allclose(
        trajectory.qd * (steps - 1), polynomial.deriv()(samples), atol=1e-12
    )
    nt.assert_allclose(
        trajectory.qdd * (steps - 1) ** 2,
        polynomial.deriv(2)(samples),
        atol=1e-12,
    )


@pytest.mark.parametrize("duration", [0.001, 1.0, 1000.0])
@pytest.mark.parametrize("origin", [-5.0, 0.0, 5.0])
def test_quintic_sampled_time_origin(duration: float, origin: float) -> None:
    times = origin + duration * np.array([0.0, 0.125, 0.5, 0.875, 1.0])
    actual_duration = times[-1] - times[0]
    # Use the represented elapsed times, including floating-point timestamp noise.
    samples = (times - times[0]) / actual_duration
    polynomial = Polynomial([1, 1, 0, -4, 7, -3])
    trajectory = quintic(1, 2, times, qd0=1 / actual_duration, qdf=2 / actual_duration)

    nt.assert_array_equal(trajectory.t, times)
    nt.assert_equal(trajectory.istime, True)
    for values, derivative, scale in (
        (trajectory.q, polynomial, 1),
        (trajectory.qd, polynomial.deriv(), actual_duration),
        (trajectory.qdd, polynomial.deriv(2), actual_duration**2),
    ):
        nt.assert_allclose(values * scale, derivative(samples), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("duration", [0.001, 1.0, 1000.0])
def test_quintic_multi_axis_native_control(duration: float) -> None:
    times = np.linspace(0, duration, 11)
    initial = np.array([1, -2, 3])
    final = np.array([2, 4, 3])
    actual = mtraj(quintic, initial, final, times)
    expected = jtraj(initial, final, times)

    nt.assert_equal(
        actual.q.shape == actual.qd.shape == actual.qdd.shape == (11, 3), True
    )
    nt.assert_array_equal(actual.t, times)
    nt.assert_equal(actual.istime, True)
    for values, reference, scale in (
        (actual.q, expected.q, 1),
        (actual.qd, expected.qd, duration),
        (actual.qdd, expected.qdd, duration**2),
    ):
        nt.assert_equal(np.issubdtype(values.dtype, np.floating), True)
        nt.assert_allclose(values * scale, reference * scale, rtol=1e-12, atol=1e-12)
    nt.assert_allclose(actual.q[[0, -1]], [initial, final], atol=1e-12)


@pytest.mark.parametrize("samples", [0.5, [0, 0.5, 1], [[0, 0.5], [0.75, 1]]])
def test_quintic_func_sample_shapes(samples: ArrayLike) -> None:
    function = quintic_func(np.int64(1), np.float32(2), 1)
    position, velocity, acceleration = function(samples)
    polynomial = Polynomial([1, 0, 0, 10, -15, 6])
    for actual, expected in zip(
        (position, velocity, acceleration),
        (
            polynomial(samples),
            polynomial.deriv()(samples),
            polynomial.deriv(2)(samples),
        ),
    ):
        nt.assert_equal(np.shape(actual) == np.shape(samples), True)
        nt.assert_equal(np.issubdtype(np.asarray(actual).dtype, np.floating), True)
        nt.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("duration", [0.001, 1.0, 1000.0])
@pytest.mark.parametrize(
    "samples",
    [
        np.float32(0.3),
        np.array([0, 0.125, 0.3, 0.5, 0.7, 0.875, 1], dtype=np.float32),
        np.array([[0, 0.3], [0.7, 1]], dtype=np.float32),
    ],
)
def test_quintic_func_float32_sample_precision(
    duration: float, samples: ArrayLike
) -> None:
    times = np.asarray(samples) * np.float32(duration)
    # Evaluate independent integer-coefficient polynomials at the exactly
    # represented samples, including the rounding of the physical timestamps.
    expected = []
    with localcontext() as context:
        context.prec = 70
        for time in np.asarray(times).flat:
            u = Decimal.from_float(float(time)) / Decimal.from_float(duration)
            expected.append(
                [
                    1 + 10 * u**3 - 15 * u**4 + 6 * u**5,
                    30 * u**2 - 60 * u**3 + 30 * u**4,
                    60 * u - 180 * u**2 + 120 * u**3,
                ]
            )
    reference = np.asarray(expected, dtype=float).T.reshape((3, *np.shape(times)))
    for order, actual in enumerate(quintic_func(1, 2, duration)(times)):
        nt.assert_equal(np.shape(actual) == np.shape(times), True)
        nt.assert_equal(np.asarray(actual).dtype == np.dtype(np.float64), True)
        nt.assert_allclose(
            actual * duration**order, reference[order], rtol=1e-12, atol=1e-12
        )


@pytest.mark.parametrize(
    "dtype",
    [
        np.float16,
        np.float32,
        np.float64,
        np.int64,
        np.complex64,
        np.complex128,
        np.longdouble,
        np.clongdouble,
    ],
)
def test_quintic_func_sample_dtype_compatibility(dtype: type) -> None:
    values = (
        [0.3 + 0.2j, 0.7 - 0.1j]
        if np.issubdtype(dtype, np.complexfloating)
        else [0, 0.3, 1]
    )
    samples = np.asarray(values, dtype=dtype)
    result_dtype = np.result_type(samples.dtype, np.float64)
    polynomial = Polynomial([1, 0, 0, 10, -15, 6])
    for order, actual in enumerate(quintic_func(1, 2, 1)(samples)):
        nt.assert_equal(isinstance(actual, np.ndarray), True)
        nt.assert_equal(actual.shape == samples.shape, True)
        nt.assert_equal(actual.dtype == result_dtype, True)
        nt.assert_allclose(
            actual,
            polynomial.deriv(order)(samples.astype(result_dtype)),
            rtol=1e-12,
            atol=1e-12,
        )


@pytest.mark.parametrize("dtype", [np.longdouble, np.clongdouble])
@pytest.mark.parametrize("argument", ["q0", "T", "qd0"])
def test_quintic_func_extended_boundary_dtype_compatibility(
    dtype: type, argument: str
) -> None:
    arguments = {"q0": 1, "qf": 2, "T": 1, "qd0": 0}
    arguments[argument] = dtype(arguments[argument])
    result_dtype = np.result_type(dtype, np.float64)
    if result_dtype in (np.dtype(np.float64), np.dtype(np.complex128)):
        # Some platforms give longdouble the same precision as float64.
        nt.assert_allclose(quintic_func(**arguments)(0.5), [1.5, 1.875, 0], atol=1e-12)
    else:
        with pytest.raises(
            TypeError, match=f"array type {result_dtype} is unsupported"
        ):
            quintic_func(**arguments)


def test_quintic_zero_duration_compatibility() -> None:
    # Preserve the existing least-squares result for the degenerate duration.
    function = quintic_func(1, 2, 0, qd0=0.2, qdf=-0.1)
    position, velocity, acceleration = function(np.array([0, 0.5, 1]))
    nt.assert_allclose(position, [1.5, 1.525, 1.55], atol=1e-12)
    nt.assert_allclose(velocity, [0.05, 0.05, 0.05], atol=1e-12)
    nt.assert_allclose(acceleration, [0, 0, 0], atol=1e-12)


@pytest.mark.parametrize("container", [list, np.array])
def test_quintic_func_single_element_endpoint_arrays(
    container: Callable[[list[int]], ArrayLike],
) -> None:
    actual = quintic_func(container([1]), container([2]), 1)([0, 0.5, 1])
    nt.assert_allclose(actual[0], [1, 1.5, 2], atol=1e-12)
    nt.assert_allclose(actual[1], [0, 1.875, 0], atol=1e-12)
    nt.assert_allclose(actual[2], [0, 0, 0], atol=1e-12)


def test_quintic_func_rejects_multi_axis_endpoints() -> None:
    with pytest.raises(np.linalg.LinAlgError, match="Incompatible dimensions"):
        quintic_func([1, 2], [3, 4], 1)


def test_quintic_invalid_sampling_argument() -> None:
    with pytest.raises(TypeError, match="bad argument for time"):
        quintic(1, 2, "not time")
