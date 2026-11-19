"""Higher derivatives must retain the dependency order of a serial chain."""

from itertools import product

import numpy as np
import pytest
from numpy.testing import assert_allclose
from spatialmath import SE3

import roboticstoolbox as rtb
from roboticstoolbox import ET


@pytest.mark.parametrize("order", [3, 4, 5])
@pytest.mark.parametrize("flip", [False, True])
@pytest.mark.parametrize("placed", [False, True])
@pytest.mark.parametrize("indices", [(0, 1, 2), (5, 2, 4)])
def test_planar_rpr_partials_match_closed_form(order, flip, placed, indices):
    base = SE3(0.2, -0.3, 0.6) * SE3.RPY(0.4, -0.6, 0.2) if placed else SE3()
    signs = np.array([-1, -1, -1] if flip else [1, 1, 1])
    offsets = np.array([0.13, 0.07, -0.21])
    q = np.array([0.3, 0.2, -0.7])
    theta, displacement, wrist = signs * q + offsets
    joints = [
        ET.Rz(jindex=indices[0], flip=flip),
        ET.tx(jindex=indices[1], flip=flip),
        ET.Rz(jindex=indices[2], flip=flip),
    ]
    for joint, offset in zip(joints, offsets):
        joint.offset = offset
    ets = (
        ET.SE3(base)
        * joints[0]
        * ET.tx(0.8)
        * joints[1]
        * joints[2]
        * ET.tx(0.5)
        * ET.SE3(SE3.Tx(0.2) * SE3.Rx(0.4))
    )
    global_q = np.zeros(max(indices) + 1)
    global_q[list(indices)] = q
    actual = ets.partial_fkine0(global_q, order)
    expected = np.zeros((3,) * (order - 1) + (6, 3))

    # p = (0.8 + d) [cos(theta), sin(theta)]
    #     + 0.7 [cos(theta + wrist), sin(theta + wrist)].
    # Differentiate these scalar expressions directly, including mixed R/P terms.
    for joints in product(range(3), repeat=order):
        nr, np_, nw = (joints.count(j) for j in range(3))
        value = np.zeros(3)
        if nw == 0 and np_ <= 1:
            length = 0.8 + displacement if np_ == 0 else signs[1]
            angle = theta + nr * np.pi / 2
            value[:2] += (
                length * signs[0] ** nr * np.array([np.cos(angle), np.sin(angle)])
            )
        if np_ == 0:
            angle = theta + wrist + (nr + nw) * np.pi / 2
            value[:2] += (
                0.7
                * signs[0] ** nr
                * signs[2] ** nw
                * np.array([np.cos(angle), np.sin(angle)])
            )
        expected[joints[:-1] + (slice(0, 3), joints[-1])] = base.R @ value

    assert actual.shape == expected.shape
    assert_allclose(actual, expected, atol=2e-12)


@pytest.mark.parametrize("order", [3, 4, 5])
@pytest.mark.parametrize("q", [(0.3, -0.7), (-0.6, 0.4)])
def test_spatial_partials_match_closed_form(order, q):
    # Position: ((a+b*cos(q1))*cos(q0), (a+b*cos(q1))*sin(q0), -b*sin(q1)).
    # Angular Jacobian: (ez, Rz(q0) ey), so only the second column varies.
    a, b = 0.8, 0.5
    ets = ET.Rz() * ET.tx(a) * ET.Ry() * ET.tx(b)
    actual = ets.partial_fkine0(q, order)
    expected = np.zeros((2,) * (order - 1) + (6, 2))
    for joints in product(range(2), repeat=order):
        nz, ny = joints.count(0), joints.count(1)
        radius = b * np.cos(q[1] + ny * np.pi / 2) + (a if ny == 0 else 0)
        angle = q[0] + nz * np.pi / 2
        expected[joints[:-1] + (slice(0, 3), joints[-1])] = [
            radius * np.cos(angle),
            radius * np.sin(angle),
            -b * np.sin(q[1] + ny * np.pi / 2) if nz == 0 else 0,
        ]
        if joints[-1] == 1 and all(j == 0 for j in joints[:-1]):
            angle = q[0] + (order - 1) * np.pi / 2
            expected[joints[:-1] + (slice(3, 6), 1)] = [
                -np.sin(angle),
                np.cos(angle),
                0,
            ]

    assert_allclose(actual, expected, atol=2e-12)


@pytest.mark.parametrize("model", [rtb.models.DH.Puma560, rtb.models.DH.Panda])
@pytest.mark.parametrize("shift", [0.0, 0.13])
def test_robot_third_partials_match_hessian_differences(model, shift):
    robot = model()
    robot.base = SE3(0.2, -0.3, 0.5) * SE3.RPY(0.4, -0.2, 0.3)
    robot.tool = SE3(0.07, -0.05, 0.1) * SE3.RPY(-0.1, 0.2, 0.4)
    q = robot.qr + shift
    ets = robot.ets()
    actual = ets.partial_fkine0(q, 3)
    step = 1e-5
    expected = np.stack(
        [
            (ets.hessian0(q + delta) - ets.hessian0(q - delta)) / (2 * step)
            for delta in step * np.eye(robot.n)
        ]
    )
    assert_allclose(actual, expected, atol=2e-9)


def test_third_partials_predict_cartesian_jerk():
    ets = ET.Rz() * ET.tx(0.8) * ET.Rz() * ET.tx(0.5)
    q = np.array([0.3, -0.7])
    qd = np.array([0.6, -0.2])
    third = ets.partial_fkine0(q, 3)[:, :, :3, :]
    jerk = np.einsum("abij,a,b,j->i", third, qd, qd, qd)
    expected = np.zeros(3)
    for length, angle, rate in [(0.8, q[0], qd[0]), (0.5, q.sum(), qd.sum())]:
        expected[:2] += length * rate**3 * np.array([np.sin(angle), -np.cos(angle)])
    assert_allclose(jerk, expected, atol=2e-12)


@pytest.mark.parametrize("order", [3, 4, 5])
def test_prismatic_chain_has_no_higher_derivatives(order):
    ets = ET.tx() * ET.Rz(0.4) * ET.ty() * ET.Ry(0.3) * ET.tz()
    assert_allclose(ets.partial_fkine0([0.2, -0.3, 0.4], order), 0, atol=1e-14)
