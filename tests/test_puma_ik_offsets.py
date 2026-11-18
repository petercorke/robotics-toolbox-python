"""Puma analytic IK uses physical DH angles and calibrated joint coordinates."""

import numpy as np
import numpy.testing as nt
import pytest
from spatialmath import SE3

import roboticstoolbox as rtb

OFFSETS = [
    pytest.param([0, 0, 0, 0, 0, 0], id="uncalibrated"),
    pytest.param([0.2, -0.35, 0.4, 0, 0, 0], id="arm"),
    pytest.param([0, 0, 0, 0.25, -0.15, 0.3], id="wrist"),
    pytest.param([0.2, -0.35, 0.4, 0.25, -0.15, 0.3], id="all-joints"),
    pytest.param([2 * np.pi + 0.2, -2 * np.pi - 0.35, 0.4, 0, 0, 0], id="full-turns"),
]
CONFIGURATIONS = ["lun", "luf", "ldn", "ldf", "run", "ruf", "rdn", "rdf"]
PHYSICAL_ANGLES = np.array([0.1, 0.3, -0.7, 0.2, 0.5, -0.3])


def calibrated_puma(offsets):
    robot = rtb.models.DH.Puma560()
    for link, offset in zip(robot.links, offsets):
        link.offset = offset
    return robot


@pytest.mark.parametrize("offsets", OFFSETS)
@pytest.mark.parametrize("config", CONFIGURATIONS)
def test_offsets_preserve_physical_solution_and_configuration(offsets, config):
    reference = rtb.models.DH.Puma560()
    robot = calibrated_puma(offsets)
    target = reference.fkine(PHYSICAL_ANGLES)
    expected = reference.ikine_a(target, config=config)

    solution = robot.ikine_a(target, config=config)

    assert expected.success
    assert solution.success, solution.reason
    assert solution.residual < 1e-20
    nt.assert_allclose(robot.fkine(solution.q).A, target.A, atol=1e-12)
    # Calibration only changes encoder coordinates; all eight branches must
    # retain their physical angles, allowing equivalent whole turns.
    physical = solution.q + offsets
    nt.assert_allclose(np.sin(physical), np.sin(expected.q), atol=1e-12)
    nt.assert_allclose(np.cos(physical), np.cos(expected.q), atol=1e-12)


@pytest.mark.parametrize("joint", [0, 1, 2])
@pytest.mark.parametrize("offset", [-0.25, 0.2])
def test_each_arm_offset_round_trips_a_reachable_pose(joint, offset):
    robot = rtb.models.DH.Puma560()
    robot.links[joint].offset = offset
    target = robot.fkine(PHYSICAL_ANGLES)

    solution = robot.ikine_a(target)

    assert solution.success, solution.reason
    nt.assert_allclose(robot.fkine(solution.q).A, target.A, atol=1e-12)


@pytest.mark.parametrize("as_array", [False, True])
@pytest.mark.parametrize("with_frames", [False, True])
def test_calibrated_trajectory_with_base_and_tool(as_array, with_frames):
    robot = calibrated_puma([0.2, -0.35, 0.4, 0.25, -0.15, 0.3])
    reference = rtb.models.DH.Puma560()
    if with_frames:
        for model in (robot, reference):
            model.base = SE3(0.4, -0.3, 0.2) * SE3.RPY(0.2, -0.4, 0.3)
            model.tool = SE3(0.05, -0.02, 0.08) * SE3.RPY(-0.3, 0.2, 0.4)
    angles = np.vstack([PHYSICAL_ANGLES, PHYSICAL_ANGLES + 0.1, PHYSICAL_ANGLES - 0.15])
    targets = reference.fkine(angles)
    poses = np.array(targets.A) if as_array else targets

    solution = robot.ikine_a(poses, config="rdf")

    assert solution.success, solution.reason
    assert solution.q.shape == (3, 6)
    assert solution.residual < 1e-20
    assert solution.iterations == solution.searches == 0
    for q, target in zip(solution.q, targets):
        nt.assert_allclose(robot.fkine(q).A, target.A, atol=1e-12)


@pytest.mark.parametrize("as_array", [False, True])
def test_unreachable_pose_does_not_corrupt_other_calibrated_solutions(as_array):
    robot = calibrated_puma([0.2, -0.35, 0.4, 0.25, -0.15, 0.3])
    reference = rtb.models.DH.Puma560()
    targets = SE3(
        [
            reference.fkine(PHYSICAL_ANGLES),
            SE3(5, 5, 5),
            reference.fkine(PHYSICAL_ANGLES + 0.1),
        ]
    )

    solution = robot.ikine_a(np.array(targets.A) if as_array else targets)

    assert not solution.success
    assert solution.residual == np.inf
    assert solution.reason == "Out of reach"
    assert np.all(np.isnan(solution.q[1]))
    for index in (0, 2):
        nt.assert_allclose(
            robot.fkine(solution.q[index]).A, targets[index].A, atol=1e-12
        )


@pytest.mark.parametrize("config", ["run", "ruf"])
def test_calibration_preserves_wrist_singularity_handling(config):
    reference = rtb.models.DH.Puma560()
    robot = calibrated_puma([0.2, -0.35, 0.4, 0.25, -0.15, 0.3])
    angles = PHYSICAL_ANGLES.copy()
    angles[4] = 0
    target = reference.fkine(angles)

    solution = robot.ikine_a(target, config=config)

    assert solution.success, solution.reason
    nt.assert_allclose(robot.fkine(solution.q).A, target.A, atol=1e-12)
