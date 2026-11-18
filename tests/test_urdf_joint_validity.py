"""Configuration checks use the URDF joint's actual position limits."""

import pytest

from roboticstoolbox.tools.urdf import URDF, Joint, JointLimit


@pytest.mark.parametrize("joint_type", ["revolute", "prismatic"])
@pytest.mark.parametrize(
    "position, expected",
    [(-0.6, False), (-0.5, True), (0.0, True), (0.5, True), (0.6, False)],
)
def test_loaded_joint_position_limits(
    joint_type: str, position: float, expected: bool
) -> None:
    model = URDF.loadstr(
        f'<robot name="limits"><link name="base"/><link name="tip"/>'
        f'<joint name="joint" type="{joint_type}">'
        '<parent link="base"/><child link="tip"/>'
        '<limit lower="-0.5" upper="0.5" effort="10" velocity="2"/>'
        "</joint></robot>",
        None,
    )
    assert model.joints[0].is_valid(position) == expected


@pytest.mark.parametrize("joint_type", ["revolute", "prismatic"])
@pytest.mark.parametrize("lower, upper", [(-0.5, None), (None, 0.5), (None, None)])
def test_unspecified_position_bound_is_unbounded(joint_type, lower, upper) -> None:
    joint = Joint(
        "joint",
        joint_type,
        "base",
        "tip",
        limit=JointLimit(effort=10, velocity=2, lower=lower, upper=upper),
    )
    assert joint.is_valid(0)
    assert joint.is_valid(-100) == (lower is None)
    assert joint.is_valid(100) == (upper is None)


@pytest.mark.parametrize("joint_type", ["fixed", "continuous"])
def test_joint_without_position_limits(joint_type: str) -> None:
    joint = Joint("joint", joint_type, "base", "tip")
    assert joint.is_valid(0)
    assert joint.is_valid(100)
