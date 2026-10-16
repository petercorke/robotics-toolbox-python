"""Parsed URDF names must be available to callers and transmission validation."""

import numpy.testing as nt
import pytest

from roboticstoolbox.tools.urdf import URDF, URDFError

ROBOT = """<robot name="name_maps">
  <link name="base"/>
  <link name="arm"/>
  <joint name="hinge" type="continuous">
    <parent link="base"/><child link="arm"/>
  </joint>
  <transmission name="drive">
    <type>transmission_interface/SimpleTransmission</type>
    <joint name="{joint}">
      <hardwareInterface>PositionJointInterface</hardwareInterface>
    </joint>
    <actuator name="motor">
      <hardwareInterface>PositionJointInterface</hardwareInterface>
      <mechanicalReduction>2</mechanicalReduction>
    </actuator>
  </transmission>
</robot>"""


def test_name_maps_preserve_parsed_object_identity_and_return_copies():
    robot = URDF.loadstr(ROBOT.format(joint="hinge"), None)

    for mapping_name, sequence_name in (
        ("link_map", "links"),
        ("joint_map", "joints"),
        ("transmission_map", "transmissions"),
    ):
        mapping = getattr(robot, mapping_name)
        objects = getattr(robot, sequence_name)
        nt.assert_equal(set(mapping) == {item.name for item in objects}, True)
        for item in objects:
            nt.assert_equal(mapping[item.name] is item, True)
        mapping.clear()
        nt.assert_equal(len(getattr(robot, mapping_name)), len(objects))


def test_no_joints_or_transmissions_have_empty_name_maps():
    robot = URDF.loadstr('<robot name="fixed"><link name="base"/></robot>', None)
    nt.assert_equal(robot.link_map["base"] is robot.links[0], True)
    nt.assert_equal(robot.joint_map, {})
    nt.assert_equal(robot.transmission_map, {})


def test_transmission_must_reference_an_existing_joint():
    with pytest.raises(
        URDFError, match="Transmission drive has invalid joint name absent"
    ):
        URDF.loadstr(ROBOT.format(joint="absent"), None)
