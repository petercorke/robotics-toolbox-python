"""
URDF/xacro loading failures must say what, where and in which element.

Before this, a bad file surfaced as a bare KeyError('forearm'), an XML
ParseError pointing into text the user never saw (the xacro-expanded
output), or a ValueError with no element name. See issue #673.
"""

import io
import unittest
from pathlib import Path

import roboticstoolbox as rtb
from roboticstoolbox.models.URDF.URDFRobot import URDF_file
from roboticstoolbox.tools.urdf import URDF, URDFError

MINIMAL = """<?xml version="1.0"?>
<robot name="two_link" xmlns:xacro="http://www.ros.org/wiki/xacro">
  <link name="base"/>
  <link name="arm">
    <inertial>
      <origin xyz="0.1 0 0" rpy="0 0 0"/>
      <mass value="1.0"/>
      <inertia ixx="0.1" iyy="0.1" izz="0.1" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>
  <joint name="j1" type="revolute">
    <parent link="base"/>
    <child link="arm"/>
    <origin xyz="0 0 0.5" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1" upper="1" effort="10" velocity="1"/>
  </joint>
</robot>
"""


def load(text):
    return URDF_file(io.StringIO(text))


class TestURDFErrors(unittest.TestCase):
    def test_valid_text_still_loads(self):
        links, name, path = load(MINIMAL)
        self.assertEqual(name, "two_link")
        self.assertIsNone(path)
        robot = rtb.Robot(links)
        self.assertEqual(robot.n, 1)
        self.assertAlmostEqual(robot.links[1].m, 1.0)

    def test_is_a_valueerror(self):
        # existing `except ValueError` handlers keep working
        self.assertIsInstance(URDFError("x"), ValueError)
        with self.assertRaises(ValueError):
            load(MINIMAL.replace('<child link="arm"/>', '<child link="nope"/>'))

    def test_source_not_well_formed_xml(self):
        text = MINIMAL.replace('<link name="arm">', '<link name="arm"')
        with self.assertRaises(URDFError) as cm:
            load(text)
        e = cm.exception
        self.assertEqual(e.stage, "xml")
        self.assertIsNotNone(e.line)
        self.assertIn("line", str(e))

    def test_undefined_xacro_property(self):
        text = MINIMAL.replace('xyz="0 0 0.5"', 'xyz="${nope} 0 0.5"')
        with self.assertRaises(URDFError) as cm:
            load(text)
        e = cm.exception
        self.assertEqual(e.stage, "xacro")
        self.assertIn("nope", str(e))

    def test_missing_package(self):
        text = MINIMAL.replace(
            '<link name="base"/>',
            '<xacro:property name="p" value="$(find no_such_pkg_xyz)"/>\n'
            '  <link name="base">\n'
            '    <visual><geometry><mesh filename="${p}/m.stl"/></geometry></visual>\n'
            "  </link>",
        )
        with self.assertRaises(URDFError) as cm:
            load(text)
        e = cm.exception
        self.assertEqual(e.stage, "xacro")
        self.assertIn("no_such_pkg_xyz", str(e))
        # the root cause is dug out of xacro's wrapping, so the hint is shown
        self.assertIn("update_package_cache", str(e))

    def test_joint_refers_to_undefined_link(self):
        text = MINIMAL.replace('<child link="arm"/>', '<child link="forearm"/>')
        with self.assertRaises(URDFError) as cm:
            load(text)
        e = cm.exception
        self.assertEqual(e.stage, "urdf")
        self.assertIn("forearm", str(e))
        self.assertIn("j1", str(e))
        self.assertIn('<joint name="j1">', e.elements)
        # the expanded URDF is saved and the line points at the joint in it
        self.assertIsNotNone(e.expanded_file)
        expanded = Path(e.expanded_file)
        self.assertTrue(expanded.is_file())
        self.assertIsNotNone(e.line)
        self.assertIn('name="j1"', expanded.read_text().splitlines()[e.line - 1])

    def test_joint_missing_type(self):
        text = MINIMAL.replace('<joint name="j1" type="revolute">', '<joint name="j1">')
        with self.assertRaises(URDFError) as cm:
            load(text)
        e = cm.exception
        self.assertEqual(e.stage, "urdf")
        self.assertIn("type", str(e))
        self.assertIn('<joint name="j1">', e.elements)

    def test_unsupported_joint_type_names_the_joint(self):
        text = MINIMAL.replace('type="revolute"', 'type="wobbly"')
        with self.assertRaises(URDFError) as cm:
            load(text)
        e = cm.exception
        self.assertIn("wobbly", str(e))
        self.assertIn('<joint name="j1">', e.elements)

    def test_missing_mass_value_names_the_link(self):
        text = MINIMAL.replace('<mass value="1.0"/>', "<mass/>")
        with self.assertRaises(URDFError) as cm:
            load(text)
        e = cm.exception
        self.assertEqual(e.stage, "urdf")
        self.assertIn("mass", str(e))
        # innermost element first, enclosing link last
        self.assertEqual(e.elements[0], "<inertial>")
        self.assertIn('<link name="arm">', e.elements)

    def test_bad_number_names_attribute_and_element(self):
        text = MINIMAL.replace('effort="10"', 'effort="ten"')
        with self.assertRaises(URDFError) as cm:
            load(text)
        e = cm.exception
        self.assertIn("effort", str(e))
        self.assertIn("ten", str(e))
        self.assertIn('<joint name="j1">', e.elements)

    def test_duplicate_link_names(self):
        text = MINIMAL.replace('<link name="base"/>', '<link name="arm"/>')
        with self.assertRaises(URDFError) as cm:
            load(text)
        self.assertIn("duplicate link", str(cm.exception))
        self.assertIn("arm", str(cm.exception))

    def test_loadstr_reports_xml_line(self):
        with self.assertRaises(URDFError) as cm:
            URDF.loadstr("<robot name='x'>\n<link name='a'>\n</robot>", None)
        e = cm.exception
        self.assertEqual(e.stage, "xml")
        self.assertEqual(e.line, 3)

    def test_missing_file(self):
        with self.assertRaises(FileNotFoundError) as cm:
            URDF_file("no_such_dir/no_such_file.urdf")
        self.assertIn("no_such_file.urdf", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
