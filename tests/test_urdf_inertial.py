"""
URDF inertia tensors must be rotated from the inertial frame into the link frame.

URDF gives each link's inertia tensor in its <inertial> frame, which may be
rotated relative to the link frame by <origin rpy="...">. The toolbox's
Link.I is the tensor about the centre of mass with axes parallel to the link
frame, so the loader has to apply I_link = R I R^T. It used to keep only the
translation of the inertial origin and pass the tensor through unrotated,
silently giving wrong dynamics for any link with a rotated inertial frame.
"""

import io
import unittest

import numpy as np
import numpy.testing as nt
from spatialmath import SE3

import roboticstoolbox as rtb
from roboticstoolbox.models.URDF.URDFRobot import URDF_file

TEMPLATE = """<?xml version="1.0"?>
<robot name="one_link">
  <link name="base"/>
  <link name="arm">
    <inertial>
      <origin xyz="{xyz}" rpy="{rpy}"/>
      <mass value="1.0"/>
      <inertia ixx="1" iyy="2" izz="3" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>
  <joint name="j1" type="revolute">
    <parent link="base"/>
    <child link="arm"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3" upper="3" effort="10" velocity="1"/>
  </joint>
</robot>
"""


def arm_link(xyz="0 0 0", rpy="0 0 0"):
    links, _, _ = URDF_file(io.StringIO(TEMPLATE.format(xyz=xyz, rpy=rpy)))
    return next(link for link in links if link.name == "arm")


class TestURDFInertialFrame(unittest.TestCase):
    def test_unrotated_inertial_frame_is_unchanged(self):
        link = arm_link()
        nt.assert_array_almost_equal(link.I, np.diag([1, 2, 3]))

    def test_yaw_swaps_x_and_y_moments(self):
        link = arm_link(rpy=f"0 0 {np.pi / 2}")
        nt.assert_array_almost_equal(link.I, np.diag([2, 1, 3]))

    def test_general_rotation_matches_R_I_Rt(self):
        rpy = [0.3, -0.7, 1.1]
        link = arm_link(rpy=" ".join(str(a) for a in rpy))
        R = SE3.RPY(rpy).R
        nt.assert_array_almost_equal(link.I, R @ np.diag([1, 2, 3]) @ R.T)
        nt.assert_array_almost_equal(link.I, link.I.T)

    def test_centre_of_mass_is_the_origin_translation(self):
        link = arm_link(xyz="0.1 0.2 0.3", rpy="0.4 0.5 0.6")
        nt.assert_array_almost_equal(link.r, [0.1, 0.2, 0.3])

    def test_rne_uses_the_rotated_tensor(self):
        # Joint rotates about z. Rolling the inertial frame by 90 deg about x
        # puts the principal moment iyy=2 on the link z axis, so a unit joint
        # acceleration (no gravity, centre of mass on the axis) needs a torque
        # of 2. With the tensor left unrotated it would wrongly be izz=3.
        links, _, _ = URDF_file(
            io.StringIO(TEMPLATE.format(xyz="0 0 0", rpy=f"{np.pi / 2} 0 0"))
        )
        robot = rtb.Robot(links)
        tau = robot.rne([0.0], [0.0], [1.0], gravity=[0, 0, 0])
        nt.assert_array_almost_equal(tau, [2.0])


if __name__ == "__main__":
    unittest.main()
