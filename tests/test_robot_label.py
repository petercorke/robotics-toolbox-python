"""The first line of a robot's display names its class: the kind of robot, then the
model, as ``DHRobot.Puma560``."""

import roboticstoolbox as rtb


def _first_line(robot):
    return str(robot).splitlines()[0]


def test_ets_model():
    assert _first_line(rtb.models.ETS.Planar_Y()).startswith("Robot.Planar_Y: ")


def test_urdf_model():
    assert _first_line(rtb.models.URDF.Panda()).startswith("URDFRobot.Panda: ")


def test_dh_model():
    assert _first_line(rtb.models.DH.Puma560()).startswith("DHRobot.Puma560: ")


def test_the_classes_themselves_have_no_model():
    robot = rtb.Robot(rtb.ET.Rz() * rtb.ET.tx(1), name="arm")
    dh = rtb.DHRobot([rtb.RevoluteDH(a=1.0)], name="dh")
    robot2 = rtb.Robot2(rtb.ET2.R() * rtb.ET2.tx(1), name="flat")

    assert _first_line(robot).startswith("Robot: arm")
    assert _first_line(dh).startswith("DHRobot: dh")
    assert _first_line(robot2).startswith("Robot2: flat")


def test_a_users_own_subclass():
    class Mine(rtb.Robot):
        def __init__(self):
            super().__init__(rtb.ET.Rz() * rtb.ET.tx(1), name="mine")

    assert _first_line(Mine()).startswith("Robot.Mine: mine")
