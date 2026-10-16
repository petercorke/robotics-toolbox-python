"""Robot.default_backend, per robot, and the class-level default used by rtbtool"""

import inspect

import pytest

import roboticstoolbox as rtb
from roboticstoolbox import Robot


@pytest.fixture
def restore_class_default():
    yield
    Robot.set_default_backend(None)


def test_class_default_reaches_every_robot(restore_class_default) -> None:
    before = rtb.models.DH.Puma560()
    assert before.default_backend is None

    Robot.set_default_backend("swift")

    after = rtb.models.DH.Panda()
    assert before.default_backend == "swift"
    assert after.default_backend == "swift"


def test_instance_value_wins_over_the_class_default(restore_class_default) -> None:
    Robot.set_default_backend("swift")
    robot = rtb.models.DH.Puma560()
    robot.default_backend = "pyplot"
    assert robot.default_backend == "pyplot"
    assert rtb.models.DH.Panda().default_backend == "swift"


def test_default_backend_remains_a_property(restore_class_default) -> None:
    # rtbtool used to replace it with a string on the class
    Robot.set_default_backend("swift")
    assert isinstance(inspect.getattr_static(Robot, "default_backend"), property)


def test_class_default_can_be_cleared(restore_class_default) -> None:
    Robot.set_default_backend("swift")
    Robot.set_default_backend(None)
    assert rtb.models.DH.Puma560().default_backend is None
