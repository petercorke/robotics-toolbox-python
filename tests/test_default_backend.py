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


# the names are not case sensitive, wherever they are given


def test_the_robot_default_is_stored_in_lower_case() -> None:
    robot = rtb.models.DH.Puma560()
    robot.default_backend = "Swift"
    assert robot.default_backend == "swift"


def test_the_class_default_is_stored_in_lower_case(restore_class_default) -> None:
    Robot.set_default_backend("PyPlot2")
    assert rtb.models.DH.Puma560().default_backend == "pyplot2"


def test_a_default_given_in_upper_case_selects_the_backend(
    restore_class_default,
) -> None:
    from roboticstoolbox.backends.PyPlot import PyPlot2

    Robot.set_default_backend("PYPLOT2")
    assert isinstance(rtb.models.DH.Puma560()._get_graphical_backend(), PyPlot2)


def test_an_explicit_backend_in_upper_case() -> None:
    from roboticstoolbox.backends.PyPlot import PyPlot

    robot = rtb.models.DH.Puma560()
    assert isinstance(robot._get_graphical_backend("PyPlot"), PyPlot)


def test_the_default_can_be_cleared_on_a_robot() -> None:
    robot = rtb.models.DH.Puma560()
    robot.default_backend = "pyplot2"
    robot.default_backend = None
    assert robot.default_backend is None


def test_a_default_that_is_not_a_name_is_a_type_error(restore_class_default) -> None:
    robot = rtb.models.DH.Puma560()
    with pytest.raises(TypeError):
        robot.default_backend = 3  # type: ignore[assignment]
    with pytest.raises(TypeError):
        Robot.set_default_backend(3)  # type: ignore[arg-type]


def test_a_registered_backend_can_be_the_default(
    restore_class_default, monkeypatch
) -> None:
    from roboticstoolbox.backends import register_backend, registry

    class Fake:
        pass

    monkeypatch.setattr(registry, "_registry", dict(registry._registry))
    register_backend("Fake", Fake)
    robot = rtb.models.DH.Puma560()
    robot.default_backend = "FAKE"
    assert isinstance(robot._get_graphical_backend(), Fake)
