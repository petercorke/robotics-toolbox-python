"""The registry of graphical backends: lookup, registration and the errors"""

from collections import OrderedDict

import pytest

from roboticstoolbox.backends import (
    load_backend,
    register_backend,
    registered_backends,
    registry,
)


class FakeBackend:
    """Stands in for a backend: the registry only makes one, it does not use it."""


@pytest.fixture(autouse=True)
def private_registry(monkeypatch):
    """Registrations made by a test are gone when it ends."""
    monkeypatch.setattr(registry, "_registry", dict(registry._registry))


# ---------------------------------------------------------------------------
# the built-in backends
# ---------------------------------------------------------------------------


def test_the_builtin_backends_are_registered() -> None:
    assert {"pyplot", "pyplot2", "swift"} <= set(registered_backends())


def test_registered_backends_are_sorted() -> None:
    names = registered_backends()
    assert names == sorted(names)


@pytest.mark.parametrize("name", ["pyplot", "PyPlot", " PYPLOT "])
def test_a_builtin_is_found_whatever_its_case(name) -> None:
    from roboticstoolbox.backends.PyPlot import PyPlot

    assert isinstance(load_backend(name), PyPlot)


def test_pyplot2_is_a_different_backend() -> None:
    from roboticstoolbox.backends.PyPlot import PyPlot2

    assert isinstance(load_backend("pyplot2"), PyPlot2)


# ---------------------------------------------------------------------------
# errors in lookup
# ---------------------------------------------------------------------------


def test_an_unknown_name_lists_the_registered_ones() -> None:
    with pytest.raises(ValueError, match="Unknown graphical backend: 'nope'") as info:
        load_backend("nope")
    for name in registered_backends():
        assert name in str(info.value)


def test_a_name_that_is_not_a_string_is_a_type_error() -> None:
    with pytest.raises(TypeError, match="not int"):
        load_backend(3)  # type: ignore[arg-type]


def test_a_missing_package_names_the_pip_extra() -> None:
    register_backend("lost", "no_such_package_xyz:Thing", extra="lost-extra")
    with pytest.raises(
        ModuleNotFoundError, match=r"roboticstoolbox-python\[lost-extra\]"
    ):
        load_backend("lost")


def test_a_missing_package_with_no_extra_is_the_import_error_itself() -> None:
    register_backend("lost", "no_such_package_xyz:Thing")
    with pytest.raises(ModuleNotFoundError, match="no_such_package_xyz") as info:
        load_backend("lost")
    assert "pip install" not in str(info.value)


# ---------------------------------------------------------------------------
# register_backend
# ---------------------------------------------------------------------------


def test_a_string_target_is_not_imported_until_it_is_chosen() -> None:
    # registering a package that is not installed is fine: nothing is imported
    register_backend("later", "no_such_package_xyz:Thing")
    assert "later" in registered_backends()


def test_a_string_target_is_imported_when_chosen() -> None:
    register_backend("od", "collections:OrderedDict")
    assert isinstance(load_backend("od"), OrderedDict)


def test_a_class_target() -> None:
    register_backend("fake", FakeBackend)
    assert isinstance(load_backend("fake"), FakeBackend)


def test_a_function_target_is_called_for_each_backend() -> None:
    register_backend("fake", lambda: FakeBackend())  # type: ignore[arg-type, return-value]
    assert load_backend("fake") is not load_backend("fake")


def test_a_registered_name_is_found_whatever_its_case() -> None:
    register_backend("MyBackend", FakeBackend)
    assert "mybackend" in registered_backends()
    assert isinstance(load_backend("MYBACKEND"), FakeBackend)


def test_a_name_registered_twice_is_an_error() -> None:
    register_backend("fake", FakeBackend)
    with pytest.raises(ValueError, match="already registered"):
        register_backend("fake", FakeBackend)


def test_the_duplicate_check_ignores_case() -> None:
    register_backend("fake", FakeBackend)
    with pytest.raises(ValueError, match="already registered"):
        register_backend("FAKE", FakeBackend)


def test_a_builtin_cannot_be_replaced_by_accident() -> None:
    with pytest.raises(ValueError, match="already registered"):
        register_backend("pyplot", FakeBackend)


def test_replace_true_replaces_the_backend() -> None:
    register_backend("fake", "collections:OrderedDict")
    register_backend("fake", FakeBackend, replace=True)
    assert isinstance(load_backend("fake"), FakeBackend)


def test_replace_true_can_replace_a_builtin() -> None:
    register_backend("pyplot", FakeBackend, replace=True)
    assert isinstance(load_backend("pyplot"), FakeBackend)


@pytest.mark.parametrize("name", ["", "   "])
def test_an_empty_name_is_an_error(name) -> None:
    with pytest.raises(ValueError, match="needs a name"):
        register_backend(name, FakeBackend)


def test_a_name_that_is_not_a_string_cannot_be_registered() -> None:
    with pytest.raises(TypeError):
        register_backend(3, FakeBackend)  # type: ignore[arg-type]


@pytest.mark.parametrize("target", ["nocolon", ":Class", "module:", "a:b:c", ""])
def test_a_string_target_must_be_module_colon_class(target) -> None:
    with pytest.raises(ValueError, match="module:Class"):
        register_backend("fake", target)


def test_a_target_that_is_neither_string_nor_callable() -> None:
    with pytest.raises(TypeError, match="not int"):
        register_backend("fake", 42)  # type: ignore[arg-type]


def test_a_failed_registration_leaves_the_registry_alone() -> None:
    before = registered_backends()
    with pytest.raises(ValueError):
        register_backend("fake", "nocolon")
    assert registered_backends() == before
