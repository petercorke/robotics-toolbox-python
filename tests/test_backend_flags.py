"""The capability flags plot() reads from a backend, instead of checking its class"""

import importlib
import types

import pytest

import roboticstoolbox as rtb
from roboticstoolbox.backends.Connector import Connector
from roboticstoolbox.backends.PyPlot import PyPlot, PyPlot2

# the package also exports a class called PyPlot, so ask for the module by name
pyplot_module = importlib.import_module("roboticstoolbox.backends.PyPlot.PyPlot")


# --- the flags and their defaults ------------------------------------------------------


@pytest.mark.parametrize("flag", ["embeds_in_notebook", "supports_movie"])
def test_a_backend_has_neither_ability_unless_it_says_so(flag) -> None:
    assert getattr(Connector, flag) is False


def test_pyplot_can_make_a_movie() -> None:
    assert PyPlot.supports_movie is True


def test_pyplot2_cannot_make_a_movie() -> None:
    # it has no getframe(), so plot() never offered it a movie
    assert PyPlot2.supports_movie is False


@pytest.mark.parametrize("backend", [PyPlot, PyPlot2])
def test_the_pyplot_backends_are_not_web_pages(backend) -> None:
    assert backend.embeds_in_notebook is False


def test_swift_is_a_web_page_that_can_be_shown_in_a_notebook() -> None:
    pytest.importorskip("swift")
    from roboticstoolbox.backends.swift import Swift

    assert Swift.embeds_in_notebook is True


# --- plot() reads the flag, not the name of the class ----------------------------------


class WebPage:
    """A stand-in for a backend that is a web page, with a name that is not Swift."""

    embeds_in_notebook = True

    def __init__(self):
        self.launched_with = None

    def launch(self, notebook=False, **kwargs):
        self.launched_with = dict(kwargs, **({"notebook": True} if notebook else {}))

    def add(self, robot, **kwargs):
        return types.SimpleNamespace(q=None)

    def step(self, dt=0.05):
        pass


def _plot_with(backend, monkeypatch, q):
    robot = rtb.models.DH.Puma560()
    monkeypatch.setattr(pyplot_module, "_isnotebook", lambda: True)
    monkeypatch.setattr(
        type(robot), "_get_graphical_backend", lambda self, name=None: backend
    )
    robot.plot(q)
    return backend.launched_with


def test_a_web_page_backend_is_shown_in_the_cell_whatever_it_is_called(
    monkeypatch,
) -> None:
    assert type(WebPage()).__name__ != "Swift"

    launched = _plot_with(WebPage(), monkeypatch, rtb.models.DH.Puma560().qn)

    assert launched["notebook"] is True
    assert launched["realtime"] is False


def test_a_backend_called_swift_that_is_not_a_web_page_gets_no_swift_options(
    monkeypatch,
) -> None:
    # the name used to decide this
    NotAWebPage = type("Swift", (WebPage,), {"embeds_in_notebook": False})

    launched = _plot_with(NotAWebPage(), monkeypatch, rtb.models.DH.Puma560().qn)

    assert "notebook" not in launched
    assert "realtime" not in launched
