#!/usr/bin/env python
"""
@author Jesse Haviland
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Connector(Protocol):
    """
    What RTB needs of a graphical backend

    A backend is any object that has these methods and flags: it does not have to
    inherit from this class.  Swift does not, since it does not depend on RTB.
    The PyPlot backends inherit from it explicitly, which supplies the default
    values of the flags and of :meth:`_add_teach_panel`, and makes a missing
    method an error at construction.

    The methods take different arguments in different backends, so they are
    declared with ``*args, **kwargs``; see the backend for the details.
    """

    #: True if the backend supports the interactive teach pendant.
    supports_teach: bool = True

    #: True if the backend can render EllipsePlot objects.
    supports_ellipse: bool = False

    #
    #  Basic methods to do with the state of the external program
    #

    @abstractmethod
    def launch(self, *args: Any, **kwargs: Any) -> Any:
        """
        Launch the external program with an empty or specific scene
        """
        ...

    @abstractmethod
    def step(self, *args: Any, **kwargs: Any) -> Any:
        """
        Trigger the external program to make a time step

        ``step(dt)`` goes through each robot in the scene and makes them act
        based on their control type (position, velocity, acceleration, or
        torque).  Upon acting, the other three of the four control types will be
        updated in the internal state of the robot object.  The control type is
        defined by the robot object, and not all robot objects support all
        control types.
        """
        ...

    @abstractmethod
    def reset(self, *args: Any, **kwargs: Any) -> Any:
        """
        Reset the external program to the state defined by :meth:`launch`
        """
        ...

    @abstractmethod
    def restart(self, *args: Any, **kwargs: Any) -> Any:
        """
        Close and relaunch the external program to the state defined by
        :meth:`launch`
        """
        ...

    @abstractmethod
    def close(self, *args: Any, **kwargs: Any) -> Any:
        """
        Gracefully close the external program
        """
        ...

    #
    #  Methods to interface with the robots created in other environments
    #

    @abstractmethod
    def add(self, *args: Any, **kwargs: Any) -> Any:
        """
        Add an object to the external environment

        ``id = add(object)``: the object must be of a class the backend knows.
        It is added to the list of objects which will act upon :meth:`step`.
        """
        ...

    @abstractmethod
    def remove(self, *args: Any, **kwargs: Any) -> Any:
        """
        Remove an object from the external environment, given its id
        """
        ...

    def _add_teach_panel(
        self, robot: Any, q: Any, handle: Any = None, block: bool = True
    ) -> Any:
        """
        Add the teach pendant controls for ``robot``, if the backend supports
        them (see ``supports_teach``).

        :return: True if the backend has already blocked until dismissed
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not have a teach pendant"
        )

    @abstractmethod
    def hold(self, *args: Any, **kwargs: Any) -> Any:  # pragma nocover
        """
        Keep the backend open, that is stop the program from closing once the
        main script has finished.  This method may need to keep an event loop
        running for the backend to stay responsive.
        """
        ...
