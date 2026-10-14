#!/usr/bin/env python
"""
@author Jesse Haviland
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Connector(ABC):
    #: True if the backend supports the interactive teach pendant.
    supports_teach: bool = True

    #: True if the backend can render EllipsePlot objects.
    supports_ellipse: bool = False

    #: velocity and force ellipsoids shown by the teach pendant, set by it
    _teach_vellipse: Any
    _teach_fellipse: Any

    def __init__(self) -> None:
        super().__init__()

    #
    #  Basic methods to do with the state of the external program
    #

    @abstractmethod
    def launch(self, *args: Any, **kwargs: Any) -> Any:
        """
        launch(args) launch the external program with an empty or
        specific scene as defined by args

        """

        pass

    @abstractmethod
    def step(self, *args: Any, **kwargs: Any) -> Any:
        """
        step(dt) triggers the external program to make a time step
        of defined time updating the state of the environment as defined by
        the robot's actions.

        The will go through each robot in the list and make them act based on
        their control type (position, velocity, acceleration, or torque). Upon
        acting, the other three of the four control types will be updated in
        the internal state of the robot object. The control type is defined
        by the robot object, and not all robot objects support all control
        types.

        """

        pass

    @abstractmethod
    def reset(self, *args: Any, **kwargs: Any) -> Any:
        """
        reset() triggers the external program to reset to the
        original state defined by launch

        """

        pass

    @abstractmethod
    def restart(self, *args: Any, **kwargs: Any) -> Any:
        """
        restart() triggers the external program to close and relaunch
        to thestate defined by launch

        """

        pass

    @abstractmethod
    def close(self, *args: Any, **kwargs: Any) -> Any:
        """
        close() triggers the external program to gracefully close

        """

        pass

    #
    #  Methods to interface with the robots created in other environemnts
    #

    @abstractmethod
    def add(self, *args: Any, **kwargs: Any) -> Any:
        """
        id = add(object) adds the object to the external environment. object must
        be of an appropriate class. This adds a object object to a list of
        objects which will act upon the step() method being called.

        """

        pass

    @abstractmethod
    def remove(self, *args: Any, **kwargs: Any) -> Any:
        """
        remove(id) removes the object from the external environment.

        """

        pass

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
        hold() keeps the backend open i.e. stops the program from closing once
        the main script has finished. This method may need keep an even loop
        running for the backend to keep it responsive.

        """

        pass
