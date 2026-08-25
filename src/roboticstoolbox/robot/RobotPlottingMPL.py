"""
MPL-specific plotting methods for robot objects.

This mixin is mixed into BaseRobot.  All matplotlib / PyPlot imports are
deferred to the first call so that importing ``roboticstoolbox`` in a
headless or browser environment (Pyodide / JupyterLite) does not pull in
matplotlib.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal as L

if TYPE_CHECKING:  # pragma: nocover
    from roboticstoolbox.backends.PyPlot import PyPlot
    from roboticstoolbox.backends.PyPlot.EllipsePlot import EllipsePlot
    from roboticstoolbox.tools.types import ArrayLike


class RobotPlottingMPLMixin:
    """Mixin that adds MPL-specific visualisation to BaseRobot subclasses."""

    # ------------------------------------------------------------------
    # Colour helpers
    # ------------------------------------------------------------------

    def linkcolormap(self, linkcolors: list[Any] | str = "viridis"):
        """
        Create a colormap for robot joints.

        :param linkcolors: list of colors or colormap name, defaults to ``"viridis"``
        :returns: the color map

        - ``cm = robot.linkcolormap()`` is an n-element colormap that gives a
          unique color for every link.  The RGBA colors for link ``j`` are
          ``cm(j)``.
        - ``cm = robot.linkcolormap(cmap)`` as above but ``cmap`` is the name
          of a valid matplotlib colormap.  The default, example above, is the
          ``viridis`` colormap.
        - ``cm = robot.linkcolormap(list of colors)`` as above but a
          colormap is created from a list of n color names given as strings,
          tuples or hexstrings.

        .. runblock:: pycon

        >>> import roboticstoolbox as rtb
        >>> robot = rtb.models.DH.Puma560()
        >>> cm = robot.linkcolormap("inferno")
        >>> print(cm(range(6))) # cm(i) is 3rd color in colormap
        >>> cm = robot.linkcolormap(
        ...     ['red', 'g', (0,0.5,0), '#0f8040', 'yellow', 'cyan'])
        >>> print(cm(range(6)))

        .. rubric:: Notes

        - Colormaps have 4-elements: red, green, blue, alpha (RGBA)
        - Names of supported colors and colormaps are defined in the
          matplotlib documentation.

          - `Specifying colors <https://matplotlib.org/3.1.0/tutorials/colors/colors.html>`_
          - `Colormaps <https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html>`_

        """

        from matplotlib import colormaps, colors

        if isinstance(linkcolors, list) and len(linkcolors) == self.n:  # type: ignore[attr-defined]
            return colors.ListedColormap(linkcolors)
        else:
            return colormaps.get_cmap(linkcolors).resampled(6)

    # ------------------------------------------------------------------
    # Ellipse creation
    # ------------------------------------------------------------------

    def fellipse(
        self,
        q: ArrayLike,
        opt: L["trans", "rot"] = "trans",
        unit: L["rad", "deg"] = "rad",
        centre: L["ee"] | ArrayLike = "ee",
        add: bool = True,
    ) -> EllipsePlot:
        """
        Create a force ellipsoid object for plotting with PyPlot.

        :param q: the joint configuration of the robot
        :param opt: ``'trans'`` or ``'rot'`` — plot the translational or
            rotational force ellipsoid
        :param unit: ``'rad'`` or ``'deg'``
        :param centre: centre of the ellipsoid — ``'ee'`` for the
            end-effector or a 3-vector ``[x, y, z]`` in the world frame
        :param add: if ``True``, add the ellipsoid to the active plot environment
        :returns: an EllipsePlot object
        :rtype: EllipsePlot

        ``robot.fellipse(q)`` creates a force ellipsoid for the robot at
        pose ``q``. By default the ellipsoid is centered at the end-effector.

        .. rubric:: Notes

        - By default the ellipsoid related to translational motion is
          drawn.  Use ``opt='rot'`` to draw the rotational velocity ellipsoid.
        - By default the ellipsoid is drawn at the end-effector.  The option
          ``centre`` allows its origin to be set to the specified 3-vector,
          or the string ``"ee"`` ensures it is drawn at the end-effector position.
        """
        import roboticstoolbox as rtb
        from roboticstoolbox.backends.PyPlot.EllipsePlot import EllipsePlot
        from spatialmath.base.argcheck import getunit

        if isinstance(self, rtb.ERobot):  # pragma: nocover
            raise NotImplementedError("ERobot fellipse not implemented yet")

        q = getunit(q, unit)  # type: ignore[assignment]
        ell = EllipsePlot(self, q, "f", opt, centre=centre)

        if add:
            self._maybe_add_ellipse_to_active_env(ell)

        return ell

    def vellipse(
        self,
        q: ArrayLike,
        opt: L["trans", "rot"] = "trans",
        unit: L["rad", "deg"] = "rad",
        centre: L["ee"] | ArrayLike = "ee",
        scale: float = 0.1,
        add: bool = True,
    ) -> EllipsePlot:
        """
        Create a velocity ellipsoid object for plotting with PyPlot.

        :param q: the joint configuration of the robot
        :param opt: ``'trans'`` or ``'rot'`` — plot the translational or
            rotational velocity ellipsoid
        :param unit: ``'rad'`` or ``'deg'``
        :param centre: centre of the ellipsoid — ``'ee'`` for the
            end-effector or a 3-vector ``[x, y, z]`` in the world frame
        :param scale: scale factor for the ellipsoid
        :param add: if ``True``, add the ellipsoid to the active plot environment
        :returns: an EllipsePlot object
        :rtype: EllipsePlot

        ``robot.vellipse(q)`` creates a velocity ellipsoid for the robot at
        pose ``q``. By default the ellipsoid is centered at the end-effector.

        .. rubric:: Notes

        - By default the ellipsoid related to translational motion is
          drawn.  Use ``opt='rot'`` to draw the rotational velocity ellipsoid.
        - By default the ellipsoid is drawn at the end-effector.  The option
          ``centre`` allows its origin to be set to the specified 3-vector,
          or the string ``"ee"`` ensures it is drawn at the end-effector position.
        """
        import roboticstoolbox as rtb
        from roboticstoolbox.backends.PyPlot.EllipsePlot import EllipsePlot
        from spatialmath.base.argcheck import getunit

        if isinstance(self, rtb.ERobot):  # pragma: nocover
            raise NotImplementedError("ERobot vellipse not implemented yet")

        q = getunit(q, unit)  # type: ignore[assignment]
        ell = EllipsePlot(self, q, "v", opt, centre=centre, scale=scale)

        if add:
            self._maybe_add_ellipse_to_active_env(ell)

        return ell

    def _maybe_add_ellipse_to_active_env(self, ellipse: EllipsePlot) -> None:
        """Add ellipse to the most recently active PyPlot environment, if available."""
        env = self._active_plot_env  # type: ignore[attr-defined]
        if env is None:
            return

        if not getattr(env, "supports_ellipse", False):
            return

        try:
            env.add(ellipse)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Ellipse plotting
    # ------------------------------------------------------------------

    def plot_ellipse(
        self,
        ellipse: EllipsePlot,
        block: bool = True,
        limits: ArrayLike | None = None,
        jointaxes: bool = True,
        eeframe: bool = True,
        shadow: bool = True,
        name: bool = True,
    ) -> PyPlot:
        """
        Plot an ellipsoid.

        :param ellipse: the ellipsoid to plot
        :param block: block operation of the code and keep the figure open
        :param limits: custom view limits ``[x1, x2, y1, y2, z1, z2]``; autoscales if not supplied
        :param jointaxes: plot an arrow indicating the joint axis
        :param eeframe: plot the end-effector coordinate frame
        :param shadow: plot a shadow of the robot in the x-y plane
        :param name: plot the name of the robot near its base
        :returns: a reference to the PyPlot object controlling the matplotlib figure
        :rtype: PyPlot

        ``robot.plot_ellipse(ellipsoid)`` displays the ellipsoid.
        """
        from roboticstoolbox.backends.PyPlot import PyPlot
        from roboticstoolbox.backends.PyPlot.EllipsePlot import EllipsePlot

        if not isinstance(ellipse, EllipsePlot):  # pragma: nocover
            raise TypeError(
                "ellipse must be of type roboticstoolbox.backend.PyPlot.EllipsePlot"
            )

        env = PyPlot()

        env.launch(ellipse.robot.name + " " + ellipse.name, limits=limits)
        env.add(ellipse, jointaxes=jointaxes, eeframe=eeframe, shadow=shadow, name=name)

        if block:  # pragma: nocover
            env.hold()

        return env

    def plot_fellipse(
        self,
        q: ArrayLike | None,
        block: bool = True,
        fellipse: EllipsePlot | None = None,
        limits: ArrayLike | None = None,
        opt: L["trans", "rot"] = "trans",
        centre: L["ee"] | ArrayLike = "ee",
        jointaxes: bool = True,
        eeframe: bool = True,
        shadow: bool = True,
        name: bool = True,
    ) -> PyPlot:
        """
        Plot the force ellipsoid for a manipulator.

        :param q: the joint configuration of the robot
        :param block: block operation of the code and keep the figure open
        :param fellipse: a pre-built force ellipsoid to plot
        :param limits: custom view limits ``[x1, x2, y1, y2, z1, z2]``; autoscales if not supplied
        :param opt: ``'trans'`` or ``'rot'`` — plot the translational or rotational force ellipsoid
        :param centre: coordinates to plot the ellipse — ``[x, y, z]`` or ``"ee"``
        :param jointaxes: plot an arrow indicating the joint axis
        :param eeframe: plot the end-effector coordinate frame
        :param shadow: plot a shadow of the robot in the x-y plane
        :param name: plot the name of the robot near its base
        :raises ValueError: if neither ``q`` nor ``fellipse`` is supplied
        :returns: a reference to the PyPlot object controlling the matplotlib figure
        :rtype: PyPlot

        ``robot.plot_fellipse(q)`` displays the force ellipsoid for the robot
        at pose ``q``. The plot will autoscale with an aspect ratio of 1.

        ``robot.plot_fellipse(vellipse=ell)`` uses a pre-built ellipse object.

        .. rubric:: Notes

        - By default the ellipsoid related to translational motion is drawn.
          Use ``opt='rot'`` to draw the rotational velocity ellipsoid.
        - By default the ellipsoid is drawn at the origin.  Use ``centre``
          to specify a 3-vector, or ``"ee"`` to draw at the end-effector.
        """
        import roboticstoolbox as rtb

        if isinstance(self, rtb.ERobot):  # pragma: nocover
            raise NotImplementedError("Ellipse Plotting of ERobot's not implemented yet")

        if fellipse is None and q is not None:
            fellipse = self.fellipse(q, opt=opt, centre=centre, add=False)
        elif fellipse is None:
            raise ValueError("Must specify either q or fellipse")  # pragma: nocover

        return self.plot_ellipse(
            fellipse,
            block=block,
            limits=limits,
            jointaxes=jointaxes,
            eeframe=eeframe,
            shadow=shadow,
            name=name,
        )

    def plot_vellipse(
        self,
        q: ArrayLike | None,
        block: bool = True,
        vellipse: EllipsePlot | None = None,
        limits: ArrayLike | None = None,
        opt: L["trans", "rot"] = "trans",
        centre: L["ee"] | ArrayLike = "ee",
        jointaxes: bool = True,
        eeframe: bool = True,
        shadow: bool = True,
        name: bool = True,
    ) -> PyPlot:
        """
        Plot the velocity ellipsoid for a manipulator.

        :param q: the joint configuration of the robot
        :param block: block operation of the code and keep the figure open
        :param vellipse: a pre-built velocity ellipsoid to plot
        :param limits: custom view limits ``[x1, x2, y1, y2, z1, z2]``; autoscales if not supplied
        :param opt: ``'trans'`` or ``'rot'`` — plot the translational or rotational velocity ellipsoid
        :param centre: coordinates to plot the ellipse — ``[x, y, z]`` or ``"ee"``
        :param jointaxes: plot an arrow indicating the joint axis
        :param eeframe: plot the end-effector coordinate frame
        :param shadow: plot a shadow of the robot in the x-y plane
        :param name: plot the name of the robot near its base
        :raises ValueError: if neither ``q`` nor ``vellipse`` is supplied
        :returns: a reference to the PyPlot object controlling the matplotlib figure
        :rtype: PyPlot

        ``robot.plot_vellipse(q)`` displays the velocity ellipsoid for the robot
        at pose ``q``. The plot will autoscale with an aspect ratio of 1.

        ``robot.plot_vellipse(vellipse=ell)`` uses a pre-built ellipse object.

        .. rubric:: Notes

        - By default the ellipsoid related to translational motion is drawn.
          Use ``opt='rot'`` to draw the rotational velocity ellipsoid.
        - By default the ellipsoid is drawn at the origin.  Use ``centre``
          to specify a 3-vector, or ``"ee"`` to draw at the end-effector.
        """
        import roboticstoolbox as rtb

        if isinstance(self, rtb.ERobot):  # pragma: nocover
            raise NotImplementedError("Ellipse Plotting of ERobot's not implemented yet")

        if vellipse is None and q is not None:
            vellipse = self.vellipse(q=q, opt=opt, centre=centre, add=False)
        elif vellipse is None:
            raise ValueError("Must specify either q or vellipse")  # pragma: nocover

        return self.plot_ellipse(
            vellipse,
            block=block,
            limits=limits,
            jointaxes=jointaxes,
            eeframe=eeframe,
            shadow=shadow,
            name=name,
        )
