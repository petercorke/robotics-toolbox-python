"""
MPL-specific plotting methods for robot objects.

This mixin is mixed into BaseRobot.  All matplotlib / PyPlot imports are
deferred to the first call so that importing ``roboticstoolbox`` in a
headless or browser environment (Pyodide / JupyterLite) does not pull in
matplotlib.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Union

from typing_extensions import Literal as L

if TYPE_CHECKING:  # pragma: nocover
    from roboticstoolbox.backends.PyPlot import PyPlot
    from roboticstoolbox.backends.PyPlot.EllipsePlot import EllipsePlot
    from roboticstoolbox.tools.types import ArrayLike


class RobotPlottingMPLMixin:
    """Mixin that adds MPL-specific visualisation to BaseRobot subclasses."""

    # ------------------------------------------------------------------
    # Colour helpers
    # ------------------------------------------------------------------

    def linkcolormap(self, linkcolors: Union[List[Any], str] = "viridis"):
        """
        Create a colormap for robot joints

        - ``cm = robot.linkcolormap()`` is an n-element colormap that gives a
          unique color for every link.  The RGBA colors for link ``j`` are
          ``cm(j)``.
        - ``cm = robot.linkcolormap(cmap)`` as above but ``cmap`` is the name
          of a valid matplotlib colormap.  The default, example above, is the
          ``viridis`` colormap.
        - ``cm = robot.linkcolormap(list of colors)`` as above but a
          colormap is created from a list of n color names given as strings,
          tuples or hexstrings.

        Parameters
        ----------
        linkcolors
            list of colors or colormap, defaults to "viridis"

        Returns
        -------
        color map
            the color map

        Examples
        --------
        .. runblock:: pycon
        >>> import roboticstoolbox as rtb
        >>> robot = rtb.models.DH.Puma560()
        >>> cm = robot.linkcolormap("inferno")
        >>> print(cm(range(6))) # cm(i) is 3rd color in colormap
        >>> cm = robot.linkcolormap(
        ...     ['red', 'g', (0,0.5,0), '#0f8040', 'yellow', 'cyan'])
        >>> print(cm(range(6)))

        Notes
        -----
        - Colormaps have 4-elements: red, green, blue, alpha (RGBA)
        - Names of supported colors and colormaps are defined in the
          matplotlib documentation.

          - `Specifying colors <https://matplotlib.org/3.1.0/tutorials/colors/colors.html>`_
          - `Colormaps <https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html>`_

        """  # noqa

        from matplotlib import cm, colors

        if isinstance(linkcolors, list) and len(linkcolors) == self.n:  # pragma: nocover
            return colors.ListedColormap(linkcolors)
        else:  # pragma: nocover
            return cm.get_cmap(linkcolors, 6)

    # ------------------------------------------------------------------
    # Ellipse creation
    # ------------------------------------------------------------------

    def fellipse(
        self,
        q: "ArrayLike",
        opt: L["trans", "rot"] = "trans",  # noqa
        unit: L["rad", "deg"] = "rad",  # noqa
        centre: Union[L["ee"], "ArrayLike"] = "ee",  # noqa
        add: bool = True,
    ) -> "EllipsePlot":
        """
        Create a force ellipsoid object for plotting with PyPlot

        ``robot.fellipse(q)`` creates a force ellipsoid for the robot at
        pose ``q``. By default the ellipsoid is centered at the end-effector.

        Parameters
        ----------
        q
            The joint configuration of the robot.
        opt
            'trans' or 'rot' will plot either the translational or
            rotational force ellipsoid
        unit
            'rad' or 'deg' will plot the ellipsoid in radians or
            degrees
        centre
            The centre of the ellipsoid, either 'ee' for the end-effector
            or a 3-vector [x, y, z] in the world frame

        Returns
        -------
        env
            An EllipsePlot object

        Notes
        -----
        - By default the ellipsoid related to translational motion is
            drawn.  Use ``opt='rot'`` to draw the rotational velocity
            ellipsoid.
        - By default the ellipsoid is drawn at the end-effector.  The option
            ``centre`` allows its origin to set to set to the specified
            3-vector, or the string "ee" ensures it is drawn at the
            end-effector position.

        """
        import roboticstoolbox as rtb
        from roboticstoolbox.backends.PyPlot.EllipsePlot import EllipsePlot
        from spatialmath.base.argcheck import getunit

        if isinstance(self, rtb.ERobot):  # pragma: nocover
            raise NotImplementedError("ERobot fellipse not implemented yet")

        q = getunit(q, unit)
        ell = EllipsePlot(self, q, "f", opt, centre=centre)

        if add:
            self._maybe_add_ellipse_to_active_env(ell)

        return ell

    def vellipse(
        self,
        q: "ArrayLike",
        opt: L["trans", "rot"] = "trans",  # noqa
        unit: L["rad", "deg"] = "rad",  # noqa
        centre: Union[L["ee"], "ArrayLike"] = "ee",  # noqa
        scale: float = 0.1,
        add: bool = True,
    ) -> "EllipsePlot":
        """
        Create a velocity ellipsoid object for plotting with PyPlot

        ``robot.vellipse(q)`` creates a force ellipsoid for the robot at
        pose ``q``. By default the ellipsoid is centered at the end-effector.

        Parameters
        ----------
        q
            The joint configuration of the robot.
        opt
            'trans' or 'rot' will plot either the translational or
            rotational force ellipsoid
        unit
            'rad' or 'deg' will plot the ellipsoid in radians or
            degrees
        centre
            The centre of the ellipsoid, either 'ee' for the end-effector
            or a 3-vector [x, y, z] in the world frame
        scale
            The scale factor for the ellipsoid

        Returns
        -------
        env
            An EllipsePlot object

        Notes
        -----
        - By default the ellipsoid related to translational motion is
            drawn.  Use ``opt='rot'`` to draw the rotational velocity
            ellipsoid.
        - By default the ellipsoid is drawn at the end-effector.  The option
            ``centre`` allows its origin to set to set to the specified
            3-vector, or the string "ee" ensures it is drawn at the
            end-effector position.

        """
        import roboticstoolbox as rtb
        from roboticstoolbox.backends.PyPlot.EllipsePlot import EllipsePlot
        from spatialmath.base.argcheck import getunit

        if isinstance(self, rtb.ERobot):  # pragma: nocover
            raise NotImplementedError("ERobot vellipse not implemented yet")

        q = getunit(q, unit)
        ell = EllipsePlot(self, q, "v", opt, centre=centre, scale=scale)

        if add:
            self._maybe_add_ellipse_to_active_env(ell)

        return ell

    def _maybe_add_ellipse_to_active_env(self, ellipse: "EllipsePlot") -> None:
        """Add ellipse to the most recently active PyPlot environment, if available."""
        env = self._active_plot_env
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
        ellipse: "EllipsePlot",
        block: bool = True,
        limits: Union["ArrayLike", None] = None,
        jointaxes: bool = True,
        eeframe: bool = True,
        shadow: bool = True,
        name: bool = True,
    ) -> "PyPlot":
        """
        Plot the an ellipsoid

        ``robot.plot_ellipse(ellipsoid)`` displays the ellipsoid.

        Parameters
        ----------
        ellipse
            the ellipsoid to plot
        block
            Block operation of the code and keep the figure open
        limits
            Custom view limits for the plot. If not supplied will
            autoscale, [x1, x2, y1, y2, z1, z2]
        jointaxes
            (Plot Option) Plot an arrow indicating the axes in
            which the joint revolves around(revolute joint) or translates
            along (prosmatic joint)
        eeframe
            (Plot Option) Plot the end-effector coordinate frame
            at the location of the end-effector. Uses three arrows, red,
            green and blue to indicate the x, y, and z-axes.
        shadow
            (Plot Option) Plot a shadow of the robot in the x-y
            plane
        name
            (Plot Option) Plot the name of the robot near its base

        Returns
        -------
        env
            A reference to the PyPlot object which controls the
            matplotlib figure

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
        q: Union["ArrayLike", None],
        block: bool = True,
        fellipse: Union["EllipsePlot", None] = None,
        limits: Union["ArrayLike", None] = None,
        opt: L["trans", "rot"] = "trans",  # noqa
        centre: Union[L["ee"], "ArrayLike"] = "ee",  # noqa
        jointaxes: bool = True,
        eeframe: bool = True,
        shadow: bool = True,
        name: bool = True,
    ) -> "PyPlot":
        """
        Plot the force ellipsoid for manipulator

        ``robot.plot_fellipse(q)`` displays the velocity ellipsoid for the
        robot at pose ``q``. The plot will autoscale with an aspect ratio
        of 1.

        ``robot.plot_fellipse(vellipse)`` specifies a custon ellipse to plot.

        Parameters
        ----------
        q
            The joint configuration of the robot
        block
            Block operation of the code and keep the figure open
        fellipse
            the vellocity ellipsoid to plot
        limits
            Custom view limits for the plot. If not supplied will
            autoscale, [x1, x2, y1, y2, z1, z2]
        opt
            'trans' or 'rot' will plot either the translational or
            rotational force ellipsoid
        centre
            The coordinates to plot the fellipse [x, y, z] or "ee"
            to plot at the end-effector location
        jointaxes
            (Plot Option) Plot an arrow indicating the axes in
            which the joint revolves around(revolute joint) or translates
            along (prosmatic joint)
        eeframe
            (Plot Option) Plot the end-effector coordinate frame
            at the location of the end-effector. Uses three arrows, red,
            green and blue to indicate the x, y, and z-axes.
        shadow
            (Plot Option) Plot a shadow of the robot in the x-y
            plane
        name
            (Plot Option) Plot the name of the robot near its base

        Raises
        ------
        ValueError
            q or fellipse must be supplied

        Returns
        -------
        env
            A reference to the PyPlot object which controls the
            matplotlib figure

        Notes
        -----
        - By default the ellipsoid related to translational motion is
            drawn.  Use ``opt='rot'`` to draw the rotational velocity
            ellipsoid.
        - By default the ellipsoid is drawn at the origin.  The option
            ``centre`` allows its origin to set to set to the specified
            3-vector, or the string "ee" ensures it is drawn at the
            end-effector position.

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
        q: Union["ArrayLike", None],
        block: bool = True,
        vellipse: Union["EllipsePlot", None] = None,
        limits: Union["ArrayLike", None] = None,
        opt: L["trans", "rot"] = "trans",  # noqa
        centre: Union[L["ee"], "ArrayLike"] = "ee",  # noqa
        jointaxes: bool = True,
        eeframe: bool = True,
        shadow: bool = True,
        name: bool = True,
    ) -> "PyPlot":
        """
        Plot the velocity ellipsoid for manipulator

        ``robot.plot_vellipse(q)`` displays the velocity ellipsoid for the
        robot at pose ``q``. The plot will autoscale with an aspect ratio
        of 1.

        ``robot.plot_vellipse(vellipse)`` specifies a custon ellipse to plot.

        block
            Block operation of the code and keep the figure open
        q
            The joint configuration of the robot
        vellipse
            the vellocity ellipsoid to plot
        limits
            Custom view limits for the plot. If not supplied will
            autoscale, [x1, x2, y1, y2, z1, z2]
        opt
            'trans' or 'rot' will plot either the translational or
            rotational velocity ellipsoid
        centre
            The coordinates to plot the vellipse [x, y, z] or "ee"
            to plot at the end-effector location
        jointaxes
            (Plot Option) Plot an arrow indicating the axes in
            which the joint revolves around(revolute joint) or translates
            along (prosmatic joint)
        eeframe
            (Plot Option) Plot the end-effector coordinate frame
            at the location of the end-effector. Uses three arrows, red,
            green and blue to indicate the x, y, and z-axes.
        shadow
            (Plot Option) Plot a shadow of the robot in the x-y
            plane
        name
            (Plot Option) Plot the name of the robot near its base

        Raises
        ------
        ValueError
            q or fellipse must be supplied

        Returns
        -------
        env
            A reference to the PyPlot object which controls the
            matplotlib figure

        Notes
        -----
        - By default the ellipsoid related to translational motion is
            drawn.  Use ``opt='rot'`` to draw the rotational velocity
            ellipsoid.
        - By default the ellipsoid is drawn at the origin.  The option
            ``centre`` allows its origin to set to set to the specified
            3-vector, or the string "ee" ensures it is drawn at the
            end-effector position.

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
