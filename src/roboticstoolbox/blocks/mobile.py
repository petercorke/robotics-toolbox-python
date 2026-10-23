from __future__ import annotations

import warnings
from collections.abc import Callable
from math import pi
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
from bdsim.block_types import ContinuousBlock, GraphicsBlock
from spatialmath import base
from spatialmath.base.types import R3, ArrayLike3

from roboticstoolbox import mobile
from roboticstoolbox.tools.types import ArrayLike, NDArray

if TYPE_CHECKING:
    from bdsim.components import SimulationState
    from matplotlib.axes import Axes


# ------------------------------------------------------------------------ #
class Bicycle(ContinuousBlock):
    r"""
    :blockname:`BICYCLE`

    Vehicle model with Bicycle kinematics.

    :inputs: 2
    :outputs: 1
    :states: 3

    .. list-table::
        :header-rows: 1

        *   - Port type
            - Port number
            - Types
            - Description
        *   - Input
            - 0
            - float
            - :math:`v`, longitudinal velocity
        *   - Input
            - 1
            - float
            - :math:`\gamma`, steer angle
        *   - Output
            - 0
            - ndarray(3)
            - :math:`\mathit{q} = (x, y, \theta)`, vehicle configuration

    Bicycle kinematic model with velocity and steering inputs and vehicle configuration
    as output.

    Various limits are applied to the inputs:

    * velocity limit ``speed_max``
    * acceleration limit ``accel_max``
    * steering limit ``steer_max``

    .. seealso:: :class:`~roboticstoolbox.mobile.Vehicle.Bicycle` :class:`Unicycle` :class:`DiffSteer`
    """

    nin = 2
    nout = 1
    inlabels = ("v", "γ")
    outlabels = ("q",)

    def __init__(
        self,
        L: float = 1,
        speed_max: float = np.inf,
        accel_max: float = np.inf,
        steer_max: float = 0.45 * pi,
        x0: ArrayLike3 | None = None,
        **blockargs: Any,
    ) -> None:
        r"""
        :param L: Wheelbase, defaults to 1
        :param speed_max: Velocity limit, defaults to math.inf
        :param accel_max: maximum acceleration, defaults to math.inf
        :param steer_max: maximum steered wheel angle, defaults to math.pi*0.45
        :param x0: Initial state, defaults to [0,0,0]
        :param blockargs: |BlockOptions|

        .. deprecated:: 1.4.0
            the ``vlim`` and ``slim`` keywords are deprecated,
            use ``speed_max`` and ``steer_max`` instead.
        """
        # TODO: add option to model the effect of steering arms, responds to
        #  gamma dot

        vlim = blockargs.pop("vlim", None)
        if vlim is not None:
            warnings.warn(
                "vlim is deprecated, use speed_max instead",
                DeprecationWarning,
                stacklevel=2,
            )
            speed_max = vlim

        slim = blockargs.pop("slim", None)
        if slim is not None:
            warnings.warn(
                "slim is deprecated, use steer_max instead",
                DeprecationWarning,
                stacklevel=2,
            )
            steer_max = slim

        super().__init__(nstates=3, **blockargs)
        self.type = "bicycle"

        self.vehicle = mobile.Bicycle(
            L=L, steer_max=steer_max, speed_max=speed_max, accel_max=accel_max
        )

        self._x0: NDArray = (
            np.zeros((self.nstates,))
            if x0 is None
            else base.getvector(x0, self.nstates)
        )

        self.inport_names(("v", r"$\gamma$"))
        self.outport_names(("q",))
        self.state_names(("x", "y", r"$\theta$"))

    def output(self, t: float, inports: list[Any], x: NDArray) -> list[R3]:
        return [x]  # one output which is ndarray(3)

    def deriv(self, t: float, inports: list[Any], x: NDArray) -> NDArray:
        return self.vehicle.deriv(x, inports)


# ------------------------------------------------------------------------ #
class Unicycle(ContinuousBlock):
    r"""
    :blockname:`UNICYCLE`

    Vehicle model with Unicycle kinematics.

    :inputs: 2
    :outputs: 1
    :states: 3

    .. list-table::
        :header-rows: 1

        *   - Port type
            - Port number
            - Types
            - Description
        *   - Input
            - 0
            - float
            - :math:`v`, longitudinal velocity
        *   - Input
            - 1
            - float
            - :math:`\omega`, turn rate
        *   - Output
            - 0
            - ndarray(3)
            - :math:`\mathit{q} = (x, y, \theta)`, vehicle configuration

    Unicycle kinematic model with velocity and steering inputs and vehicle configuration
    as output.

    Various limits are applied to the inputs:

    * velocity limit ``speed_max``
    * acceleration limit ``accel_max``
    * steering limit ``steer_max``

    .. seealso:: :class:`~roboticstoolbox.mobile.Vehicle.Unicycle` :class:`Bicycle` :class:`DiffSteer`

    """

    nin = 2
    nout = 1
    inlabels = ("v", "ω")
    outlabels = ("q",)

    def __init__(
        self,
        w: float = 1,
        speed_max: float = np.inf,
        accel_max: float = np.inf,
        steer_max: float = np.inf,
        # a=0,  implement this, RVC2 p111, change output and deriv
        x0: ArrayLike3 | None = None,
        **blockargs: Any,
    ) -> None:
        r"""

        :param w: vehicle width, defaults to 1
        :param speed_max: Velocity limit, defaults to math.inf
        :param accel_max: maximum acceleration, defaults to math.inf
        :param steer_max: maximum turn rate :math:`\omega`, defaults to math.inf
        :param x0: Initial state, defaults to [0,0,0]
        :param blockargs: |BlockOptions|
        """
        super().__init__(nstates=3, **blockargs)
        self.type = "unicycle"

        self._x0: NDArray = (
            np.zeros((self.nstates,))
            if x0 is None
            else base.getvector(x0, self.nstates)
        )

        self.vehicle = mobile.Unicycle(
            W=w, steer_max=steer_max, speed_max=speed_max, accel_max=accel_max
        )

        # TODO, add support for origin shift
        #         If ``a`` is non-zero then the planar velocity of that point $x=a$
        # can be controlled by

        # .. math::

        #     \begin{pmatrix} v \\ \omega \end{pmatrix} =
        #     \begin{pmatrix}
        #         \cos \theta & \sin \theta \\
        #         -\frac{1}{a}\sin \theta & \frac{1}{a}\cos \theta
        #     \end{pmatrix}\begin{pmatrix}
        #         \dot{x} \\ \dot{y}
        #     \end{pmatrix}

    def output(self, t: float, inports: list[Any], x: NDArray) -> list[R3]:
        return [x]

    def deriv(self, t: float, inports: list[Any], x: NDArray) -> NDArray:
        return self.vehicle.deriv(x, inports)


# ------------------------------------------------------------------------ #
class DiffSteer(ContinuousBlock):
    r"""
    :blockname:`DIFFSTEER`

    Differential steer vehicle model

    :inputs: 2
    :outputs: 1
    :states: 3

    .. list-table::
        :header-rows: 1

        *   - Port type
            - Port number
            - Types
            - Description
        *   - Input
            - 0
            - float
            - :math:`\omega_L`, left-wheel angular velocity (radians/sec).
        *   - Input
            - 1
            - float
            - :math:`\omega_R`, right-wheel angular velocity (radians/sec).
        *   - Output
            - 0
            - ndarray(3)
            - :math:`\mathit{q} = (x, y, \theta)`, vehicle configuration

    Differential steering kinematic model with wheel velocity inputs and vehicle
    configuration as output.

    Various limits are applied to the inputs:

    * velocity limit ``speed_max``
    * acceleration limit ``accel_max``
    * steering limit ``steer_max``

    .. note:: Wheel velocity is defined such that if both are positive the vehicle
            moves forward.

    .. seealso:: :class:`~roboticstoolbox.mobile.Vehicle.DiffSteer` :class:`Bicycle` :class:`Unicycle`
    """

    nin = 2
    nout = 1

    inlabels = ("ωL", "ωR")
    outlabels = ("q",)

    def __init__(
        self,
        w: float = 1,
        R: float = 1,
        speed_max: float = np.inf,
        accel_max: float = np.inf,
        steer_max: float | None = None,
        x0: ArrayLike3 | None = None,
        **blockargs: Any,
    ) -> None:
        r"""
        :param w: vehicle width, defaults to 1
        :param R: Wheel radius, defaults to 1
        :param speed_max: Velocity limit, defaults to math.inf
        :param accel_max: maximum acceleration, defaults to math.inf
        :param steer_max: maximum steering rate, defaults to None (no limit)
        :param x0: Initial state, defaults to [0,0,0]
        :param blockargs: |BlockOptions|
        """
        super().__init__(nstates=3, **blockargs)
        self.type = "diffsteer"
        self.R = R

        self._x0: NDArray = (
            np.zeros((self.nstates,))
            if x0 is None
            else base.getvector(x0, self.nstates)
        )

        self.vehicle = mobile.DiffSteer(
            W=w, steer_max=steer_max, speed_max=speed_max, accel_max=accel_max
        )

    def output(self, t: float, inports: list[Any], x: NDArray) -> list[R3]:
        return [x]

    def deriv(self, t: float, inports: list[Any], x: NDArray) -> NDArray:
        return self.vehicle.deriv(x, inports)


# ------------------------------------------------------------------------ #


class VehiclePlot(GraphicsBlock):
    r"""
    :blockname:`VEHICLEPLOT`

    Vehicle animation

    :inputs: 1
    :outputs: 0
    :states: 0

    .. list-table::
        :header-rows: 1

        *   - Port type
            - Port number
            - Types
            - Description
        *   - Input
            - 0
            - float
            - :math:`\mathit{q} = (x, y, \theta)`, vehicle configuration

    Create a vehicle animation similar to the figure below.

    .. figure:: ../figs/rvc4_4.gif
        :width: 500px
        :alt: example of generated graphic

        Example of vehicle display (animated).  The label at the top is the
        block name.
    """

    nin = 1
    nout = 0
    AXES_POLICY = "subplot"
    TIMESTAMP = True

    inlabels = ("q",)

    # TODO add ability to render an image instead of an outline

    def __init__(
        self,
        animation: mobile.VehicleAnimationBase | None = None,
        path_style: str | dict[str, Any] | None = None,
        labels: tuple[str, str] = ("X", "Y"),
        square: bool = True,
        init_func: Callable[[Axes], None] | None = None,
        limits: str | ArrayLike = "auto",
        shape: str = "car",
        size: float | ArrayLike | None = None,
        polygon_style: dict[str, Any] | None = None,
        **blockargs: Any,
    ) -> None:
        """
        :param animation: Graphical animation of vehicle, a
            :class:`~roboticstoolbox.mobile.Animations.VehicleAnimationBase`
            subclass, defaults to a :class:`~roboticstoolbox.mobile.Animations.VehiclePolygon`
        :param path_style: Matplotlib linestyle string or dict of arguments used to
            plot the path taken by vehicle, defaults to None (no path)
        :param labels: axis labels (xlabel, ylabel), defaults to ("X", "Y")
        :param square: Set aspect ratio to 1, defaults to True
        :param init_func: function called with the axes to initialize graphics,
            defaults to None
        :param limits: axis limits: ``"auto"``, ``[min, max]`` used for both axes, or
            ``[xmin, xmax, ymin, ymax]``, defaults to ``"auto"``
        :param shape: shape of vehicle, passed to the default animation, defaults
            to ``"car"``
        :param size: size of vehicle, passed to the default animation, defaults to
            None (the animation's default)
        :param polygon_style: Matplotlib patch options (``color``, ``alpha``,
            ``edgecolor``, ...) for the default animation
            :class:`~roboticstoolbox.mobile.Animations.VehiclePolygon`, defaults
            to None.  The dict is not modified.
        :param blockargs: |GraphicsBlockOptions|
        :raises TypeError: ``animation`` is not a ``VehicleAnimationBase``

        .. deprecated:: 1.5.0
            the arguments ``path``, ``init``, ``scale`` and ``polyargs`` are
            deprecated, use ``path_style``, ``init_func``, ``limits`` and
            ``polygon_style`` instead.

        .. note::

            - The ``init_func`` function is called after the axes are initialized
              and can be used to draw application specific detail on the
              plot. In the example below, this is the dot and star.
            - If an ``animation`` object is not provided then a default polygon animation is
              created.  It's size and shape can be set by the ``size`` and ``shape`` parameters, which are passed to the animation object,
              or a dictionary of arguments ``polygon_style`` which is passed to the default animation object.
            - A dynamic trail, showing path to date can be animated if
              the option ``path_style`` is set to a linestyle.
        """
        # deprecated names, mapped to the new ones
        for old, new in (
            ("path", "path_style"),
            ("init", "init_func"),
            ("scale", "limits"),
            ("polyargs", "polygon_style"),
        ):
            if old in blockargs:
                warnings.warn(
                    f"{old} is deprecated, use {new} instead",
                    DeprecationWarning,
                    stacklevel=2,
                )
                value = blockargs.pop(old)
                if new == "path_style":
                    path_style = value
                elif new == "init_func":
                    init_func = value
                elif new == "limits":
                    limits = value
                else:
                    polygon_style = value

        super().__init__(**blockargs)
        self.xdata: list[float] = []
        self.ydata: list[float] = []
        # self.type = "vehicleplot"
        if init_func is not None:
            if not callable(init_func):
                raise TypeError("graphics init function must be callable")
        self.init_func = init_func
        self.square = square

        self.path_style = path_style

        self.limits: str | list[float]
        if isinstance(limits, str):
            self.limits = limits
        else:
            limits = list(limits)
            # [min, max] applies to both axes
            self.limits = limits * 2 if len(limits) == 2 else limits
        self.labels = labels

        # a copy, so as not to modify the caller's dict
        polyargs = dict(polygon_style or {})
        polyargs.update(
            {k: v for k, v in [("shape", shape), ("scale", size)] if v is not None}
        )

        self.line: Any = None  # Matplotlib Line2D, if the path is plotted

        if animation is None:
            animation = mobile.VehiclePolygon(**polyargs)
        elif not isinstance(animation, mobile.VehicleAnimationBase):
            raise TypeError("animation object must be VehicleAnimationBase subclass")

        self.animation = animation

    def start(self, simstate: SimulationState) -> None:
        super().start(simstate)

        if not self._enabled:
            return

        # create the plot
        # super().reset()
        # create the figures
        assert self.fig is not None and self.ax is not None  # nosec B101: narrows the type for mypy

        if self.square:
            self.ax.set_aspect("equal")

        self.ax.set_axisbelow(True)  # keeps grid beneath artists
        self.ax.grid(True)
        self.ax.set_xlabel(self.labels[0])
        self.ax.set_ylabel(self.labels[1])
        self.ax.set_title(self.name)
        if not isinstance(self.limits, str):
            self.ax.set_xlim(*self.limits[0:2])
            self.ax.set_ylim(*self.limits[2:4])
        if self.init_func is not None:
            self.init_func(self.ax)

        if isinstance(self.path_style, str):
            (self.line,) = plt.plot(0, 0, self.path_style)
        elif isinstance(self.path_style, dict):
            (self.line,) = plt.plot(0, 0, **self.path_style)

        self.animation.add()

        # plt.draw()
        # plt.show(block=False)

    def step(self, t: float, inports: list[Any]) -> None:
        if not self._enabled:
            return

        # inputs are set
        xyt = inports[0]

        # update the path line
        self.xdata.append(xyt[0])
        self.ydata.append(xyt[1])
        # plt.figure(self.fig.number)
        if self.line is not None:
            self.line.set_data(self.xdata, self.ydata)

        # update the vehicle pose
        self.animation.update(xyt)

        if isinstance(self.limits, str):
            self.ax.relim()
            self.ax.autoscale_view()
        super().step(t, inports)
