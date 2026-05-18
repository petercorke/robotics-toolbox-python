from typing import Type
import numpy as np
from math import sin, cos, atan2, tan, sqrt, pi

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import time

from bdsim.components import TransferBlock
from bdsim.graphics import GraphicsBlock

from roboticstoolbox import mobile

# ------------------------------------------------------------------------ #
class Bicycle(TransferBlock):
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

    :seealso: :class:`~roboticstoolbox.mobile.Vehicle.Bicycle` :class:`Unicycle` :class:`DiffSteer`
    """

    nin = 2
    nout = 1
    inlabels = ("v", "γ")
    outlabels = ("q",)

    def __init__(
        self,
        L=1,
        speed_max=np.inf,
        accel_max=np.inf,
        steer_max=0.45 * pi,
        x0=None,
        **blockargs,
    ):
        r"""
        :param L: Wheelbase, defaults to 1
        :type L: float, optional
        :param speed_max: Velocity limit, defaults to math.inf
        :type speed_max: float, optional
        :param accel_max: maximum acceleration, defaults to math.inf
        :type accel_max: float, optional
        :param steer_max: maximum steered wheel angle, defaults to math.pi*0.45
        :type steer_max: float, optional
        :param x0: Initial state, defaults to [0,0,0]
        :type x0: array_like(3), optional
        :param blockargs: |BlockOptions|
        :type blockargs: dict
        """
        # TODO: add option to model the effect of steering arms, responds to
        #  gamma dot

        super().__init__(nstates=3, **blockargs)
        self.type = "bicycle"

        self.vehicle = mobile.Bicycle(
            L=L, steer_max=steer_max, speed_max=speed_max, accel_max=accel_max
        )

        if x0 is None:
            self._x0 = np.zeros((self.nstates,))
        else:
            assert len(x0) == self.nstates, "x0 is {:d} long, should be {:d}".format(
                len(x0), self.nstates
            )
            self._x0 = x0

        self.inport_names(("v", "$\gamma$"))
        self.outport_names(("q",))
        self.state_names(("x", "y", r"$\theta$"))

    def output(self, t, inports, x):
        return [x]  # one output which is ndarray(3)

    def deriv(self, t, inports, x):
        return self.vehicle.deriv(x, inports)


# ------------------------------------------------------------------------ #
class Unicycle(TransferBlock):
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

    :seealso: :class:`~roboticstoolbox.mobile.Vehicle.Unicycle` :class:`Bicycle` :class:`DiffSteer`

    """
    nin = 2
    nout = 1
    inlabels = ("v", "ω")
    outlabels = ("q",)

    def __init__(
        self,
        w=1,
        speed_max=np.inf,
        accel_max=np.inf,
        steer_max=np.inf,
        # a=0,  implement this, RVC2 p111, change output and deriv
        x0=None,
        **blockargs,
    ):
        r"""

        :param w: vehicle width, defaults to 1
        :type w: float, optional
        :param speed_max: Velocity limit, defaults to math.inf
        :type speed_max: float, optional
        :param accel_max: maximum acceleration, defaults to math.inf
        :type accel_max: float, optional
        :param steer_max: maximum turn rate :math:`\omega`, defaults to math.inf
        :type steer_max: float, optional
        :param x0: Inital state, defaults to [0,0,0]
        :type x0: array_like(3), optional
        :param blockargs: |BlockOptions|
        :type blockargs: dict
        """
        super().__init__(nstates=3, **blockargs)
        self.type = "unicycle"

        if x0 is None:
            self._x0 = np.zeros((self.nstates,))
        else:
            assert len(x0) == self.nstates, "x0 is {:d} long, should be {:d}".format(
                len(x0), self.nstates
            )
            self._x0 = x0

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

    def output(self, t, inports, x):
        return [x]

    def deriv(self, t, inports, x):
        return self.vehicle.deriv(x, inports)


# ------------------------------------------------------------------------ #
class DiffSteer(TransferBlock):
    """
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

    :seealso: :class:`~roboticstoolbox.mobile.Vehicle.Diffsteer` :class:`Bicycle` :class:`Unicycle`
    """

    nin = 2
    nout = 1

    inlabels = ("ωL", "ωR")
    outlabels = ("q",)

    def __init__(
        self,
        w=1,
        R=1,
        speed_max=np.inf,
        accel_max=np.inf,
        steer_max=None,
        a=0,
        x0=None,
        **blockargs,
    ):
        r"""
        :param w: vehicle width, defaults to 1
        :type w: float, optional
        :param R: Wheel radius, defaults to 1
        :type R: float, optional
        :param speed_max: Velocity limit, defaults to 1
        :type speed_max: float, optional
        :param accel_max: maximum acceleration, defaults to math.inf
        :type accel_max: float, optional
        :param steer_max: maximum steering rate, defaults to 1
        :type steer_max: float, optional
        :param x0: Inital state, defaults to None
        :type x0: array_like, optional
        :param blockargs: |BlockOptions|
        :type blockargs: dict
        """
        super().__init__(nstates=3, **blockargs)
        self.type = "diffsteer"
        self.R = R

        if x0 is None:
            self._x0 = np.zeros((slef.nstates,))
        else:
            assert len(x0) == self.nstates, "x0 is {:d} long, should be {:d}".format(
                len(x0), self.nstates
            )
            self._x0 = x0

        self.vehicle = mobile.DiffSteer(
            W=w, steer_max=steer_max, speed_max=speed_max, accel_max=accel_max
        )

    def output(self, t, inports, x):
        return [x]

    def deriv(self, t, inports, x):
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

    inlabels = ("q",)

    # TODO add ability to render an image instead of an outline

    def __init__(
        self,
        animation=None,
        path=None,
        labels=["X", "Y"],
        square=True,
        init=None,
        scale="auto",
        polyargs={},
        **blockargs,
    ):
        """
        :param animation: Graphical animation of vehicle, defaults to None
        :type animation: VehicleAnimation subclass, optional
        :param path: linestyle to plot path taken by vehicle, defaults to None
        :type path: str or dict, optional
        :param labels: axis labels (xlabel, ylabel), defaults to ["X","Y"]
        :type labels: array_like(2) or list
        :param square: Set aspect ratio to 1, defaults to True
        :type square: bool, optional
        :param init: function to initialize graphics, defaults to None
        :type init: callable, optional
        :param scale: scale of plot, defaults to "auto"
        :type scale: list or str, optional
        :param polyargs: arguments passed to :meth:`Animation.Polygon`
        :type polyargs: dict
        :param blockargs: |BlockOptions|
        :type blockargs: dict

        .. note::

            - The ``init`` function is called after the axes are initialized
              and can be used to draw application specific detail on the
              plot. In the example below, this is the dot and star.
            - A dynamic trail, showing path to date can be animated if
              the option ``path`` is set to a linestyle.
        """
        super().__init__(**blockargs)
        self.xdata = []
        self.ydata = []
        # self.type = "vehicleplot"
        if init is not None:
            assert callable(init), "graphics init function must be callable"
        self.init = init
        self.square = square

        self.pathstyle = path

        if scale != "auto":
            if len(scale) == 2:
                scale = scale * 2
        self.scale = scale
        self.labels = labels

        if animation is None:
            animation = mobile.VehiclePolygon(**polyargs)
        elif not isinstance(animation, mobile.VehicleAnimationBase):
            raise TypeError("animation object must be VehicleAnimationBase subclass")

        self.animation = animation

    def start(self, simstate):
        super().start(simstate)

        # create the plot
        # super().reset()
        # create the figures
        self.fig = self.create_figure(simstate)
        self.ax = self.fig.add_subplot(111)

        if self.square:
            self.ax.set_aspect("equal")
        print("done")

        self.ax.grid(True)
        self.ax.set_xlabel(self.labels[0])
        self.ax.set_ylabel(self.labels[1])
        self.ax.set_title(self.name)
        if self.scale != "auto":
            self.ax.set_xlim(*self.scale[0:2])
            self.ax.set_ylim(*self.scale[2:4])
        if self.init is not None:
            self.init(self.ax)

        if isinstance(self.pathstyle, str):
            (self.line,) = plt.plot(0, 0, self.pathstyle)
        elif isinstance(self.pathstyle, dict):
            (self.line,) = plt.plot(0, 0, **self.pathstyle)

        self.animation.add()

        # plt.draw()
        # plt.show(block=False)

    def step(self, t, inports):
        # inputs are set
        xyt = inports[0]

        # update the path line
        self.xdata.append(xyt[0])
        self.ydata.append(xyt[1])
        # plt.figure(self.fig.number)
        if self.pathstyle is not None:
            self.line.set_data(self.xdata, self.ydata)

        # update the vehicle pose
        self.animation.update(xyt)

        if isinstance(self.scale, str) and self.scale == "auto":
            self.ax.relim()
            self.ax.autoscale_view()
        super().step(t, inports)
