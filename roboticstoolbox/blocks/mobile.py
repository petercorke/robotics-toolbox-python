from typing import Type
import numpy as np
from math import sin, cos, atan2, tan, sqrt, pi

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import time

from bdsim.components import TransferBlock
from bdsim.graphics import GraphicsBlock

from spatialmath import base
from roboticstoolbox import mobile

# ------------------------------------------------------------------------ #
class Bicycle(TransferBlock):
    """
    :blockname:`BICYCLE`
    
    .. table::
       :align: left
    
    +------------+------------+---------+
    | inputs     | outputs    |  states |
    +------------+------------+---------+
    | 2          | 1          | 3       |
    +------------+------------+---------+
    | float      | ndarray(3) |         | 
    +------------+------------+---------+
    """

    nin = 2
    nout = 1
    inlabels = ('v', 'γ')
    outlabels = ('q',)

    def __init__(self, L=1, speed_max=np.inf, accel_max=np.inf, steer_max=0.45 * pi, 
        x0=None, **blockargs):
        r"""
        Create a vehicle model with Bicycle kinematics.

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
        :return: a BICYCLE block
        :rtype: Bicycle instance

        
        Bicycle kinematic model with state/configuration :math:`[x, y, \theta]`.  
        
        **Block ports**
            
            :input v: Vehicle speed (metres/sec).  The velocity limit ``speed_max``
                and acceleration limit ``accel_max`` is
                applied to this input.
            :input γ: Steered wheel angle (radians).  The steering limit ``steer_max``
                is applied to this input.
            
            :output q: configuration (x, y, θ)

        :seealso: :class:`roboticstoolbox.mobile.Bicycle` :class:`Unicycle` :class:`DiffSteer` 
        """
        # TODO: add option to model the effect of steering arms, responds to
        #  gamma dot
        
        super().__init__(nstates=3, **blockargs)

        self.vehicle = mobile.Bicycle(L=L,
            steer_max=steer_max, speed_max=speed_max, accel_max=accel_max)

        if x0 is None:
            self._x0 = np.zeros((self.nstates,))
        else:
            assert len(x0) == self.nstates, "x0 is {:d} long, should be {:d}".format(len(x0), self.nstates)
            self._x0 = x0
            
        self.inport_names(('v', '$\gamma$'))
        self.outport_names(('q',))
        self.state_names(('x', 'y', r'$\theta$'))
        
    def output(self, t):
        return [self._x]  # one output which is ndarray(3)
    
    def deriv(self):
        return self.vehicle.deriv(self._x, self.inputs)
    
# ------------------------------------------------------------------------ #
class Unicycle(TransferBlock):
    r"""
    :blockname:`UNICYCLE`
    
    .. table::
       :align: left
    
    +------------+------------+---------+
    | inputs     | outputs    |  states |
    +------------+------------+---------+
    | 2          | 1          | 3       |
    +------------+------------+---------+
    | float      | ndarray(3) |         | 
    +------------+------------+---------+
    """
    nin = 2
    nout = 1
    inlabels = ('v', 'ω')
    outlabels = ('q',)

    def __init__(self, w=1, speed_max=np.inf, accel_max=np.inf, steer_max=np.inf, 
        a=0, x0=None, **blockargs):
        r"""
        Create a vehicle model with Unicycle kinematics.

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
        :return: a UNICYCLE block
        :rtype: Unicycle instance

        Unicycle kinematic model with state/configuration :math:`[x, y, \theta]`.

        **Block ports**
            
            :input v: Vehicle speed (metres/sec).  The velocity limit ``speed_max`` and
                acceleration limit ``accel_max`` is
                applied to this input.
            :input ω: Angular velocity (radians/sec).  The steering limit ``steer_max``
                is applied to this input.
            
            :output q: configuration (x, y, θ)

        :seealso: :class:`roboticstoolbox.mobile.Unicycle` :class:`Bicycle` :class:`DiffSteer`
        """        
        super().__init__(nstates=3, **blockargs)
        
        if x0 is None:
            self._x0 = np.zeros((self.nstates,))
        else:
            assert len(x0) == self.nstates, "x0 is {:d} long, should be {:d}".format(len(x0), self.nstates)
            self._x0 = x0

        self.vehicle = mobile.Unicycle(w=w,
            steer_max=steer_max, speed_max=speed_max, accel_max=accel_max)

        #TODO, add support for origin shift
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
        
    def output(self, t):
        return self._x
    
    def deriv(self):
        return self.vehicle.deriv(self._x, self.inputs)
    
# ------------------------------------------------------------------------ #
class DiffSteer(TransferBlock):
    """
    :blockname:`DIFFSTEER`
    
    .. table::
       :align: left
    
    +------------+------------+---------+
    | inputs     | outputs    |  states |
    +------------+------------+---------+
    | 2          | 1          | 3       |
    +------------+------------+---------+
    | float      | ndarray(3) |         | 
    +------------+------------+---------+
    """

    nin = 2
    nout = 1

    inlabels = ('ωL', 'ωR')
    outlabels = ('q',)

    def __init__(self, w=1, R=1, speed_max=np.inf, accel_max=np.inf, steer_max=None, 
        a=0, x0=None, **blockargs):
        r"""
        Create a differential steer vehicle model

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
        :return: a DIFFSTEER block
        :rtype: DiffSteer instance
        
        Unicycle kinematic model with state :math:`[x, y, \theta]`, with
        inputs given as wheel angular velocity.

        **Block ports**

            :input ωL: Left-wheel angular velocity (radians/sec).
            :input ωR: Right-wheel angular velocity (radians/sec).
              
            :output q: configuration (x, y, θ)

        The resulting forward velocity and turning rate from ωL and ωR have
        the velocity limit ``speed_max`` and acceleration limit ``accel_max``
        applied, as well as the turning rate limit ``steer_max``.

        .. note:: Wheel velocity is defined such that if both are positive the vehicle
              moves forward.

        :seealso: :class:`roboticstoolbox.mobile.Unicycle` :class:`Bicycle` :class:`Unicycle`
        """
        super().__init__(nstates=3, **blockargs)
        self.type = 'diffsteer'
        self.R = R
        
        if x0 is None:
            self._x0 = np.zeros((slef.nstates,))
        else:
            assert len(x0) == self.nstates, "x0 is {:d} long, should be {:d}".format(len(x0), self.nstates)
            self._x0 = x0

        self.vehicle = mobile.Unicycle(w=w,
            steer_max=steer_max, speed_max=speed_max, accel_max=accel_max)

    def output(self, t):
        return self._x
    
    def deriv(self):
        # compute (v, omega) from left/right wheel speeds
        v = self.R * (self.inputs[0] + self.inputs[1]) / 2
        omega = (self.inputs[1] + self.inputs[0]) / self.W
        return self.vehicle.deriv(self._x, (v, omega))

# ------------------------------------------------------------------------ #

class VehiclePlot(GraphicsBlock):
    """
    :blockname:`VEHICLEPLOT`
    
    .. table::
       :align: left
    
    +--------+---------+---------+
    | inputs | outputs |  states |
    +--------+---------+---------+
    | 1      | 0       | 0       |
    +--------+---------+---------+
    | ndarray|         |         | 
    +--------+---------+---------+
    """

    nin = 1
    nout = 0

    inlabels = ('q',)

    # TODO add ability to render an image instead of an outline
    
    def __init__(self, animation=None, path=None, labels=['X', 'Y'], square=True, init=None, scale=True, **blockargs):
        """
        Create a vehicle animation

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
        :param blockargs: |BlockOptions|
        :type blockargs: dict
        :return: A VEHICLEPLOT block
        :rtype: VehiclePlot instance

        Create a vehicle animation similar to the figure below.

        **Block ports**

            :input q: configuration (x, y, θ)
        
        Notes:
            
            - The ``init`` function is called after the axes are initialized
              and can be used to draw application specific detail on the
              plot. In the example below, this is the dot and star.
            - A dynamic trail, showing path to date can be animated if
              the option ``path`` is set to a linestyle.
        
        .. figure:: ../../figs/rvc4_4.gif
           :width: 500px
           :alt: example of generated graphic

           Example of vehicle display (animated).  The label at the top is the
           block name.
        """
        super().__init__(**blockargs)
        self.xdata = []
        self.ydata = []
        self.type = 'vehicleplot'
        if init is not None:
            assert callable(init), 'graphics init function must be callable'
        self.init = init
        self.square = square

        self.pathstyle = path
        
        if scale != 'auto':
            if len(scale) == 2:
                scale = scale * 2
        self.scale = scale
        self.labels = labels
        
        if animation is None:
            animation = mobile.VehiclePolygon()
        elif not isinstance(animation, mobile.VehicleAnimationBase):
            raise TypeError('animation object must be VehicleAnimationBase subclass')

        self.animation = animation

        
    def start(self, state=None):
        # create the plot
        # super().reset()
        # create the figures
        self.fig = self.create_figure(state)
        self.ax = self.fig.add_subplot(111)
        
        if self.square:
            self.ax.set_aspect('equal')
        print('done')

        self.ax.grid(True)
        self.ax.set_xlabel(self.labels[0])
        self.ax.set_ylabel(self.labels[1])
        self.ax.set_title(self.name)
        if self.scale != 'auto':
            self.ax.set_xlim(*self.scale[0:2])
            self.ax.set_ylim(*self.scale[2:4])
        if self.init is not None:
            self.init(self.ax)

        if isinstance(self.pathstyle, str):
            self.line, = plt.plot(0, 0, self.pathstyle)
        elif isinstance(self.pathstyle, dict):
            self.line, = plt.plot(0, 0, **self.pathstyle)

        self.animation.add()
            
        plt.draw()
        plt.show(block=False)

        super().start()
        
    def step(self, state=None, **kwargs):
        # inputs are set
        xyt = self.inputs[0]

        # update the path line
        self.xdata.append(xyt[0])
        self.ydata.append(xyt[1])
        #plt.figure(self.fig.number)
        if self.pathstyle is not None:
            self.line.set_data(self.xdata, self.ydata)

        # update the vehicle pose
        self.animation.update(xyt)
    
        if self.scale == 'auto':
            self.ax.relim()
            self.ax.autoscale_view()
        super().step(state=state)
        
    def done(self, block=False, **kwargs):
        if self.bd.options.graphics:
            plt.show(block=block)
            
            super().done()
