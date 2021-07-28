import numpy as np
from math import sin, cos, atan2, tan, sqrt, pi

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import time

from bdsim.components import TransferBlock
from bdsim.graphics import GraphicsBlock

from spatialmath import base
from roboticstoolbox import Bicycle, Unicycle

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

    def __init__(self, *inputs, x0=None, L=1, vlim=1, slim=1, **kwargs):
        r"""
        Create a vehicle model with Bicycle kinematics.

        :param ``*inputs``: Optional incoming connections
        :type ``*inputs``: Block or Plug
        :param x0: Inital state, defaults to None
        :type x0: array_like, optional
        :param L: Wheelbase, defaults to 1
        :type L: float, optional
        :param vlim: Velocity limit, defaults to 1
        :type vlim: float, optional
        :param slim: Steering limit, defaults to 1
        :type slim: float, optional
        :param ``**kwargs``: common Block options
        :return: a BICYCLE block
        :rtype: Bicycle instance

        
        Bicycle kinematic model with state :math:`[x, y, \theta]`.  
        
        **Block ports**
            
            :input v: Vehicle speed (metres/sec).  The velocity limit ``vlim`` is
                applied to the magnitude of this input.
            :input γ: Steering wheel angle (radians).  The steering limit ``slim``
                is applied to the magnitude of this input.
            
            :output q: configuration (x, y, θ)

        :seealso: :class:`Unicycle` :class:`DiffSteer`
        """
        super().__init__(inputs=inputs, **kwargs)

        self.nstates = 3
        self.vlim = vlim
        self.slim = slim

        self.L = L
        if x0 is None:
            self._x0 = np.zeros((self.nstates,))
        else:
            assert len(x0) == self.nstates, "x0 is {:d} long, should be {:d}".format(len(x0), self.nstates)
            self._x0 = x0
            
        self.inport_names(('v', '$\gamma$'))
        self.outport_names(('x', 'y', r'$\theta$'))
        self.state_names(('x', 'y', r'$\theta$'))
        
    def output(self, t):
        return [self._x]  # one output which is ndarray(3)
    
    def deriv(self):
        theta = self._x[2]
        
        # get inputs and clip them
        v = self.inputs[0]
        v = min(self.vlim, max(v, -self.vlim))
        gamma = self.inputs[1]
        gamma = min(self.slim, max(gamma, -self.slim))
        
        xd = np.r_[v * cos(theta), v * sin(theta), v * tan(gamma)/self.L ]
        return xd
    
# ------------------------------------------------------------------------ #
class Unicycle(TransferBlock):
    """
    :blockname:`UNICYCLE`
    
    .. table::
       :align: left
    
    +------------+---------+---------+
    | inputs     | outputs |  states |
    +------------+---------+---------+
    | 2          | 3       | 3       |
    +------------+---------+---------+
    | float      | float   |         | 
    +------------+---------+---------+
    """
    nin = 2
    nout = 1

    def __init__(self, *inputs, x0=None, **kwargs):
        r"""
        Create a vehicle model with Unicycle kinematics.

        :param ``*inputs``: Optional incoming connections
        :type ``*inputs``: Block or Plug
        :param x0: Inital state, defaults to None
        :type x0: array_like, optional
        :param ``*inputs``: Optional incoming connections
        :type ``*inputs``: Block or Plug
        :param ``**kwargs``: common Block options
        :return: a UNICYCLE block
        :rtype: Unicycle instance


        Unicycle kinematic model with state :math:`[x, y, \theta]`.
            
        **Block ports**
            
            :input v: Vehicle speed (metres/sec).  The velocity limit ``vlim`` is
                applied to the magnitude of this input.
            :input ω: Angular velocity (radians/sec).  The steering limit ``slim``
                is applied to the magnitude of this input.
            
            :output q: configuration (x, y, θ)

        :seealso: :class:`Bicycle` :class:`DiffSteer`
        """        
        super().__init__(inputs=inputs, **kwargs)
        self.nstates = 3
        
        if x0 is None:
            self._x0 = np.zeros((slef.nstates,))
        else:
            assert len(x0) == self.nstates, "x0 is {:d} long, should be {:d}".format(len(x0), self.nstates)
            self._x0 = x0
        
    def output(self, t):
        return self._x
    
    def deriv(self):
        theta = self._x[2]
        v = self.inputs[0]
        omega = self.inputs[1]
        xd = np.r_[v * cos(theta), v * sin(theta), omega]
        return xd
    
# ------------------------------------------------------------------------ #
class DiffSteer(TransferBlock):
    """
    :blockname:`DIFFSTEER`
    
    .. table::
       :align: left
    
    +------------+---------+---------+
    | inputs     | outputs |  states |
    +------------+---------+---------+
    | 2          | 3       | 3       |
    +------------+---------+---------+
    | float      | float   |         | 
    +------------+---------+---------+
    """

    nin = 2
    nout = 1

    inlabels = ('ωL', 'ωR')
    outlabels = ('q',)

    def __init__(self, *inputs, R=1, W=1, x0=None, **kwargs):
        """
        Create a differential steer vehicle model

        :param ``*inputs``: Optional incoming connections
        :type ``*inputs``: Block or Plug
        :param x0: Inital state, defaults to None
        :type x0: array_like, optional
        :param R: Wheel radius, defaults to 1
        :type R: float, optional
        :param W: Wheel separation in lateral direction, defaults to 1
        :type W: float, optional
        :param ``**kwargs``: common Block options
        :return: a DIFFSTEER block
        :rtype: DifSteer instance
        
        Unicycle kinematic model with state :math:`[x, y, \theta]`, with
        with inputs given as wheel angular velocity.

        **Block ports**

            :input ωL: Left-wheel angular velocity (radians/sec).
            :input ωR: Right-wheel angular velocity (radians/sec).
              
            :output q: configuration (x, y, θ)

        .. note:: Wheel velocity is defined such that if both are positive the vehicle
              moves forward.

        :seealso: :class:`Bicycle` :class:`Unicycle`
        """
        super().__init__(nin=2, nout=3, inputs=inputs, **kwargs)
        self.nstates = 3
        self.type = 'diffsteer'
        self.R = R
        self.W = W
        
        if x0 is None:
            self._x0 = np.zeros((slef.nstates,))
        else:
            assert len(x0) == self.nstates, "x0 is {:d} long, should be {:d}".format(len(x0), self.nstates)
            self._x0 = x0
        
    def output(self, t):
        return self._x
    
    def deriv(self):
        theta = self._x[2]
        v = self.R * (self.inputs[0] + self.inputs[1]) / 2
        omega = (self.inputs[1] + self.inputs[0]) / self.W
    
        xd = np.r_[v * cos(theta), v * sin(theta), omega]
        return xd
    
# ------------------------------------------------------------------------ #

class VehiclePlot(GraphicsBlock):
    """
    :blockname:`VEHICLEPLOT`
    
    .. table::
       :align: left
    
    +--------+---------+---------+
    | inputs | outputs |  states |
    +--------+---------+---------+
    | 3      | 0       | 0       |
    +--------+---------+---------+
    | float  |         |         | 
    +--------+---------+---------+
    """

    nin = 1
    nout = 0

    inlabels = ('q',)

    # TODO add ability to render an image instead of an outline
    
    def __init__(self, *inputs, path=True, pathstyle=None, shape='triangle', color="blue", fill="white", size=1, scale='auto', labels=['X', 'Y'], square=True, init=None, **kwargs):
        """
        Create a vehicle animation

        :param ``*inputs``: Optional incoming connections
        :type ``*inputs``: Block or Plug
        :param path: plot path taken by vehicle, defaults to True
        :type path: bool, optional
        :param pathstyle: linestyle for path, defaults to None
        :type pathstyle: str or dict, optional
        :param shape: vehicle shape, one of: 'triangle' [default], 'box'
        :type shape: str, optional
        :param color: vehicle outline color, defaults to "blue"
        :type color: str, optional
        :param fill: vehicle fill color, defaults to "white"
        :type fill: str, optional
        :param size: length of vehicle, defaults to 1
        :type size: float, optional
        :param scale: x- and y-axis scale, defaults to 'auto'
        :type scale: array_like(2) or array_like(4), optional
        :param labels: axis labels (xlabel, ylabel), defaults to ["X","Y"]
        :type labels: array_like(2) or list
        :param square: Set aspect ratio to 1, defaults to True
        :type square: bool, optional
        :param init: initialize graphics, defaults to None
        :type init: callable, optional
        :param ``**kwargs``: common Block options
        :return: A VEHICLEPLOT block
        :rtype: VehiclePlot instance

        Create a vehicle animation similar to the figure below.

        **Block ports**

            :input qL: configuration (x, y, θ)
        
        Notes:
            
            - The ``init`` function is called after the axes are initialized
              and can be used to draw application specific detail on the
              plot. In the example below, this is the dot and star.
            - A dynamic trail, showing path to date can be animated if
              the option ``path`` is True.
            - Two shapes of vehicle can be drawn, a narrow triangle and a box
              (as seen below).
        
        .. figure:: ../../figs/rvc4_4.gif
           :width: 500px
           :alt: example of generated graphic

           Example of vehicle display (animated).  The label at the top is the
           block name.
        """
        super().__init__(nin=3, inputs=inputs, **kwargs)
        self.xdata = []
        self.ydata = []
        self.type = 'vehicleplot'
        if init is not None:
            assert callable(init), 'graphics init function must be callable'
        self.init = init
        self.square = square

        self.path = path
        if path:
            self.pathstyle = pathstyle
        self.color = color
        self.fill = fill
        
        if scale != 'auto':
            if len(scale) == 2:
                scale = scale * 2
        self.scale = scale
        self.labels = labels
        
        d = size
        if shape == 'triangle':
            L = d
            W = 0.6*d
            vertices = [(L, 0), (-L, -W), (-L, W)]
        elif shape == 'box':
            L1 = d
            L2 = d
            W = 0.6*d
            vertices = [(-L1, W), (0.6*L2, W), (L2, 0.5*W), (L2, -0.5*W), (0.6*L2, -W), (-L1, -W)]
        else:
            raise ValueError('bad vehicle shape specified')
        self.vertices_hom = base.e2h(np.array(vertices).T)
        self.vertices = np.array(vertices)

        
    def start(self, state=None, **kwargs):
        # create the plot
        super().reset()
        try:
            print('graphics start')
            self.fig = self.create_figure(state)
            print('fig created')
            self.ax = self.fig.gca()
            print('axes')
        except:
            print('aaargh')
        if self.square:
            self.ax.set_aspect('equal')
        print('done')

        args = []
        kwargs = {}
        if self.path:
            style = self.pathstyle
            if isinstance(style, dict):
                kwargs = style
            elif isinstance(style, str):
                args = [style]
            self.line, = self.ax.plot(self.xdata, self.ydata, *args, **kwargs)
        poly = Polygon(self.vertices, closed=True, edgecolor=self.color, facecolor=self.fill)
        self.vehicle = self.ax.add_patch(poly)

        self.ax.grid(True)
        self.ax.set_xlabel(self.labels[0])
        self.ax.set_ylabel(self.labels[1])
        self.ax.set_title(self.name)
        if self.scale != 'auto':
            self.ax.set_xlim(*self.scale[0:2])
            self.ax.set_ylim(*self.scale[2:4])
        if self.init is not None:
            self.init(self.ax)
            
        plt.draw()
        plt.show(block=False)

        super().start()
        
    def step(self, state=None, **kwargs):
        # inputs are set
        self.xdata.append(self.inputs[0])
        self.ydata.append(self.inputs[1])
        #plt.figure(self.fig.number)
        if self.path:
            self.line.set_data(self.xdata, self.ydata)
        T = base.transl2(self.inputs[0], self.inputs[1]) @ base.trot2(self.inputs[2])
        new = base.h2e(T @ self.vertices_hom)
        self.vehicle.set_xy(new.T)
    
        if self.scale == 'auto':
            self.ax.relim()
            self.ax.autoscale_view()
        super().step(state=state)
        
    def done(self, block=False, **kwargs):
        if self.bd.options.graphics:
            plt.show(block=block)
            
            super().done()
