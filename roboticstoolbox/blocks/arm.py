import numpy as np
from math import sin, cos, atan2, tan, sqrt, pi

import matplotlib.pyplot as plt
import time
from spatialmath import base, SE3

from bdsim.components import TransferBlock, FunctionBlock, SourceBlock, block
from bdsim.graphics import GraphicsBlock

from roboticstoolbox import tpoly_func, lspb_func

"""
Robot blocks:
- have inputs and outputs
- are a subclass of ``FunctionBlock`` |rarr| ``Block`` for kinematics and have no states
- are a subclass of ``TransferBlock`` |rarr| ``Block`` for dynamics and have states

"""
# The constructor of each class ``MyClass`` with a ``@block`` decorator becomes a method ``MYCLASS()`` of the BlockDiagram instance.

# ------------------------------------------------------------------------ #
@block
class Forward_Kinematics(FunctionBlock):
    """
    :blockname:`FORWARD_KINEMATICS`
    
    .. table::
       :align: left
    
       +------------+---------+---------+
       | inputs     | outputs |  states |
       +------------+---------+---------+
       | 1          | 1       | 0       |
       +------------+---------+---------+
       | ndarray    | SE3     |         | 
       +------------+---------+---------+
    """

    def __init__(self, robot=None, args={}, *inputs, **kwargs):
        """
        :param ``*inputs``: Optional incoming connections
        :type ``*inputs``: Block or Plug
        :param robot: Robot model
        :type robot: Robot subclass
        :param args: Options for fkine
        :type args: dict
        :param ``**kwargs``: common Block options
        :return: a FORWARD_KINEMATICS block
        :rtype: Foward_Kinematics instance
        
        Robot arm forward kinematic model.
        
        The block has one input port:
            
            1. Joint configuration vector as an ndarray.
            
        and one output port:
            
            1. End-effector pose as an SE(3) object


        """
        super().__init__(nin=1, nout=1, inputs=inputs, **kwargs)
        self.type = 'forward-kinematics'

        self.robot = robot
        self.args = args
            
        self.inport_names(('q',))
        self.outport_names(('T',))
        
    def output(self, t=None):
        return [self.robot.fkine(self.inputs[0], **self.args)]

@block
class Inverse_Kinematics(FunctionBlock):
    """
    :blockname:`INVERSE_KINEMATICS`
    
    .. table::
       :align: left
    
       +------------+---------+---------+
       | inputs     | outputs |  states |
       +------------+---------+---------+
       | 1          | 1       | 0       |
       +------------+---------+---------+
       | SE3        | ndarray |         | 
       +------------+---------+---------+
    """

    def __init__(self, robot=None, q0=None, useprevious=True, ik=None, *inputs, **kwargs):
        """
        :param robot: Robot model
        :type robot: Robot subclass
        :param q0: Initial joint angles
        :type q0: array_like(n)
        :param useprevious: Use previous IK solution as q0
        :type useprevious: bool
        :param ik: Specify an IK function, default is ikine_LM
        :type ik: callable f(T)
        :param ``**kwargs``: common Block options
        :param ``*inputs``: Optional incoming connections
        :type ``*inputs``: Block or Plug
        :return: an INVERSE_KINEMATICS block
        :rtype: Inverse_Kinematics instance
        
        Robot arm inverse kinematic model.
        
        The block has one input port:
            
            1. End-effector pose as an SE(3) object
            
        and one output port:
            
            1. Joint configuration vector as an ndarray.


        """
        super().__init__(nin=1, nout=1, inputs=inputs, **kwargs)
        self.type = 'inverse-kinematics'

        self.robot = robot
        self.q0 = q0
        self.qprev = q0
        self.useprevious = useprevious
        self.ik = None
            
        self.inport_names(('T',))
        self.outport_names(('q',))
    
    def start(self):
        super().start()
        if self.useprevious:
            self.qprev = self.q0

    def output(self, t=None):
        if self.useprevious:
            q0 = self.qprev
        else:
            q0 = self.q0

        if self.ik is None:
            sol = self.robot.ikine_LM(self.inputs[0], q0=q0)
        else:
            sol = self.ik(self.inputs[0])

        if not sol.success:
            raise RuntimeError('inverse kinematic failure for pose', self.inputs[0])

        if self.useprevious:
            self.qprev = sol.q

        return [sol.q]
# ------------------------------------------------------------------------ #
@block
class Jacobian(FunctionBlock):
    """
    :blockname:`JACOBIAN`
    
    .. table::
       :align: left
    
       +------------+---------+---------+
       | inputs     | outputs |  states |
       +------------+---------+---------+
       | 1          | 1       | 0       |
       +------------+---------+---------+
       | ndarray    | ndarray |         | 
       +------------+---------+---------+
    """

    def __init__(self, robot, *inputs, frame='0', inverse=False, transpose=False, **kwargs):
        """
        :param ``*inputs``: Optional incoming connections
        :type ``*inputs``: Block or Plug
        :param robot: Robot model
        :type robot: Robot subclass
        :param frame: Frame to compute Jacobian for: '0' (default) or 'e'
        :type frame: str
        :param ``**kwargs``: common Block options
        :return: a JACOBIAN block
        :rtype: Jacobian instance
        
        Robot arm Jacobian.
        
        The block has one input port:
            
            1. Joint configuration vector as an ndarray.
            
        and one output port:
            
            1. Jacobian matrix as an ndarray(6,n)


        """
        super().__init__(nin=1, nout=1, inputs=inputs, **kwargs)
        self.type = 'jacobian'

        self.robot = robot

        if frame == '0':
            self.jfunc = robot.jacob0
        elif frame == 'e':
            self.jfunc = robot.jacobe
        else:
            raise ValueError('unknown frame')
        self.inverse = inverse
        self.transpose = transpose
            
        self.inport_names(('q',))
        self.outport_names(('J',))
        
    def output(self, t=None):
        J = self.jfunc(self.inputs[0])
        if self.inverse:
            J = np.linalg.inv(J)
        if self.transpose:
            J = J.T
        return [J]

@block
class Tr2Delta(FunctionBlock):
    """
    :blockname:`TR2DELTA`
    
    .. table::
       :align: left
    
       +------------+------------+---------+
       | inputs     | outputs    |  states |
       +------------+------------+---------+
       | 2          | 1          | 0       |
       +------------+------------+---------+
       | SE3, SE3   | ndarray(6) |         | 
       +------------+------------+---------+
    """

    def __init__(self, *inputs, **kwargs):
        """
        :param ``*inputs``: Optional incoming connections
        :type ``*inputs``: Block or Plug
        :param ``**kwargs``: common Block options
        :return: a TR2DELTA block
        :rtype: Tr2Delta instance
        
        Difference between T1 and T2 as a 6-vector
        
        The block has two input port:
            
            1. T1 as an SE3.
            2. T2 as an SE3.
            
        and one output port:
            
            1. delta as an ndarray(6,n)

        :seealso: :func:`spatialmath.base.tr2delta`
        """
        super().__init__(nin=2, nout=1, inputs=inputs, **kwargs)
        self.type = 'tr2delta'
           
        self.inport_names(('T1', 'T2'))
        self.outport_names(('$\delta$',))
        
    def output(self, t=None):
        return [base.tr2delta(self.inputs[0].A, self.inputs[1].A)]

# ------------------------------------------------------------------------ #

@block
class Delta2Tr(FunctionBlock):
    """
    :blockname:`DELTA2TR`
    
    .. table::
       :align: left
    
       +------------+----------+---------+
       | inputs     | outputs  |  states |
       +------------+----------+---------+
       | 1          | 1        | 0       |
       +------------+----------+---------+
       | ndarray(6) | SE3      |         | 
       +------------+----------+---------+
    """

    def __init__(self, *inputs, **kwargs):
        """
        :param ``*inputs``: Optional incoming connections
        :type ``*inputs``: Block or Plug
        :param ``**kwargs``: common Block options
        :return: a DELTA2TR block
        :rtype: Delta2Tr instance
        
        Delta to SE(3)
        
        The block has one input port:
            
            1. delta as an ndarray(6,n)
            
        and one output port:
            
            1. T as an SE3

        :seealso: :func:`spatialmath.base.delta2tr`
        """
        super().__init__(nin=1, nout=1, inputs=inputs, **kwargs)
        self.type = 'delta2tr'
           
        self.inport_names(('$\delta$',))
        self.outport_names(('T',))
        
    def output(self, t=None):
        return [SE3.Delta(self.inputs[0])]

# ------------------------------------------------------------------------ #

@block
class Point2Tr(FunctionBlock):
    """
    :blockname:`POINT2TR`
    
    .. table::
       :align: left
    
       +------------+----------+---------+
       | inputs     | outputs  |  states |
       +------------+----------+---------+
       | 1          | 1        | 0       |
       +------------+----------+---------+
       | ndarray(3) | SE3      |         | 
       +------------+----------+---------+
    """

    def __init__(self, T, *inputs, **kwargs):
        """
        :param T: the transform
        :type T: SE3
        :param ``*inputs``: Optional incoming connections
        :type ``*inputs``: Block or Plug
        :param ``**kwargs``: common Block options
        :return: a POINT2TR block
        :rtype: Point2Tr instance
        
        The block has one input port:
            
            1. a 3D point as an ndarray(3)
            
        and one output port:
            
            1. T as an SE3 with its position part replaced by the input

        :seealso: :func:`spatialmath.base.delta2tr`
        """
        super().__init__(nin=1, nout=1, inputs=inputs, **kwargs)
        self.type = 'point2tr'
           
        self.inport_names(('t',))
        self.outport_names(('T',))
        self.pose = T
        
    def output(self, t=None):
        T = SE3.SO3(self.pose.R, t=self.inputs[0])
        return [T]

# ------------------------------------------------------------------------ #
@block
class Foward_Dynamics(TransferBlock):
    """
    :blockname:`FORWARD_DYNAMICS`
    
    .. table::
       :align: left
    
       +------------+---------+---------+
       | inputs     | outputs |  states |
       +------------+---------+---------+
       | 1          | 3       | 0       |
       +------------+---------+---------+
       | ndarray    | ndarray,|         | 
       |            | ndarray,|         |
       |            | ndarray |         |
       +------------+---------+---------+
    """

    def __init__(self, robot, *inputs, q0=None, **kwargs):
        """
        :param ``*inputs``: Optional incoming connections
        :type ``*inputs``: Block or Plug
        :param robot: Robot model
        :type robot: Robot subclass
        :param end: Link to compute pose of, defaults to end-effector
        :type end: Link or str
        :param ``**kwargs``: common Block options
        :return: a FORWARD_DYNAMICS block
        :rtype: Foward_Dynamics instance
        
        Robot arm forward dynamics model.
        
        The block has one input port:
            
            1. Joint configuration vector as an ndarray.
            
        and three output ports:
            
            1. joint configuration
            2. joint velocity
            3. joint acceleration


        """
        super().__init__(nin=1, nout=3, inputs=inputs, **kwargs)
        self.type = 'forward-dynamics'

        self.robot = robot
        self.nstates = robot.n * 2

        # state vector is [q qd]

        self.inport_names(('$\tau$',))
        self.outport_names(('q', 'qd', 'qdd'))

        if q0 is None:
            q0 = np.zeros((robot.n,))
        else:
            q0 = base.getvector(q0, robot.n)
        self._x0 = np.r_[q0, np.zeros((robot.n,))]
        self._qdd = None
        
    def output(self, t=None):
        n = self.robot.n
        q = self._x[:n]
        qd = self._x[n:]
        qdd = self._qdd  # from last deriv
        return [q, qd, qdd]

    def deriv(self):
        # return [qd qdd]
        Q = self.inputs[0]
        n = self.robot.n
        assert len(Q) == n, 'torque vector wrong size'

        q = self._x[:n]
        qd = self._x[n:]
        qdd = self.robot.accel(q, qd, Q)
        self._qdd = qdd
        return np.r_[qd, qdd]

@block
class Inverse_Dynamics(FunctionBlock):
    """
    :blockname:`INVERSE_DYNAMICS`
    
    .. table::
       :align: left
    
       +------------+---------+---------+
       | inputs     | outputs |  states |
       +------------+---------+---------+
       | 3          | 1       | 0       |
       +------------+---------+---------+
       | ndarray,   | ndarray |         | 
       | ndarray,   |         |         |
       | ndarray    |         |         |  
       +------------+---------+---------+
    """

    def __init__(self, robot, *inputs, gravity=None, **kwargs):
        """

        :param robot: Robot model
        :type robot: Robot subclass
        :param gravity: Link to compute pose of, defaults to end-effector
        :type gravity: Link or str
        :param ``*inputs``: Optional incoming connections
        :type ``*inputs``: Block or Plug
        :param ``**kwargs``: common Block options
        :return: an INVERSE_DYNAMICS block
        :rtype: Inverse_Dynamics instance
        
        Robot arm forward dynamics model.
        
        The block has three input port:
            
            1. Joint configuration vector as an ndarray.
            2. Joint velocity vector as an ndarray.
            3. Joint acceleration vector as an ndarray.
            
        and one output ports:
            
            1. joint torque/force

        .. TODO:: end-effector wrench input, base wrench output, payload input
        """
        super().__init__(nin=3, nout=1, inputs=inputs, **kwargs)
        self.type = 'inverse-dynamics'

        self.robot = robot

        # state vector is [q qd]

        self.inport_names(('q', 'qd', 'qdd'))
        self.outport_names(('$\tau$',))
        
    def output(self, t=None):
        tau = self.robot.rne(self.inputs[0], self.inputs[1], self.inputs[2])
        return [tau]

        
# ------------------------------------------------------------------------ #

@block
class ArmPlot(GraphicsBlock):
    """
    :blockname:`ARMPLOT`
    
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
        
    def __init__(self, robot=None, *inputs, q0=None, backend=None, **kwargs):
        """
        :param ``*inputs``: Optional incoming connections
        :type ``*inputs``: Block or Plug
        :param robot: Robot model
        :type robot: Robot subclass
        :param backend: RTB backend name, default is pyplot
        :type backend: str
        :param ``**kwargs``: common Block options
        :return: An ARMPLOT block
        :rtype: ArmPlot instance


        Create a robot animation.
        
        Notes:
            
            - Uses RTB ``plot`` method

           Example of vehicle display (animated).  The label at the top is the
           block name.
        """
        super().__init__(nin=1, inputs=inputs, **kwargs)
        self.type = 'armplot'

        if q0 is None:
            q0 = np.zeros((robot.n,))
        self.robot = robot
        self.backend = backend
        self.q0 = q0
        self.env = None
        
    def start(self, **kwargs):
        # create the plot
        super().reset()
        if self.bd.options.graphics:
            self.fig = self.create_figure()
            self.env = self.robot.plot(self.q0, backend=self.backend, fig=self.fig, block=False)
            super().start()
        
    def step(self):
        # inputs are set
        if self.bd.options.graphics:
            
            self.robot.q = self.inputs[0]
            self.env.step()

            super().step()
        
    def done(self, block=False, **kwargs):
        if self.bd.options.graphics:
            plt.show(block=block)
            
            super().done()


@block
class Traj(FunctionBlock):
    """
    :blockname:`TRAJ`
    
    .. table::
       :align: left
    
       +------------+---------+---------+
       | inputs     | outputs |  states |
       +------------+---------+---------+
       | 0 or 1     | 1       | 0       |
       +------------+---------+---------+
       | float      | float   |         | 
       +------------+---------+---------+
    """

    def __init__(self, y0=0, yf=1, T=None, time=False, traj='lspb', *inputs, **kwargs):
        """

        :param y0: initial value
        :type y0: array_like(m)
        :param yf: final value
        :type yf: array_like(m)
        :param time: x is simulation time, defaults to False
        :type time: bool
        :param traj: trajectory type: 'lspb' [default] or 'tpoly'
        :type traj: str
        :param ``*inputs``: Optional incoming connections
        :type ``*inputs``: Block or Plug
        :param ``**kwargs``: common Block options
        :return: TRAJ block
        :rtype: Traj instance
        
        Create a trajectory block.
    
        A block that generates a trajectory using a trapezoidal or quintic
        polynomial profile.
        
        A simple triangle function with domain [0,10] and range [0,1] can be
        defined by::
            
            INTERPOLATE(x=(0,5,10), y=(0,1,0))
        
        We might also express this as::
            
            INTERPOLATE(xy=[(0,0), (5,1), (10,0)])
        
        The distance along the trajectory comes from:
            
        - Input port 0
        - Simulation time, if ``time=True``.  In this case the block has no
          input ports and is a ``Source`` not a ``Function``.
        """
        self.time = time
        if time:
            nin = 0
            self.blockclass = 'source'
        else:
            nin = 1
        super().__init__(nin=nin, nout=3, inputs=inputs, **kwargs)
        self.type = 'function'

        y0 = base.getvector(y0)
        yf = base.getvector(yf)
        assert len(y0) == len(yf), 'y0 and yf must have same length'

        self.y0 = y0
        self.yf = yf
        self.time = time
        self.T = T
        self.traj = traj

        self.outport_names(('y', 'yd', 'ydd'))
                
    def start(self, **kwargs):
        if self.time:
            assert self.x[0] <= 0, 'interpolation not defined for t=0'
            assert self.x[-1] >= self.bd.T, 'interpolation not defined for t=T'

        if self.traj == 'lspb':
            trajfunc = lspb_func
        elif self.traj == 'tpoly':
            trajfunc = tpoly_func

        self.trajfuncs = []
        for i in range(len(self.y0)):
            self.trajfuncs.append(trajfunc(self.y0[i], self.yf[i], self.T))
        
    def output(self, t=None):
        if not self.time:
            t = self.inputs[0]

        out = []
        for i in range(len(self.y0)):
            out.append(self.trajfuncs[i](t))

        # we have a list of tuples out[i][j]
        # i is the timestep, j is y/yd/ydd
        y = [o[0] for o in out]
        yd = [o[1] for o in out]
        ydd = [o[2] for o in out]

        return [np.hstack(y), np.hstack(yd), np.hstack(ydd)]

@block
class CirclePath(SourceBlock):
    """
    :blockname:`CIRCLEPATH`
    
    .. table::
       :align: left
    
       +------------+---------+---------+
       | inputs     | outputs |  states |
       +------------+---------+---------+
       | 0 or 1     | 1       | 0       |
       +------------+---------+---------+
       | float      | float   |         | 
       +------------+---------+---------+
    """

    def __init__(self, radius=1, centre=(0,0, 0), pose=None, phase=0, frequency=1, unit='rps', *inputs, **kwargs):
        """

        :param y0: initial value
        :type y0: array_like(m)
        :param yf: final value
        :type yf: array_like(m)
        :param time: x is simulation time, defaults to False
        :type time: bool
        :param traj: trajectory type: 'lspb' [default] or 'tpoly'
        :type traj: str
        :param ``*inputs``: Optional incoming connections
        :type ``*inputs``: Block or Plug
        :param ``**kwargs``: common Block options
        :return: TRAJ block
        :rtype: Traj instance
        
        Create a trajectory block.
    
        A block that generates a trajectory using a trapezoidal or quintic
        polynomial profile.
        
        A simple triangle function with domain [0,10] and range [0,1] can be
        defined by::
            
            INTERPOLATE(x=(0,5,10), y=(0,1,0))
        
        We might also express this as::
            
            INTERPOLATE(xy=[(0,0), (5,1), (10,0)])
        
        The distance along the trajectory comes from:
            
        - Input port 0
        - Simulation time, if ``time=True``.  In this case the block has no
          input ports and is a ``Source`` not a ``Function``.
        """

        super().__init__(nin=0, nout=1, **kwargs)
        self.blockclass = 'circlepath'

        if unit == 'rps':
            omega = frequency * 2 * pi
            phase = frequency * 2 * pi
        elif unit == 'rad':
            omega = frequency
            phase = phase
        else:
            raise ValueError('bad units: rps or rad')

        self.radius = radius
        assert len(centre) == 3, 'centre must be a 3 vector'
        self.centre = centre
        self.pose = pose
        self.omega = omega
        self.phase = phase

        self.outport_names(('y',))
        
    def output(self, t=None):
        theta = t * self.omega + self.phase
        x = self.radius * cos(theta) + self.centre[0]
        y = self.radius * sin(theta) + self.centre[1]
        p = [x, y, self.centre[2]]

        if self.pose is not None:
            p = self.pose * [x, y, 0]
            p = p.flatten()
        
        return [p]

if __name__ == "__main__":

    import pathlib
    import os.path

    exec(open(os.path.join(pathlib.Path(__file__).parent.absolute(), "test_robots.py")).read())
