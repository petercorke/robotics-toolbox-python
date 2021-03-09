import numpy as np
from math import sin, cos, atan2, tan, sqrt, pi

import matplotlib.pyplot as plt
import time
from spatialmath import base

from bdsim.components import TransferBlock, FunctionBlock, block
from bdsim.graphics import GraphicsBlock

"""
Robot blocks:
- have inputs and outputs
- are a subclass of ``FunctionBlock`` |rarr| ``Block`` for kinematics and have no states
- are a subclass of ``TransferBlock`` |rarr| ``Block`` for dynamics and have states

"""
# The constructor of each class ``MyClass`` with a ``@block`` decorator becomes a method ``MYCLASS()`` of the BlockDiagram instance.

# ------------------------------------------------------------------------ #
@block
class FowardKinematics(FunctionBlock):
    """
    :blockname:`FORWARD_KINEMATICS`
    
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

    def __init__(self, robot=None, end=None, *inputs, **kwargs):
        """
        :param ``*inputs``: Optional incoming connections
        :type ``*inputs``: Block or Plug
        :param robot: Robot model
        :type robot: Robot subclass
        :param end: Link to compute pose of, defaults to end-effector
        :type end: Link or str
        :param ``**kwargs``: common Block options
        :return: a FORWARD_KINEMATICS block
        :rtype: FowardKinematics instance
        
        Robot arm forward kinematic model.
        
        The block has one input port:
            
            1. Joint configuration vector as an ndarray.
            
        and one output port:
            
            1. end-effector pose as an SE(3) object


        """
        super().__init__(nin=1, nout=1, inputs=inputs, **kwargs)
        self.type = 'forward-kinematics'

        self.robot = robot
        self.end = end
            
        self.inport_names(('q',))
        self.outport_names(('T',))
        
    def output(self, t=None):
        return [self.robot.fkine(self.inputs[0], end=self.end)]

# ------------------------------------------------------------------------ #
@block
class Jacobian(FunctionBlock):
    """
    :blockname:`Jacobian`
    
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
        :return: a FORWARD_KINEMATICS block
        :rtype: FowardKinematics instance
        
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

# ------------------------------------------------------------------------ #
@block
class FowardDynamics(TransferBlock):
    """
    :blockname:`FORWARD_DYNAMICS`
    
    .. table::
       :align: left
    
       +------------+---------+---------+
       | inputs     | outputs |  states |
       +------------+---------+---------+
       | 1          | 3       | 0       |
       +------------+---------+---------+
       | ndarray    | ndarray |         | 
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
        :return: a FORWARD_KINEMATICS block
        :rtype: FowardDynamics instance
        
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

if __name__ == "__main__":

    import pathlib
    import os.path

    exec(open(os.path.join(pathlib.Path(__file__).parent.absolute(), "test_robots.py")).read())
