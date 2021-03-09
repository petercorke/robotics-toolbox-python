import numpy as np
from math import sin, cos, atan2, tan, sqrt, pi

import matplotlib.pyplot as plt
import time

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

    def __init__(self, *inputs, robot=None, end=None, **kwargs):
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

    def __init__(self, *inputs, robot=None, frame='0', inverse=False, transpose=False, **kwargs):
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
        
    def __init__(self, *inputs, robot=None, q0=None, backend=None, **kwargs):
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
        if self.bd.options['graphics']:
            plt.show(block=block)
            
            super().done()

if __name__ == "__main__":

    import pathlib
    import os.path

    exec(open(os.path.join(pathlib.Path(__file__).parent.absolute(), "test_robots.py")).read())
