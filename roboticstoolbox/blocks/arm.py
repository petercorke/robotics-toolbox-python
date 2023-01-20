import numpy as np
from math import sin, cos, pi

# import matplotlib.pyplot as plt
import time
from spatialmath import base, SE3

from bdsim.components import TransferBlock, FunctionBlock, SourceBlock
from bdsim.graphics import GraphicsBlock

from roboticstoolbox import quintic_func, trapezoidal_func

"""
Robot blocks:
- have inputs and outputs
- are a subclass of ``FunctionBlock`` |rarr| ``Block`` for kinematics and have no states
- are a subclass of ``TransferBlock`` |rarr| ``Block`` for dynamics and have states

"""
# The constructor of each class ``MyClass`` with a ``@block`` decorator becomes a method ``MYCLASS()`` of the BlockDiagram instance.

# ------------------------------------------------------------------------ #
class FKine(FunctionBlock):
    """
    :blockname:`FKINE`

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

    nin = 1
    nout = 1
    inlabels = ('q',)
    outlabels = ('T',)

    def __init__(self, robot=None, args={}, **blockargs):
        """
        :param ``*inputs``: Optional incoming connections
        :type ``*inputs``: Block or Plug
        :param robot: Robot model, defaults to None
        :type robot: Robot subclass, optional
        :param args: Options for fkine, defaults to {}
        :type args: dict, optional
        :param blockargs: |BlockOptions|
        :type blockargs: dict
        :return: a FORWARD_KINEMATICS block
        :rtype: Foward_Kinematics instance

        Robot arm forward kinematic model.

        **Block ports**

            :input q: Joint configuration vector as an ndarray.

            :output T: End-effector pose as an SE(3) object
        """
        if robot is None:
            raise ValueError('robot is not defined')

        super().__init__(**blockargs)
        self.type = "forward-kinematics"

        self.robot = robot
        self.args = args

        self.inport_names(("q",))
        self.outport_names(("T",))

    def output(self, t=None):
        q = self.inputs[0]
        return [self.robot.fkine(self.inputs[0], **self.args)]


class IKine(FunctionBlock):
    """
    :blockname:`IKINE`

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

    nin = 1
    nout = 1
    inlabels = ('T',)
    outlabels = ('q',)

    def __init__(
        self, robot=None, q0=None, useprevious=True, ik=None, seed=None, **blockargs
    ):
        """
        :param robot: Robot model, defaults to None
        :type robot: Robot subclass, optional
        :param q0: Initial joint angles, defaults to None
        :type q0: array_like(n), optional
        :param useprevious: Use previous IK solution as q0, defaults to True
        :type useprevious: bool, optional
        :param ik: Specify an IK function, defaults to 'ikine_LM'
        :type ik: callable f(T)
        :param seed: random seed for solution
        :type seed: int
        :param blockargs: |BlockOptions|
        :type blockargs: dict
        :return: an INVERSE_KINEMATICS block
        :rtype: Inverse_Kinematics instance

        Robot arm inverse kinematic model.

        The block has one input port:

            1. End-effector pose as an SE(3) object

        and one output port:

            1. Joint configuration vector as an ndarray.


        """
        if robot is None:
            raise ValueError('robot is not defined')

        super().__init__(**blockargs)
        self.type = "inverse-kinematics"

        self.robot = robot
        self.q0 = q0
        self.qprev = q0
        self.useprevious = useprevious
        self.ik = None
        self.seed = 0

        self.inport_names(("T",))
        self.outport_names(("q",))

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
            sol = self.robot.ikine_LM(self.inputs[0], q0=q0, seed=self.seed)
        else:
            sol = self.ik(self.inputs[0])

        if not sol.success:
            raise RuntimeError("inverse kinematic failure for pose", self.inputs[0])

        if self.useprevious:
            self.qprev = sol.q

        return [sol.q]


# ------------------------------------------------------------------------ #

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

    nin = 1
    nout = 1
    inlabels = ('q',)
    outlabels = ('J',)

    def __init__(
        self,
        robot,
        frame="0",
        inverse=False,
        pinv=False,
        transpose=False,
        **blockargs
    ):
        """
        :param robot: Robot model
        :type robot: Robot subclass
        :param frame: Frame to compute Jacobian for, one of: '0' [default], 'e'
        :type frame: str, optional
        :param inverse: output inverse of Jacobian, defaults to False
        :type inverse: bool, optional
        :param pinv: output pseudo-inverse of Jacobian, defaults to False
        :type pinv: bool, optional
        :param transpose: output transpose of Jacobian, defaults to False
        :type transpose: bool, optional
        :param blockargs: |BlockOptions|
        :type blockargs: dict
        :return: a JACOBIAN block
        :rtype: Jacobian instance

        Robot arm Jacobian.

        The block has one input port:

            1. Joint configuration vector as an ndarray.

        and one output port:

            1. Jacobian matrix as an ndarray(6,n)

        .. notes::
            - Only one of ``inverse`` or ``pinv`` can be True
            - ``inverse`` or ``pinv`` can be used in conjunction with ``transpose``
            - ``inverse`` requires that the Jacobian is square
            - If ``inverse`` is True and the Jacobian is singular a runtime
              error will occur.
        """
        if robot is None:
            raise ValueError('robot is not defined')
            
        super().__init__(**blockargs)

        self.robot = robot

        if frame in (0, "0"):
            self.jfunc = robot.jacob0
        elif frame == "e":
            self.jfunc = robot.jacobe
        else:
            raise ValueError("unknown frame")

        if inverse and robot.n != 6:
            raise ValueError("cannot invert a non square Jacobian")
        if inverse and pinv:
            raise ValueError("can only set one of inverse and pinv")
        self.inverse = inverse
        self.pinv = pinv
        self.transpose = transpose

        self.inport_names(("q",))
        self.outport_names(("J",))

    def output(self, t=None):
        J = self.jfunc(self.inputs[0])
        if self.inverse:
            J = np.linalg.inv(J)
        if self.pinv:
            J = np.linalg.pinv(J)
        if self.transpose:
            J = J.T
        return [J]



# ------------------------------------------------------------------------ #

class FDyn(TransferBlock):
    """
    :blockname:`FDYN`

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

    nin = 1
    nout = 3
    outlabels = ('q', 'qd', 'qdd')
    inlabels = ('τ')

    def __init__(self, robot, q0=None, **blockargs):
        """
        :param robot: Robot model
        :type robot: Robot subclass
        :param q0: Initial joint configuration
        :type q0: array_like(n)
        :param blockargs: |BlockOptions|
        :type blockargs: dict
        :return: a FORWARD_DYNAMICS block
        :rtype: Foward_Dynamics instance

        Robot arm forward dynamics model.

        The block has one input port:

            1. Joint force/torque as an ndarray.

        and three output ports:

            1. joint configuration
            2. joint velocity
            3. joint acceleration


        """
        if robot is None:
            raise ValueError('robot is not defined')
            
        super().__init__(**blockargs)
        self.type = "forward-dynamics"

        self.robot = robot
        self.nstates = robot.n * 2

        # state vector is [q qd]

        self.inport_names(("$\tau$",))
        self.outport_names(("q", "qd", "qdd"))

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
        assert len(Q) == n, "torque vector wrong size"

        q = self._x[:n]
        qd = self._x[n:]
        qdd = self.robot.accel(q, qd, Q)
        self._qdd = qdd
        return np.r_[qd, qdd]


class IDyn(FunctionBlock):
    """
    :blockname:`IDYN`

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

    nin = 3
    nout = 1
    inlabels = ('q', 'qd', 'qdd')
    outlabels = ('τ')

    def __init__(self, robot, gravity=None, **blockargs):
        """

        :param robot: Robot model
        :type robot: Robot subclass
        :param gravity: gravitational acceleration
        :type gravity: float
        :param blockargs: |BlockOptions|
        :type blockargs: dict
        :return: an INVERSE_DYNAMICS block
        :rtype: Inverse_Dynamics instance

        Robot arm forward dynamics model.

        The block has three input port:

            1. Joint configuration vector as an ndarray.
            2. Joint velocity vector as an ndarray.
            3. Joint acceleration vector as an ndarray.

        and one output port:

            1. joint torque/force

        .. TODO:: end-effector wrench input, base wrench output, payload input
        """
        if robot is None:
            raise ValueError('robot is not defined')

        super().__init__(**blockargs)
        self.type = "inverse-dynamics"

        self.robot = robot
        self.gravity = gravity

        # state vector is [q qd]

        self.inport_names(("q", "qd", "qdd"))
        self.outport_names(("$\tau$",))

    def output(self, t=None):
        tau = self.robot.rne(self.inputs[0], self.inputs[1], self.inputs[2], gravity=self.gravity)
        return [tau]


class Gravload(FunctionBlock):
    """
    :blockname:`GRAVLOAD`

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

    nin = 1
    nout = 1
    inlabels = ('q',)
    outlabels = ('τ')

    def __init__(self, robot, gravity=None, **blockargs):
        """

        :param robot: Robot model
        :type robot: Robot subclass
        :param gravity: gravitational acceleration
        :type gravity: float
        :param blockargs: |BlockOptions|
        :type blockargs: dict
        :return: a GRAVLOAD block
        :rtype: Gravload instance

        Robot arm gravity torque.

        The block has one input port:

            1. Joint configuration vector as an ndarray.

        and one output port:

            1. joint torque/force due to gravity

        """
        if robot is None:
            raise ValueError('robot is not defined')
            
        super().__init__(**blockargs)
        self.type = "gravload"

        self.robot = robot
        self.gravity = gravity
        self.inport_names(("q",))
        self.outport_names(("$\tau$",))

    def output(self, t=None):
        tau = self.robot.gravload(self.inputs[0], gravity=self.gravity)
        return [tau]

class Gravload_X(FunctionBlock):
    """
    :blockname:`GRAVLOAD_X`

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

    nin = 1
    nout = 1
    inlabels = ('q',)
    outlabels = ('w')

    def __init__(self, robot, representation="rpy/xyz", gravity=None, **blockargs):
        """

        :param robot: Robot model
        :type robot: Robot subclass
        :param gravity: gravitational acceleration
        :type gravity: float
        :param blockargs: |BlockOptions|
        :type blockargs: dict
        :return: a GRAVLOAD block
        :rtype: Gravload instance

        Robot arm gravity torque.

        The block has one input port:

            1. Joint configuration vector as an ndarray.

        and one output port:

            1. joint torque/force due to gravity

        """
        if robot is None:
            raise ValueError('robot is not defined')
            
        super().__init__(**blockargs)
        self.type = "gravload-x"

        self.robot = robot
        self.gravity = gravity
        self.inport_names(("q",))
        self.outport_names(("$\tau$",))
        self.representation = representation

    def output(self, t=None):
        q = self.inputs[0]
        w = self.robot.gravload_x(q, representation=self.representation, gravity=self.gravity)
        return [w]

class Inertia(FunctionBlock):
    """
    :blockname:`INERTIA`

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

    nin = 1
    nout = 1
    inlabels = ('q',)
    outlabels = ('M')

    def __init__(self, robot, **blockargs):
        """

        :param robot: Robot model
        :type robot: Robot subclass
        :param blockargs: |BlockOptions|
        :type blockargs: dict
        :return: an INERTIA block
        :rtype: Inertia instance

        Robot arm inertia matrix.

        The block has one input port:

            1. Joint configuration vector as an ndarray.

        and one output port:

            1. Joint-space inertia matrix :math:`\mat{M}(q)`

        """
        if robot is None:
            raise ValueError('robot is not defined')
            
        super().__init__(**blockargs)
        self.type = "inertia"

        self.robot = robot
        self.inport_names(("q",))
        self.outport_names(("M",))

    def output(self, t=None):
        M = self.robot.inertia(self.inputs[0])
        return [M]

class Inertia_X(FunctionBlock):
    """
    :blockname:`INERTIA_X`

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

    nin = 1
    nout = 1
    inlabels = ('q',)
    outlabels = ('M')

    def __init__(self, robot, representation="rpy/xyz", pinv=False, **blockargs):
        """

        :param robot: Robot model
        :type robot: Robot subclass
        :param blockargs: |BlockOptions|
        :type blockargs: dict
        :return: an INERTIA_X block
        :rtype: Inertia_X instance

        Robot arm task-space inertia matrix.

        The block has one input port:

            1. Joint configuration vector as an ndarray.

        and one output port:

            1. Task-space inertia matrix :math:`\mat{M}_x(q)`

        """
        if robot is None:
            raise ValueError('robot is not defined')
            
        super().__init__(**blockargs)
        self.type = "inertia-x"

        self.robot = robot
        self.representation = representation
        self.pinv = pinv
        self.inport_names(("q",))
        self.outport_names(("M",))

    def output(self, t=None):
        q = self.inputs[0]
        Mx = self.robot.inertia_x(q, pinv=self.pinv, representation=self.representation)
        return [Mx]
# ------------------------------------------------------------------------ #

class FDyn_X(TransferBlock):
    """
    :blockname:`FDYN_X`

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

    nin = 1
    nout = 5
    outlabels = ('q', 'qd', 'x', 'xd', 'xdd')
    inlabels = ('w')

    def __init__(self, robot, q0=None, gravcomp=False, velcomp=False, representation='rpy/xyz', **blockargs):
        """
        :param robot: Robot model
        :type robot: Robot subclass
        :param end: Link to compute pose of, defaults to end-effector
        :type end: Link or str
        :param blockargs: |BlockOptions|
        :type blockargs: dict
        :return: a FDYN_X block
        :rtype: FDyn_X instance

        Robot arm forward dynamics model.

        The block has one input port:

            1. Applied end-effector wrench as an ndarray.

        and three output ports:

            1. task space pose
            2. task space velocity
            3. task space acceleration


        """
        if robot is None:
            raise ValueError('robot is not defined')
            
        super().__init__(**blockargs)
        self.type = "forward-dynamics-x"

        self.robot = robot
        self.nstates = robot.n * 2
        self.gravcomp = gravcomp
        self.velcomp = velcomp
        self.representation = representation

        # state vector is [q qd]

        self.inport_names(("w",))
        self.outport_names(("q", "qd", "x", "xd", "xdd"))

        if q0 is None:
            q0 = np.zeros((robot.n,))
        else:
            q0 = base.getvector(q0, robot.n)
        # append qd0, assumed to be zero
        self._x0 = np.r_[q0, np.zeros((robot.n,))]
        self._qdd = None

    def output(self, t=None):
        n = self.robot.n
        q = self._x[:n]
        qd = self._x[n:]
        qdd = self._qdd  # from last deriv

        T = self.robot.fkine(q)
        x = base.tr2x(T.A)

        Ja = self.robot.jacob0_analytical(q, self.representation)
        xd = Ja @ qd
        # print(q)
        # print(qd)
        # print(xd)
        # print(Ja)
        # print()

        if qdd is None:
            xdd = None
        else:
            Ja_dot = self.robot.jacob_dot(q, qd, J0=Ja)
            xdd = Ja @ qdd + Ja_dot @ qd

        return [q, qd, x, xd, xdd]

    def deriv(self):
        # return [qd qdd]

        # get current joint space state
        n = self.robot.n
        q = self._x[:n]
        qd = self._x[n:]

        # compute joint forces
        w = self.inputs[0]
        assert len(w) == 6, "wrench vector wrong size"
        Q = self.robot.jacob0_analytical(q, representation=self.representation).T @ w

        if self.gravcomp or self.velcomp:
            if self.velcomp:
                qd_rne = qd
            else:
                qd_rne = np.zeros((n,))
            Q_rne = self.robot.rne(q, qd_rne, np.zeros((n,)))
            Q += Q_rne
        qdd = self.robot.accel(q, qd, Q)

        self._qdd = qdd
        return np.r_[qd, qdd]


# ------------------------------------------------------------------------ #

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

    nin = 1
    nout = 0
    inlabels = ('q',)

    def __init__(self, robot=None, q0=None, backend=None, **blockargs):
        """
        :param robot: Robot model
        :type robot: Robot subclass
        :param backend: RTB backend name, defaults to 'pyplot'
        :type backend: str, optional
        :param blockargs: |BlockOptions|
        :type blockargs: dict
        :return: An ARMPLOT block
        :rtype: ArmPlot instance


        Create a robot animation.

        Notes:

            - Uses RTB ``plot`` method

           Example of vehicle display (animated).  The label at the top is the
           block name.
        """
        if robot is None:
            raise ValueError('robot is not defined')

        super().__init__(**blockargs)
        self.inport_names(("q",))

        if q0 is None:
            q0 = np.zeros((robot.n,))
        self.robot = robot
        self.backend = backend
        self.q0 = q0
        self.env = None
        print('ARMPLOT init')

    def start(self, state):
        # create the plot
        # super().reset()
        # if state.options.graphics:
        print('ARMPLOT init')
        self.fig = self.create_figure(state)
        self.env = self.robot.plot(
            self.q0, backend=self.backend, fig=self.fig, block=False
        )
        super().start()

    def step(self, state):

        # update the robot plot
        self.robot.q = self.inputs[0]
        self.env.step()

        super().step(state)



if __name__ == "__main__":

    from pathlib import Path

    exec(
        open(
            Path(__file__).parent.parent.parent.absolute() / "tests" / "test_blocks.py"
        ).read()
    )
