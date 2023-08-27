import numpy as np
from math import sin, cos, pi

# import matplotlib.pyplot as plt
import time
from spatialmath import SE3
import spatialmath.base as smb

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
    r"""
    :blockname:`FKINE`

    Robot arm forward kinematics.

    :inputs: 1
    :outputs: 1
    :states: 0

    .. list-table::
        :header-rows: 1

        *   - Port type
            - Port number
            - Types
            - Description
        *   - Input
            - 0
            - ndarray(N)
            - :math:`\mathit{q}`
        *   - Output
            - 0
            - SE3
            - :math:`\mathbf{T}`

    Compute the end-effector pose as an SE(3) object as a function of the input joint
    configuration.

    :seealso: :meth:`~roboticstoolbox.robot.Robot.Robot.fkine`
    """

    nin = 1
    nout = 1
    inlabels = ("q",)
    outlabels = ("T",)

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
        """
        if robot is None:
            raise ValueError("robot is not defined")

        super().__init__(**blockargs)
        # self.type = "forward-kinematics"

        self.robot = robot
        self.args = args

        self.inport_names(("q",))
        self.outport_names(("T",))

    def output(self, t, inports, x):
        q = inports[0]
        return [self.robot.fkine(q, **self.args)]


# ------------------------------------------------------------------------ #


class IKine(FunctionBlock):
    r"""
    :blockname:`IKINE`

    Robot arm inverse kinematics.

    :inputs: 1
    :outputs: 1
    :states: 0

    .. list-table::
        :header-rows: 1

        *   - Port type
            - Port number
            - Types
            - Description
        *   - Input
            - 0
            - SE3
            - :math:`\mathbf{T}`
        *   - Output
            - 0
            - ndarray(N)
            - :math:`\mathit{q}`

    Compute joint configuration required to achieve end-effector pose input as
    an SE(3) object.

    :note: The solution may not exist and is not unique.  The solution will depend
        on the initial joint configuration ``q0``.

    :seealso: :meth:`~roboticstoolbox.robot.Robot.Robot.ik_LM`
    """

    nin = 1
    nout = 1
    inlabels = ("T",)
    outlabels = ("q",)

    def __init__(
        self,
        robot=None,
        q0=None,
        useprevious=True,
        ik=None,
        args={},
        seed=None,
        **blockargs,
    ):
        """
        :param robot: Robot model, defaults to None
        :type robot: Robot subclass, optional
        :param q0: Initial joint angles, defaults to None
        :type q0: array_like(n), optional
        :param useprevious: Use previous IK solution as q0, defaults to True
        :type useprevious: bool, optional
        :param ik: Specify an IK function, defaults to ``Robot.ikine_LM``
        :type ik: callable
        :param args: Options passed to IK function
        :type args: dict
        :param seed: random seed for solution
        :type seed: int
        :param blockargs: |BlockOptions|
        :type blockargs: dict
        """
        if robot is None:
            raise ValueError("robot is not defined")

        super().__init__(**blockargs)
        # self.type = "inverse-kinematics"

        self.robot = robot
        self.q0 = q0
        self.qprev = q0
        self.useprevious = useprevious
        if ik is None:
            ik = robot.ikine_LM
        self.ik = ik
        self.args = args
        self.seed = 0

        self.inport_names(("T",))
        self.outport_names(("q",))

    def start(self):
        super().start()
        if self.useprevious:
            self.qprev = self.q0

    def output(self, t, inports, x):
        if self.useprevious:
            q0 = self.qprev
        else:
            q0 = self.q0

        sol = self.ik(inports[0], q0=q0, seed=self.seed, **self.args)

        if not sol.success:
            raise RuntimeError("inverse kinematic failure for pose", inports[0])

        if self.useprevious:
            self.qprev = sol.q

        return [sol.q]


# ------------------------------------------------------------------------ #


class Jacobian(FunctionBlock):
    r"""
    :blockname:`JACOBIAN`

    Robot arm Jacobian matrix.

    :inputs: 1
    :outputs: 1
    :states: 0

    .. list-table::
        :header-rows: 1

        *   - Port type
            - Port number
            - Types
            - Description
        *   - Input
            - 0
            - ndarray(N)
            - :math:`\mathit{q}`
        *   - Output
            - 0
            - ndarray(N,N)
            - :math:`\mathbf{J}`

    Compute the Jacobian matrix as a function of the input joint configuration.  The
    Jacobian can be computed in the world or end-effector frame, for spatial or
    analytical velocity, and its inverse, damped inverse or transpose can be returned.

    :seealso: :meth:`~roboticstoolbox.robot.Robot.Robot.jacob0`
        :meth:`~roboticstoolbox.robot.Robot.Robot.jacobe`
        :meth:`~roboticstoolbox.robot.Robot.Robot.jacob0_analytical`
    """

    nin = 1
    nout = 1
    inlabels = ("q",)
    outlabels = ("J",)

    def __init__(
        self,
        robot,
        frame="0",
        representation=None,
        inverse=False,
        pinv=False,
        damping=None,
        transpose=False,
        **blockargs,
    ):
        """
        :param robot: Robot model
        :type robot: Robot subclass
        :param frame: Frame to compute Jacobian for, one of: "0" [default], "e"
        :type frame: str, optional
        :param representation: representation for analytical Jacobian
        :type representation: str, optional
        :param inverse: output inverse of Jacobian, defaults to False
        :type inverse: bool, optional
        :param pinv: output pseudo-inverse of Jacobian, defaults to False
        :type pinv: bool, optional
        :param damping: damping term for inverse, defaults to None
        :type damping: float or array_like(N)
        :param transpose: output transpose of Jacobian, defaults to False
        :type transpose: bool, optional
        :param blockargs: |BlockOptions|
        :type blockargs: dict

        If an inverse is requested and ``damping`` is not None it is added to the
        diagonal of the Jacobian prior to the inversion.  If a scalar is provided it is
        added to each element of the diagonal, otherwise an N-vector is assumed.

        .. note::
            - Only one of ``inverse`` or ``pinv`` can be True
            - ``inverse`` or ``pinv`` can be used in conjunction with ``transpose``
            - ``inverse`` requires that the Jacobian is square
            - If ``inverse`` is True and the Jacobian is singular a runtime
              error will occur.
        """
        if robot is None:
            raise ValueError("robot is not defined")

        super().__init__(**blockargs)

        self.robot = robot

        if frame in (0, "0"):
            if representation is None:
                self.jfunc = robot.jacob0
            else:
                self.jfunc = lambda q: robot.jacob0_analytical(
                    q, representation=representation
                )
        elif frame == "e":
            if representation is None:
                self.jfunc = robot.jacobe
            else:
                raise ValueError("cannot compute analytical Jacobian in EE frame")
        else:
            raise ValueError("unknown frame")

        if inverse and robot.n != 6:
            raise ValueError("cannot invert a non square Jacobian")
        if inverse and pinv:
            raise ValueError("can only set one of inverse and pinv")
        self.inverse = inverse
        self.pinv = pinv
        self.damping = damping
        self.transpose = transpose
        self.representation = representation

        self.inport_names(("q",))
        self.outport_names(("J",))

    def output(self, t, inports, x):
        q = inports[0]

        J = self.jfunc(q)

        # add damping term if given
        if (self.inverse or self.pinv) and self.damping is not None:
            D = np.zeros(J.shape)
            np.fill_diagonal(D, self.damping)
            J = J + D

        # optionally invert the Jacobian
        if self.inverse:
            J = np.linalg.inv(J)
        if self.pinv:
            J = np.linalg.pinv(J)

        # optionally transpose the Jacobian
        if self.transpose:
            J = J.T
        return [J]


# ------------------------------------------------------------------------ #


class ArmPlot(GraphicsBlock):
    r"""
    :blockname:`ARMPLOT`

    Plot robot arm.

    :inputs: 1 [ndarray(N)]
    :outputs: 0
    :states: 0

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
            - ndarray(N)
            - :math:`\mathit{q}`, joint configuration
        *   - Output

    Create a robot animation using the robot's ``plot`` method.

    :seealso: :meth:`~roboticstoolbox.robot.Robot.Robot.plot`
    """

    nin = 1
    nout = 0
    inlabels = ("q",)

    def __init__(self, robot=None, q0=None, backend=None, **blockargs):
        """
        :param robot: Robot model
        :type robot: Robot subclass
        :param q0: initial joint angles, defaults to None
        :type q0: ndarray(N)
        :param backend: RTB backend name, defaults to 'pyplot'
        :type backend: str, optional
        :param blockargs: |BlockOptions|
        :type blockargs: dict
        """
        if robot is None:
            raise ValueError("robot is not defined")

        super().__init__(**blockargs)
        self.inport_names(("q",))

        if q0 is None:
            q0 = np.zeros((robot.n,))
        self.robot = robot
        self.backend = backend
        self.q0 = q0
        self.env = None

    def start(self, simstate):
        # create the plot
        # super().reset()
        # if state.options.graphics:
        super().start(simstate)
        self.fig = self.create_figure(simstate)
        self.env = self.robot.plot(
            self.q0, backend=self.backend, fig=self.fig, block=False
        )

    def step(self, t, inports):
        # update the robot plot
        self.robot.q = inports[0]
        self.env.step()

        super().step(t, inports)


# ======================================================================== #


class JTraj(SourceBlock):
    """
    :blockname:`JTRAJ`

    Joint-space trajectory

    :inputs: 0
    :outputs: 3
    :states: 0

    .. list-table::
        :header-rows: 1

        *   - Port type
            - Port number
            - Types
            - Description
        *   - Output
            - 0
            - ndarray
            - :math:`q(s)`
        *   - Output
            - 1
            - ndarray
            - :math:`\dot{q}(s)`
        *   - Output
            - 2
            - ndarray
            - :math:`\ddot{q}(s)`

    Outputs a joint space trajectory where the joint coordinates vary from ``q0`` to
    ``qf`` over the course of the simulation. A quintic (5th order) polynomial is used
    with default zero boundary conditions for velocity and acceleration.

    :seealso: :func:`~roboticstoolbox.tools.trajectory.ctraj`
        :func:`~roboticstoolbox.tools.trajectory.xplot`
        :func:`~roboticstoolbox.tools.trajectory.jtraj`
    """

    nin = 0
    nout = 3
    outlabels = ("q", "qd", "qdd")

    def __init__(self, q0, qf, qd0=None, qdf=None, T=None, **blockargs):
        """

        :param q0: initial joint coordinate
        :type q0: array_like(n)
        :param qf: final joint coordinate
        :type qf: array_like(n)
        :param T: time vector or number of steps, defaults to None
        :type T: array_like or int, optional
        :param qd0: initial velocity, defaults to None
        :type qd0: array_like(n), optional
        :param qdf: final velocity, defaults to None
        :type qdf: array_like(n), optional
        :param blockargs: |BlockOptions|
        :type blockargs: dict
        """
        super().__init__(**blockargs)
        self.outport_names(
            (
                "q",
                "qd",
                "qdd",
            )
        )

        q0 = smb.getvector(q0)
        qf = smb.getvector(qf)

        if not len(q0) == len(qf):
            raise ValueError("q0 and q1 must be same size")

        if qd0 is None:
            qd0 = np.zeros(q0.shape)
        else:
            qd0 = getvector(qd0)
            if not len(qd0) == len(q0):
                raise ValueError("qd0 has wrong size")
        if qdf is None:
            qdf = np.zeros(q0.shape)
        else:
            qd1 = getvector(qdf)
            if not len(qd1) == len(q0):
                raise ValueError("qd1 has wrong size")

        self.q0 = q0
        self.qf = qf
        self.qd0 = qd0
        self.qdf = qf

        # call start now, so that output works when called by compile
        # set T to 1 just for now
        if T is None:
            self.T = 1
        self.T = T

        # valid value, to allow compile to call output() before start()
        self.start(None)

    def start(self, simstate):
        if self.T is None:
            # use simulation tmax
            self.T = simstate.T

        tscal = self.T
        self.tscal = tscal

        q0 = self.q0
        qf = self.qf
        qd0 = self.qd0
        qdf = self.qdf

        # compute the polynomial coefficients
        A = 6 * (qf - q0) - 3 * (qdf + qd0) * tscal
        B = -15 * (qf - q0) + (8 * qd0 + 7 * qdf) * tscal
        C = 10 * (qf - q0) - (6 * qd0 + 4 * qdf) * tscal
        E = qd0 * tscal
        F = q0

        self.coeffs = np.array([A, B, C, np.zeros(A.shape), E, F])
        self.dcoeffs = np.array(
            [np.zeros(A.shape), 5 * A, 4 * B, 3 * C, np.zeros(A.shape), E]
        )
        self.ddcoeffs = np.array(
            [
                np.zeros(A.shape),
                np.zeros(A.shape),
                20 * A,
                12 * B,
                6 * C,
                np.zeros(A.shape),
            ]
        )

    def output(self, t, inports, x):
        tscal = self.tscal
        ts = t / tscal
        tt = np.array([ts**5, ts**4, ts**3, ts**2, ts, 1]).T

        qt = tt @ self.coeffs

        # compute  velocity
        qdt = tt @ self.dcoeffs / tscal

        # compute  acceleration
        qddt = tt @ self.ddcoeffs / tscal**2

        return [qt, qdt, qddt]


# ------------------------------------------------------------------------ #


class CTraj(SourceBlock):
    r"""
    :blockname:`CTRAJ`

    Task space trajectory

    :inputs: 0
    :outputs: 1
    :states: 0

    .. list-table::
        :header-rows: 1

        *   - Port type
            - Port number
            - Types
            - Description
        *   - Output
            - 0
            - SE3
            - :math:`\mathbf{T}(t)`

    The block outputs a pose that varies smoothly from ``T1`` to ``T2`` over
    the course of ``T`` seconds.

    If ``T`` is not given it defaults to the simulation time.

    If ``trapezoidal`` is True then a trapezoidal motion profile is used along the path
    to provide initial acceleration and final deceleration.  Otherwise,
    motion is at constant velocity.

    :seealso: :meth:`~spatialmath.pose3d.SE3.interp`
        :func:`~roboticstoolbox.tools.trajectory.ctraj`
        :func:`~roboticstoolbox.tools.trajectory.xplot`
        :func:`~roboticstoolbox.tools.trajectory.jtraj`
    """

    nin = 0
    nout = 1
    outlabels = ("T",)

    def __init__(self, T1, T2, T, trapezoidal=True, **blockargs):
        """
        :param T1: initial pose
        :type T1: SE3
        :param T2: final pose
        :type T2: SE3
        :param T: motion time
        :type T: float
        :param trapezoidal: Use LSPB motion profile along the path
        :type trapezoidal: bool
        :param blockargs: |BlockOptions|
        :type blockargs: dict
        """

        # TODO
        # flag to rotate the frame rather than just translate it
        super().__init__(**blockargs)

        self.T1 = T1
        self.T2 = T2
        self.T = T
        self.trapezoidal = trapezoidal

    def start(self, simstate):
        if self.T is None:
            self.T = simstate.T
        if self.trapezoidal:
            self.trapezoidalfunc = trapezoidal_func(0.0, 1.0, self.T)

    def output(self, t, inports, x):
        if self.trapezoidal:
            s = self.trapezoidalfunc(t)
        else:
            s = np.min(t / self.T, 1.0)

        return [self.T1.interp(self.T2, s)]


# ------------------------------------------------------------------------ #


class CirclePath(SourceBlock):
    """
    :blockname:`CIRCLEPATH`

    Circular motion.

    :inputs: 0 or 1
    :outputs: 1
    :states: 0

    .. list-table::
        :header-rows: 1

        *   - Port type
            - Port number
            - Types
            - Description
        *   - Output
            - 0
            - ndarray(3) or SE3
            - :math:`\mathit{p}(t)` or :math:`\mathbf{T}(t)`

    The block outputs the coordinates of a point moving in a circle of
    radius ``r`` centred at ``centre`` and parallel to the xy-plane.

    By default the output is a 3-vector :math:`(x, y, z)` but if
    ``pose`` is an ``SE3`` instance the output is a copy of that pose with
    its translation set to the coordinate of the moving point.  This is the
    motion of a frame with fixed orientation following a circular path.
    """

    nin = 0
    nout = 1

    def __init__(
        self,
        radius=1,
        centre=(0, 0, 0),
        pose=None,
        frequency=1,
        unit="rps",
        phase=0,
        **blockargs,
    ):
        """
        :param radius: radius of circle, defaults to 1
        :type radius: float
        :param centre: center of circle, defaults to [0,0,0]
        :type centre: array_like(3)
        :param pose: SE3 pose of output, defaults to None
        :type pose: SE3
        :param frequency: rotational frequency, defaults to 1
        :type frequency: float
        :param unit: unit for frequency, one of: 'rps' [default], 'rad'
        :type unit: str
        :param phase: phase
        :type phase: float
        :param blockargs: |BlockOptions|
        :type blockargs: dict
        """

        # TODO
        # flag to rotate the frame rather than just translate it
        super().__init__(**blockargs)

        if unit == "rps":
            omega = frequency * 2 * pi
            phase = phase * 2 * pi
        elif unit == "rad":
            omega = frequency

            # Redundant assignment, commented for LGTM
            # phase = phase
        else:
            raise ValueError("bad units: rps or rad")

        self.radius = radius
        assert len(centre) == 3, "centre must be a 3 vector"
        self.centre = centre
        self.pose = pose
        self.omega = omega
        self.phase = phase

        self.outport_names(("y",))

    def output(self, t, inports, x):
        theta = t * self.omega + self.phase
        x = self.radius * cos(theta) + self.centre[0]
        y = self.radius * sin(theta) + self.centre[1]
        p = (x, y, self.centre[2])

        if self.pose is not None:
            pp = SE3.Rt(self.pose.R, p)
            p = pp

        return [p]


class Trapezoidal(SourceBlock):
    r"""
    :blockname:`Trapezoidal`

    Trapezoidal scalar trajectory

    :inputs: 0
    :outputs: 3
    :states: 0

    .. list-table::
        :header-rows: 1

        *   - Port type
            - Port number
            - Types
            - Description
        *   - Output
            - 0
            - float
            - :math:`q(t)`
        *   - Output
            - 1
            - float
            - :math:`\dot{q}(t)`
        *   - Output
            - 2
            - float
            - :math:`\ddot{q}(t)`


    Scalar trapezoidal trajectory that varies from ``q0`` to ``qf`` over the
    simulation period.

    :seealso: :func:`ctraj`, :func:`qplot`, :func:`~SerialLink.jtraj`
    """

    nin = 0
    nout = 3
    outlabels = ("q", "qd", "qdd")

    # TODO: change name to Trapezoidal, check if used anywhere

    def __init__(self, q0, qf, V=None, T=None, **blockargs):
        """
        Compute a joint-space trajectory

        :param q0: initial joint coordinate
        :type q0: float
        :param qf: final joint coordinate
        :type qf: float
        :param T: maximum time, defaults to None
        :type T: float, optional
        :param blockargs: |BlockOptions|
        :type blockargs: dict

        If ``T`` is given the value ``qf`` is reached at this time.  This can be
        less or greater than the simulation time.
        """
        super().__init__(nout=3, **blockargs)
        self.T = T
        self.q0 = q0
        self.qf = qf

    def start(self, simstate):
        if self.T is None:
            self.T = simstate.T
        self.trapezoidalfunc = trapezoidal_func(self.q0, self.qf, self.T)

    def output(self, t, inports, x):
        return list(self.trapezoidalfunc(t))


# ------------------------------------------------------------------------ #


class Traj(FunctionBlock):
    """
    :blockname:`TRAJ`

    Vector trajectory

    :inputs: 0 or 1
    :outputs: 3
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
            - :math:`s \in [0, 1]` distance along trajectory.
        *   - Output
            - 0
            - ndarray
            - :math:`y(s)`
        *   - Output
            - 1
            - ndarray
            - :math:`\dot{y}(s)`
        *   - Output
            - 2
            - ndarray
            - :math:`\ddot{y}(s)`

    Generates a vector trajectory using a trapezoidal or quintic
    polynomial profile that varies from ``y0`` to ``yf``

    The distance along the trajectory is either:

    - a linear function from 0 to ``T`` or maximum simulation time
    - the value [0, 1] given on inport port.

    :seealso: :func:`spatialmath.base.mtraj`
    """

    nin = -1
    nout = 3
    outlabels = ("q",)

    # TODO: this needs work, need better description of what this does for
    # time-based case

    def __init__(self, y0=0, yf=1, T=None, time=False, traj="trapezoidal", **blockargs):
        """
        :param y0: initial value, defaults to 0
        :type y0: array_like(m), optional
        :param yf: final value, defaults to 1
        :type yf: array_like(m), optional
        :param T: maximum time, defaults to None
        :type T: float, optional
        :param time: x is simulation time, defaults to False
        :type time: bool, optional
        :param traj: trajectory type, one of: 'trapezoidal' [default], 'quintic'
        :type traj: str, optional
        :param blockargs: |BlockOptions|
        :type blockargs: dict
        """
        self.time = time
        if time:
            # function of time in simulation
            nin = 0
            blockclass = "source"
        else:
            # function of input port
            nin = 1
            blockclass = "function"

        super().__init__(nin=nin, blockclass=blockclass, **blockargs)

        y0 = smb.getvector(y0)
        yf = smb.getvector(yf)
        assert len(y0) == len(yf), "y0 and yf must have same length"

        self.y0 = y0
        self.yf = yf
        self.time = time
        self.T = T
        self.traj = traj

        self.outport_names(("y", "yd", "ydd"))

    def start(self, simstate):
        # if self.time:
        #     assert self.x[0] <= 0, "interpolation not defined for t=0"
        #     assert self.x[-1] >= simstate.T, "interpolation not defined for t=T"

        if self.traj == "trapezoidal":
            trajfunc = trapezoidal_func
        elif self.traj == "quintic":
            trajfunc = quintic_func

        self.trajfuncs = []

        if self.time:
            # time based
            if self.T is not None:
                xmax = self.T
            else:
                xmax = simstate.T
        else:
            # input based
            xmax = 1
        self.xmax = xmax

        for i in range(len(self.y0)):
            self.trajfuncs.append(trajfunc(self.y0[i], self.yf[i], xmax))

    def output(self, t, inports, x):
        if not self.time:
            t = inports[0]

        assert t >= 0, "interpolation not defined for x<0"
        assert t <= self.xmax, "interpolation not defined for x>" + str(self.xmax)

        out = []
        for i in range(len(self.y0)):
            out.append(self.trajfuncs[i](t))

        # we have a list of tuples out[i][j]
        # i is the timestep, j is y/yd/ydd
        y = [o[0] for o in out]
        yd = [o[1] for o in out]
        ydd = [o[2] for o in out]

        return [np.hstack(y), np.hstack(yd), np.hstack(ydd)]


# ======================================================================== #


class IDyn(FunctionBlock):
    r"""
    :blockname:`IDYN`

    Robot arm forward dynamics model.

    :inputs: 3
    :outputs: 1
    :states: 0

    .. list-table::
        :header-rows: 1

        *   - Port type
            - Port number
            - Types
            - Description
        *   - Input
            - 0
            - ndarray(N)
            - :math:`\mathit{q}`, joint configuration
        *   - Input
            - 1
            - ndarray(N)
            - :math:`\dot{\mathit{q}}`, joint velocity
        *   - Input
            - 2
            - ndarray(N)
            - :math:`\ddot{\mathit{q}}`, joint acceleration
        *   - Output
            - 0
            - ndarray(N)
            - :math:`\mathit{Q}`, generalized joint force

    Compute the generalized joint torques required to achieve the input joint
    configuration, velocity and acceleration.  This uses the recursive Newton-Euler
    (RNE) algorithm.

    .. todo:: end-effector wrench input, base wrench output, payload input

    :seealso: :meth:`~roboticstoolbox.robot.Robot.Robot.rne`
    """

    nin = 3
    nout = 1
    inlabels = ("q", "qd", "qdd")
    outlabels = "τ"

    def __init__(self, robot, gravity=None, **blockargs):
        """
        :param robot: Robot model
        :type robot: Robot subclass
        :param gravity: gravitational acceleration
        :type gravity: float
        :param blockargs: |BlockOptions|
        :type blockargs: dict
        """
        if robot is None:
            raise ValueError("robot is not defined")

        super().__init__(**blockargs)
        # self.type = "inverse-dynamics"

        self.robot = robot
        self.gravity = gravity

        # state vector is [q qd]

        self.inport_names(("q", "qd", "qdd"))
        self.outport_names(("$\tau$",))

    def output(self, t, inports, x):
        tau = self.robot.rne(inports[0], inports[1], inports[2], gravity=self.gravity)
        return [tau]


# ------------------------------------------------------------------------ #


class Gravload(FunctionBlock):
    r"""
    :blockname:`GRAVLOAD`

    Robot arm gravity load.

    :inputs: 1
    :outputs: 1
    :states: 0

    .. list-table::
        :header-rows: 1

        *   - Port type
            - Port number
            - Types
            - Description
        *   - Input
            - 0
            - ndarray(N)
            - :math:`\mathit{q}`, joint configuration
        *   - Output
            - 0
            - ndarray(N)
            - :math:`\mathit{g}`, generalized joint force

    Compute generalized joint forces due to gravity for the input joint
    configuration.

    :seealso: :meth:`~roboticstoolbox.robot.Robot.Robot.gravload`
    """

    nin = 1
    nout = 1
    inlabels = ("q",)
    outlabels = "τ"

    def __init__(self, robot, gravity=None, **blockargs):
        """
        :param robot: Robot model
        :type robot: Robot subclass
        :param gravity: gravitational acceleration
        :type gravity: float
        :param blockargs: |BlockOptions|
        :type blockargs: dict
        """
        if robot is None:
            raise ValueError("robot is not defined")

        super().__init__(**blockargs)
        # self.type = "gravload"

        self.robot = robot
        self.gravity = gravity
        self.inport_names(("q",))
        self.outport_names(("$\tau$",))

    def output(self, t, inports, x):
        tau = self.robot.gravload(inports[0], gravity=self.gravity)
        return [tau]


class Gravload_X(FunctionBlock):
    r"""
    :blockname:`GRAVLOAD_X`

    Task-space robot arm gravity wrench.

    :inputs: 1
    :outputs: 1
    :states: 0

    .. list-table::
        :header-rows: 1

        *   - Port type
            - Port number
            - Types
            - Description
        *   - Input
            - 0
            - ndarray(6)
            - :math:`\mathit{x}`, end-effector pose
        *   - Output
            - 0
            - ndarray(6)
            - :math:`\mathit{g}_x`, generalized joint force

    Compute end-effector wrench due to gravity for the input end-effector pose.

    :seealso: :meth:`~roboticstoolbox.robot.Robot.Robot.gravload_x`
        :meth:`~roboticstoolbox.robot.Robot.Robot.gravload`
    """

    nin = 1
    nout = 1
    inlabels = ("q",)
    outlabels = "w"

    def __init__(self, robot, representation="rpy/xyz", gravity=None, **blockargs):
        """
        :param robot: Robot model
        :type robot: Robot subclass
        :param representation: task-space representation, defaults to "rpy/xyz"
        :type representation: str
        :param gravity: gravitational acceleration
        :type gravity: float
        :param blockargs: |BlockOptions|
        :type blockargs: dict
        """
        if robot is None:
            raise ValueError("robot is not defined")

        super().__init__(**blockargs)
        # self.type = "gravload-x"

        self.robot = robot
        self.gravity = gravity
        self.inport_names(("q",))
        self.outport_names(("$\tau$",))
        self.representation = representation

    def output(self, t, inports, x):
        q = inports[0]
        w = self.robot.gravload_x(
            q, representation=self.representation, gravity=self.gravity
        )
        return [w]


# ------------------------------------------------------------------------ #


class Inertia(FunctionBlock):
    r"""
    :blockname:`INERTIA`

    Robot arm inertia matrix.

    :inputs: 1 [ndarray(N)]
    :outputs: 3 [ndarray(N,N)]
    :states: 0

    :inputs: 1
    :outputs: 1
    :states: 0

    .. list-table::
        :header-rows: 1

        *   - Port type
            - Port number
            - Types
            - Description
        *   - Input
            - 0
            - ndarray
            - :math:`\mathit{q}`, joint configuration
        *   - Output
            - 0
            - ndarray(N,N)
            - :math:`\mathbf{M}`, mass matrix

    Joint-space inertia matrix (mass matrix) as a function of joint configuration.

    :seealso: :meth:`~roboticstoolbox.robot.Robot.Robot.inertia`
    """

    nin = 1
    nout = 1
    inlabels = ("q",)
    outlabels = "M"

    def __init__(self, robot, **blockargs):
        """
        :param robot: Robot model
        :type robot: Robot subclass
        :param blockargs: |BlockOptions|
        :type blockargs: dict
        """
        if robot is None:
            raise ValueError("robot is not defined")

        super().__init__(**blockargs)
        # self.type = "inertia"

        self.robot = robot
        self.inport_names(("q",))
        self.outport_names(("M",))

    def output(self, t, inports, x):
        q = inports[0]
        M = self.robot.inertia(q)
        return [M]


# ------------------------------------------------------------------------ #


class Inertia_X(FunctionBlock):
    r"""
    :blockname:`INERTIA_X`

    Task-space robot arm inertia matrix.

    :inputs: 1
    :outputs: 1
    :states: 0

    .. list-table::
        :header-rows: 1

        *   - Port type
            - Port number
            - Types
            - Description
        *   - Input
            - 0
            - ndarray(6)
            - :math:`\mathit{x}`, end-effector pose
        *   - Output
            - 0
            - ndarray(6,6)
            - :math:`\mathbf{M_x}`, task-space mass matrix

    Task-space inertia matrix as a function of end-effector pose.

    :seealso: :meth:`~roboticstoolbox.robot.Robot.Robot.inertia_x`
    """

    nin = 1
    nout = 1
    inlabels = ("q",)
    outlabels = "M"

    def __init__(self, robot, representation="rpy/xyz", pinv=False, **blockargs):
        """
        :param robot: Robot model
        :type robot: Robot subclass
        :param representation: task-space representation, defaults to "rpy/xyz"
        :type representation: str
        :param blockargs: |BlockOptions|
        :type blockargs: dict
        """
        if robot is None:
            raise ValueError("robot is not defined")

        super().__init__(**blockargs)
        # self.type = "inertia-x"

        self.robot = robot
        self.representation = representation
        self.pinv = pinv
        self.inport_names(("q",))
        self.outport_names(("M",))

    def output(self, t, inports, x):
        q = inports[0]
        Mx = self.robot.inertia_x(q, pinv=self.pinv, representation=self.representation)
        return [Mx]


# ------------------------------------------------------------------------ #


class FDyn(TransferBlock):
    r"""
    :blockname:`FDYN`

    Robot arm forward dynamics.

    :inputs: 1
    :outputs: 3
    :states: 2N

    .. list-table::
        :header-rows: 1

        *   - Port type
            - Port number
            - Types
            - Description
        *   - Input
            - 0
            - ndarray(N)
            - :math:`\mathit{Q}`, generalized joint force
        *   - Output
            - 0
            - ndarray(N)
            - :math:`\mathit{q}`, joint configuration
        *   - Output
            - 1
            - ndarray(N)
            - :math:`\dot{\mathit{q}}`, joint velocity
        *   - Output
            - 2
            - ndarray(N)
            - :math:`\ddot{\mathit{q}}`, joint acceleration

    Compute the manipulator arm forward dynamics in joint space, the joint acceleration
    for the input configuration and applied joint forces.  The acceleration is
    integrated to obtain joint velocity and joint configuration.

    :seealso: :meth:`~roboticstoolbox.robot.Robot.Robot.fdyn`
    """

    nin = 1
    nout = 3
    outlabels = ("q", "qd", "qdd")
    inlabels = "τ"

    def __init__(self, robot, q0=None, **blockargs):
        """
        :param robot: Robot model
        :type robot: Robot subclass
        :param q0: Initial joint configuration
        :type q0: array_like(n)
        :param blockargs: |BlockOptions|
        :type blockargs: dict
        """
        if robot is None:
            raise ValueError("robot is not defined")

        super().__init__(**blockargs)
        # self.type = "forward-dynamics"

        self.robot = robot
        self.nstates = robot.n * 2

        # state vector is [q qd]

        self.inport_names(("$\tau$",))
        self.outport_names(("q", "qd", "qdd"))

        if q0 is None:
            q0 = np.zeros((robot.n,))
        else:
            q0 = smb.getvector(q0, robot.n)
        self._x0 = np.r_[q0, np.zeros((robot.n,))]
        self._qdd = None

    def output(self, t, inports, x):
        n = self.robot.n
        q = x[:n]
        qd = x[n:]
        qdd = self._qdd  # from last deriv
        return [q, qd, qdd]

    def deriv(self, t, inports, x):
        # return [qd qdd]
        Q = inports[0]
        n = self.robot.n
        assert len(Q) == n, "torque vector wrong size"

        q = x[:n]
        qd = x[n:]
        qdd = self.robot.accel(q, qd, Q)
        self._qdd = qdd
        return np.r_[qd, qdd]


# ------------------------------------------------------------------------ #


class FDyn_X(TransferBlock):
    r"""
    :blockname:`FDYN_X`

    Task-space robot arm forward dynamics.

    :inputs: 1
    :outputs: 3
    :states: 12

    .. list-table::
        :header-rows: 1

        *   - Port type
            - Port number
            - Types
            - Description
        *   - Input
            - 0
            - ndarray(6)
            - :math:`\mathit{\tau}`, end-effector wrench
        *   - Output
            - 0
            - ndarray(6)
            - :math:`\mathit{x}`, end-effector pose
        *   - Output
            - 1
            - ndarray(6)
            - :math:`\dot{\mathit{x}}`, end-effector velocity
        *   - Output
            - 2
            - ndarray(6)
            - :math:`\dot{\mathit{x}}``, end-effector acceleration

    Compute the manipulator arm forward dynamics in task space, the end-effector
    acceleration for the input end-effector pose and applied end-effector wrench.  The
    acceleration is integrated to obtain task-space velocity and task-space pose.

    :seealso: :meth:`~roboticstoolbox.robot.Robot.Robot.fdyn_x`
    """

    nin = 1
    nout = 5
    outlabels = ("q", "qd", "x", "xd", "xdd")
    inlabels = "w"

    def __init__(
        self,
        robot,
        q0=None,
        gravcomp=False,
        velcomp=False,
        representation="rpy/xyz",
        **blockargs,
    ):
        """
        :param robot: Robot model
        :type robot: Robot subclass
        :param q0: Initial joint configuration
        :type q0: array_like(n)
        :param gravcomp: perform gravity compensation
        :type gravcomp: bool
        :param velcomp: perform velocity term compensation
        :type velcomp: bool
        :param representation: task-space representation, defaults to "rpy/xyz"
        :type representation: str
    
        :param blockargs: |BlockOptions|
        :type blockargs: dict
        """
        if robot is None:
            raise ValueError("robot is not defined")

        super().__init__(**blockargs)
        # self.type = "forward-dynamics-x"

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
            q0 = smb.getvector(q0, robot.n)
        # append qd0, assumed to be zero
        self._x0 = np.r_[q0, np.zeros((robot.n,))]
        self._qdd = None

    def output(self, t, inports, x):
        n = self.robot.n
        q = x[:n]
        qd = x[n:]
        qdd = self._qdd  # from last deriv

        T = self.robot.fkine(q)
        x = smb.tr2x(T.A)

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
            Ja_dot = self.robot.jacob0_dot(q, qd, J0=Ja)
            xdd = Ja @ qdd + Ja_dot @ qd

        return [q, qd, x, xd, xdd]

    def deriv(self, t, inports, x):
        # return [qd qdd]

        # get current joint space state
        n = self.robot.n
        q = x[:n]
        qd = x[n:]

        # compute joint forces
        w = inports[0]
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


if __name__ == "__main__":
    from pathlib import Path

    exec(
        open(
            Path(__file__).parent.parent.parent.absolute() / "tests" / "test_blocks.py"
        ).read()
    )
