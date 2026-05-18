import numpy as np
from math import sin, cos, atan2, tan, sqrt, pi

import matplotlib.pyplot as plt
import time

from bdsim.components import TransferBlock, FunctionBlock
from bdsim.graphics import GraphicsBlock


class MultiRotor(TransferBlock):
    r"""
    :blockname:`MULTIROTOR`

    Dynamic model of a multi-rotor flying robot.

    :inputs: 1
    :outputs: 1
    :states: 16

    .. list-table::
        :header-rows: 1

        *   - Port type
            - Port number
            - Types
            - Description
        *   - Input
            - 0
            - ndarray(N)
            - :math:`\varpi`, rotor velocities in (radians/sec)
        *   - Output
            - 0
            - dict
            - :math:`\mathit{x}`, vehicle state

    Dynamic model of a multi-rotor flying robot that includes rotor flapping.  The
    vehicle state is a dict containing the following items:

        - ``x`` pose in the world frame as :math:`[x, y, z, \theta_Y, \theta_P, \theta_R]`
        - ``trans`` position and velocity in the world frame as 
          :math:`[x, y, z, \dot{x}, \dot{y}, \dot{z}]`
        - ``rot`` orientation and angular rate in the world frame as 
          :math:`[\theta_Y, \theta_P, \theta_R, \dot{\theta_Y}, \dot{\theta_P}, \dot{\theta_R}]`
        - ``vb`` translational velocity in the body frame as
          :math:`[\dot{x}, \dot{y}, \dot{z}]`
        - ``w`` angular rates in the body frame as
           :math:`[\dot{\theta_Y}, \dot{\theta_P}, \dot{\theta_R}]`
        - ``a1s`` longitudinal flapping angles (radians)
        - ``b1s`` lateral flapping angles (radians)
        - ``X`` full state vector as 
          :math:`[x, y, z, \theta_Y, \theta_P, \theta_R, \dot{x}, \dot{y}, \dot{z}, \dot{\theta_Y}, \dot{\theta_P}, \dot{\theta_R}]`

    The dynamic model is a dict with the following key/value pairs.

    ===========   ==========================================
    key           description
    ===========   ==========================================
    ``nrotors``   Number of rotors (even integer)
    ``J``         Flyer rotational inertia matrix (3x3)
    ``h``         Height of rotors above CoG
    ``d``         Length of flyer arms
    ``nb``        Number of blades per rotor
    ``r``         Rotor radius
    ``c``         Blade chord
    ``e``         Flapping hinge offset
    ``Mb``        Rotor blade mass
    ``Mc``        Estimated hub clamp mass
    ``ec``        Blade root clamp displacement
    ``Ib``        Rotor blade rotational inertia
    ``Ic``        Estimated root clamp inertia
    ``mb``        Static blade moment
    ``Ir``        Total rotor inertia
    ``Ct``        Non-dim. thrust coefficient
    ``Cq``        Non-dim. torque coefficient
    ``sigma``     Rotor solidity ratio
    ``thetat``    Blade tip angle
    ``theta0``    Blade root angle
    ``theta1``    Blade twist angle
    ``theta75``   3/4 blade angle
    ``thetai``    Blade ideal root approximation
    ``a``         Lift slope gradient
    ``A``         Rotor disc area
    ``gamma``     Lock number
    ===========   ==========================================

    .. note::
        - Based on MATLAB code developed by Pauline Pounds 2004.
        - SI units are used.
        - Rotor velocity is defined looking down, clockwise from the front rotor
          which lies on the x-axis.

    :References:
        - Design, Construction and Control of a Large Quadrotor micro air vehicle.
          P.Pounds, `PhD thesis <https://openresearch-repository.anu.edu.au/handle/1885/146543>`_
          Australian National University, 2007.
        - Robotics, Vision & Control by Peter Corke, sec 4.2 in all editions

    :seealso: :class:`MultiRotorMixer` :class:`MultiRotorPlot`
    """

    nin = 1
    nout = 1

    # Flyer2dynamics lovingly coded by Paul Pounds, first coded 12/4/04
    # A simulation of idealised X-4 Flyer II flight dynamics.
    # version 2.0 2005 modified to be compatible with latest version of Matlab
    # version 3.0 2006 fixed rotation matrix problem
    # version 4.0 4/2/10, fixed rotor flapping rotation matrix bug, mirroring
    # version 5.0 8/8/11, simplified and restructured
    # version 6.0 25/10/13, fixed rotation matrix/inverse wronskian definitions, flapping cross-product bug
    #
    # New in version 2:
    #   - Generalised rotor thrust model
    #   - Rotor flapping model
    #   - Frame aerodynamic drag model
    #   - Frame aerodynamic surfaces model
    #   - Internal motor model
    #   - Much coolage
    #
    # Version 1.3
    #   - Rigid body dynamic model
    #   - Rotor gyroscopic model
    #   - External motor model
    #
    # ARGUMENTS
    #   u       Reference inputs                1x4
    #   tele    Enable telemetry (1 or 0)       1x1
    #   crash   Enable crash detection (1 or 0) 1x1
    #   init    Initial conditions              1x12
    #
    # INPUTS
    #   u = [N S E W]
    #   NSEW motor commands                     1x4
    #
    # CONTINUOUS STATES
    #   z      Position                         3x1   (x,y,z)
    #   v      Velocity                         3x1   (xd,yd,zd)
    #   n      Attitude                         3x1   (Y,P,R)
    #   o      Angular velocity                 3x1   (wx,wy,wz)
    #   w      Rotor angular velocity           4x1
    #
    # Notes: z-axis downward so altitude is -z(3)
    #
    # CONTINUOUS STATE MATRIX MAPPING
    #   x = [z1 z2 z3 n1 n2 n3 z1 z2 z3 o1 o2 o3 w1 w2 w3 w4]
    #
    #
    # CONTINUOUS STATE EQUATIONS
    #   z` = v
    #   v` = g*e3 - (1/m)*T*R*e3
    #   I*o` = -o X I*o + G + torq
    #   R = f(n)
    #   n` = inv(W)*o
    #
    def __init__(self, model, groundcheck=True, speedcheck=True, x0=None, **blockargs):
        r"""
        Create a multi-rotor dynamic model block.

        :param model: A dictionary of vehicle geometric and inertial properties
        :type model: dict
        :param groundcheck: Prevent vehicle moving below ground :math:`z>0`, defaults to True
        :type groundcheck: bool
        :param speedcheck: Check for non-positive rotor speed, defaults to True
        :type speedcheck: bool
        :param x0: Initial state, defaults to None
        :type x0: array_like(6) or array_like(12), optional
        :param blockargs: |BlockOptions|
        :type blockargs: dict
        """
        if model is None:
            raise ValueError("no model provided")

        super().__init__(nin=1, nout=1, **blockargs)

        try:
            nrotors = model["nrotors"]
        except KeyError:
            raise RuntimeError("vehicle model does not contain nrotors")
        assert nrotors % 2 == 0, "Must have an even number of rotors"

        self.nstates = 12
        if x0 is None:
            x0 = np.zeros((self.nstates,))
        else:
            x0 = np.r_[x0]
            if len(x0) == 6:
                # assume all derivative are zero
                x0 = np.r_[x0, np.zeros((6,))]
            elif len(x0) == 4:
                # assume x,y,z,yaw
                x0 = np.r_[x0[:3], 0, 0, x0[3], np.zeros((6,))]
            elif len(x0) == 3:
                # assume x,y,z
                x0 = np.r_[x0[:3], np.zeros((9,))]
            elif len(x0) != self.nstates:
                raise ValueError("x0 is the wrong length")

        self._x0 = x0

        self.nrotors = nrotors
        self.model = model

        self.groundcheck = groundcheck
        self.speedcheck = speedcheck

        self.D = np.zeros((3, self.nrotors))
        self.theta = np.zeros((self.nrotors,))
        for i in range(0, self.nrotors):
            theta = i / self.nrotors * 2 * pi
            #  Di      Rotor hub displacements (1x3)
            # first rotor is on the x-axis, clockwise order looking down from above
            self.D[:, i] = np.r_[
                model["d"] * cos(theta), model["d"] * sin(theta), model["h"]
            ]
            self.theta[i] = theta

        self.a1s = np.zeros((self.nrotors,))
        self.b1s = np.zeros((self.nrotors,))

    def output(self, t, inports, x):

        model = self.model

        # compute output vector as a function of state vector
        #   z      Position                         3x1   (x,y,z)
        #   v      Velocity                         3x1   (xd,yd,zd)
        #   n      Attitude                         3x1   (Y,P,R)
        #   o      Angular velocity                 3x1   (Yd,Pd,Rd)

        n = x[3:6]  # RPY angles
        phi = n[0]  # yaw
        the = n[1]  # pitch
        psi = n[2]  # roll

        # rotz(phi)*roty(the)*rotx(psi)
        #  BBF > Inertial rotation matrix
        R = np.array(
            [
                [
                    cos(the) * cos(phi),
                    sin(psi) * sin(the) * cos(phi) - cos(psi) * sin(phi),
                    cos(psi) * sin(the) * cos(phi) + sin(psi) * sin(phi),
                ],
                [
                    cos(the) * sin(phi),
                    sin(psi) * sin(the) * sin(phi) + cos(psi) * cos(phi),
                    cos(psi) * sin(the) * sin(phi) - sin(psi) * cos(phi),
                ],
                [-sin(the), sin(psi) * cos(the), cos(psi) * cos(the)],
            ]
        )

        # inverted Wronskian
        iW = np.array(
            [
                [0, sin(psi), cos(psi)],
                [0, cos(psi) * cos(the), -sin(psi) * cos(the)],
                [cos(the), sin(psi) * sin(the), cos(psi) * sin(the)],
            ]
        ) / cos(the)

        # return velocity in the body frame

        vd = np.linalg.inv(R) @ x[6:9]  # translational velocity mapped to body frame
        rpyd = iW @ x[9:12]  # RPY rates mapped to body frame

        out = {}

        out["x"] = x[0:6]
        out["trans"] = np.r_[x[:3], vd]
        out["rot"] = np.r_[x[3:6], rpyd]
        out["vb"] = np.linalg.inv(R) @ x[6:9]   # translational velocity mapped to body frame
        out["w"] = iW @ x[9:12]                 # RPY rates mapped to body frame

        out["a1s"] = self.a1s
        out["b1s"] = self.b1s
        out["X"] = np.r_[x[:6], vd, rpyd]

        # sys = [ x(1:6);
        #     inv(R)*x(7:9);   % translational velocity mapped to body frame
        #     iW*x(10:12)];

        return [out]

    def deriv(self, t, inports, x):

        model = self.model

        # Body-fixed frame references
        #   ei      Body fixed frame references 3x1
        e3 = np.r_[0, 0, 1]

        # process inputs
        w = inports[0]
        if len(w) != self.nrotors:
            raise RuntimeError("input vector wrong size")

        if self.speedcheck and np.any(w == 0):
            # might need to fix this, preculudes aerobatics :(
            # mu becomes NaN due to 0/0
            raise RuntimeError("quadrotor_dynamics: not defined for zero rotor speed")

        # EXTRACT STATES FROM X
        z = x[0:3]  # position in {W}
        n = x[3:6]  # RPY angles {W}
        v = x[6:9]  # velocity in {W}
        o = x[9:12]  # angular velocity in {W}

        # PREPROCESS ROTATION AND WRONSKIAN MATRICIES
        phi = n[0]  # yaw
        the = n[1]  # pitch
        psi = n[2]  # roll

        # phi = n(1);    % yaw
        # the = n(2);    % pitch
        # psi = n(3);    % roll

        # rotz(phi)*roty(the)*rotx(psi)
        # BBF > Inertial rotation matrix
        R = np.array(
            [
                [
                    cos(the) * cos(phi),
                    sin(psi) * sin(the) * cos(phi) - cos(psi) * sin(phi),
                    cos(psi) * sin(the) * cos(phi) + sin(psi) * sin(phi),
                ],
                [
                    cos(the) * sin(phi),
                    sin(psi) * sin(the) * sin(phi) + cos(psi) * cos(phi),
                    cos(psi) * sin(the) * sin(phi) - sin(psi) * cos(phi),
                ],
                [-sin(the), sin(psi) * cos(the), cos(psi) * cos(the)],
            ]
        )

        # Manual Construction
        #     Q3 = [cos(phi) -sin(phi) 0;sin(phi) cos(phi) 0;0 0 1];   % RZ %Rotation mappings
        #     Q2 = [cos(the) 0 sin(the);0 1 0;-sin(the) 0 cos(the)];   % RY
        #     Q1 = [1 0 0;0 cos(psi) -sin(psi);0 sin(psi) cos(psi)];   % RX
        #     R = Q3*Q2*Q1    %Rotation matrix
        #
        #    RZ * RY * RX
        # inverted Wronskian
        iW = np.array(
            [
                [0, sin(psi), cos(psi)],
                [0, cos(psi) * cos(the), -sin(psi) * cos(the)],
                [cos(the), sin(psi) * sin(the), cos(psi) * sin(the)],
            ]
        ) / cos(the)

        #     % rotz(phi)*roty(the)*rotx(psi)
        # R = [cos(the)*cos(phi) sin(psi)*sin(the)*cos(phi)-cos(psi)*sin(phi) cos(psi)*sin(the)*cos(phi)+sin(psi)*sin(phi);   %BBF > Inertial rotation matrix
        #      cos(the)*sin(phi) sin(psi)*sin(the)*sin(phi)+cos(psi)*cos(phi) cos(psi)*sin(the)*sin(phi)-sin(psi)*cos(phi);
        #      -sin(the)         sin(psi)*cos(the)                            cos(psi)*cos(the)];

        # iW = [0        sin(psi)          cos(psi);             %inverted Wronskian
        #       0        cos(psi)*cos(the) -sin(psi)*cos(the);
        #       cos(the) sin(psi)*sin(the) cos(psi)*sin(the)] / cos(the);

        # ROTOR MODEL
        T = np.zeros((3, 4))
        Q = np.zeros((3, 4))
        tau = np.zeros((3, 4))

        a1s = self.a1s
        b1s = self.b1s

        for i in range(0, self.nrotors):  # for each rotor

            # Relative motion
            Vr = np.cross(o, self.D[:, i]) + v
            mu = sqrt(np.sum(Vr[0:2] ** 2)) / (
                abs(w[i]) * model["r"]
            )  # Magnitude of mu, planar components
            lc = Vr[2] / (abs(w[i]) * model["r"])  # Non-dimensionalised normal inflow
            li = mu  # Non-dimensionalised induced velocity approximation
            alphas = atan2(lc, mu)
            j = atan2(Vr[1], Vr[0])  # Sideslip azimuth relative to e1 (zero over nose)
            J = np.array(
                [[cos(j), -sin(j)], [sin(j), cos(j)]]
            )  # BBF > mu sideslip rotation matrix

            # Flapping
            beta = np.array(
                [
                    [
                        (
                            (8 / 3 * model["theta0"] + 2 * model["theta1"]) * mu
                            - 2 * lc * mu
                        )
                        / (1 - mu**2 / 2)
                    ],  # Longitudinal flapping
                    [0],  # Lattitudinal flapping (note sign)
                ]
            )

            # sign(w) * (4/3)*((Ct/sigma)*(2*mu*gamma/3/a)/(1+3*e/2/r) + li)/(1+mu^2/2)];

            beta = J.T @ beta
            # Rotate the beta flapping angles to longitudinal and lateral coordinates.
            a1s[i] = beta[0] - 16 / model["gamma"] / abs(w[i]) * o[1]
            b1s[i] = beta[1] - 16 / model["gamma"] / abs(w[i]) * o[0]

            # Forces and torques

            # Rotor thrust, linearised angle approximations

            T[:, i] = (
                model["Ct"]
                * model["rho"]
                * model["A"]
                * model["r"] ** 2
                * w[i] ** 2
                * np.r_[
                    -cos(b1s[i]) * sin(a1s[i]), sin(b1s[i]), -cos(a1s[i]) * cos(b1s[i])
                ]
            )

            # Rotor drag torque - note that this preserves w[i] direction sign

            Q[:, i] = (
                -model["Cq"]
                * model["rho"]
                * model["A"]
                * model["r"] ** 3
                * w[i]
                * abs(w[i])
                * e3
            )

            tau[:, i] = np.cross(T[:, i], self.D[:, i])  # Torque due to rotor thrust

        # print(f"{tau=}")
        # print(f"{T=}")
        # RIGID BODY DYNAMIC MODEL
        dz = v
        dn = iW @ o

        dv = model["g"] * e3 + R @ np.sum(T, axis=1) / model["M"]
        do = -np.linalg.inv(model["J"]) @ (
            np.cross(o, model["J"] @ o) + np.sum(tau, axis=1) + np.sum(Q, axis=1)
        )  # row sum of torques

        # dv = quad.g*e3 + R*(1/quad.M)*sum(T,2);
        # do = inv(quad.J)*(cross(-o,quad.J*o) + sum(tau,2) + sum(Q,2)); %row sum of torques

        # vehicle can't fall below ground, remember z is down
        if self.groundcheck and z[2] > 0:
            z[0] = 0
            dz[0] = 0

        # # stash the flapping information for plotting
        # self.a1s = a1s
        # self.b1s = b1s

        return np.r_[dz, dn, dv, do]  # This is the state derivative vector


# ------------------------------------------------------------------------ #


class MultiRotorMixer(FunctionBlock):
    r"""
    :blockname:`MULTIROTORMIXER`

    Speed mixer for a multi-rotor flying vehicle.

    :inputs: 4
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
            - float
            - :math:`\tau_R`, roll torque
        *   - Input
            - 1
            - float
            - :math:`\tau_P`, pitch torque
        *   - Input
            - 2
            - float
            - :math:`\tau_Y`, yaw torque
        *   - Input
            - 3
            - float
            - :math:`T`, total thrust
        *   - Output
            - 0
            - ndarray(N)
            - :math:`\varpi`, rotor speeds

    This block converts airframe moments and total thrust into a 1D
    array of rotor speeds which can be input to the ``MULTIROTOR`` block.

    The model is a dict with the following key/value pairs.

    ===========   ==========================================
    key           description
    ===========   ==========================================
    ``nrotors``   Number of rotors (even integer)
    ``h``         Height of rotors above CoG
    ``d``         Length of flyer arms
    ``r``         Rotor radius
    ===========   ==========================================

    .. note:: Based on MATLAB code developed by Pauline Pounds 2004.

    :seealso: :class:`MultiRotor` :class:`MultiRotorPlot`
    """

    nin = 4
    nout = 1
    inlabels = ("𝛕r", "𝛕p", "𝛕y", "T")
    outlabels = ("ω",)

    def __init__(self, model=None, wmax=1000, wmin=5, **blockargs):
        """
        :param model: A dictionary of vehicle geometric and inertial properties
        :type model: dict
        :param maxw: maximum rotor speed in rad/s, defaults to 1000
        :type maxw: float
        :param minw: minimum rotor speed in rad/s, defaults to 5
        :type minw: float
        :param blockargs: |BlockOptions|
        :type blockargs: dict
        """
        if model is None:
            raise ValueError("no model provided")

        super().__init__(**blockargs)
        self.type = "multirotormixer"
        self.model = model
        self.nrotors = model["nrotors"]
        self.minw = wmin**2
        self.maxw = wmax**2
        self.theta = np.arange(self.nrotors) / self.nrotors * 2 * np.pi

        # build the Nx4 mixer matrix
        M = []
        s = []
        for i in range(self.nrotors):
            # roll and pitch coupling
            column = np.r_[
                -sin(self.theta[i]) * model["d"] * model["b"],
                cos(self.theta[i]) * model["d"] * model["b"],
                model["k"] if (i % 2) == 0 else -model["k"],
                -model["b"],
            ]
            s.append(1 if (i % 2) == 0 else -1)
            M.append(column)
        self.M = np.array(M).T
        self.Minv = np.linalg.inv(self.M)
        self.signs = np.array(s)

    def output(self, t, inports, x):
        tau = inports

        # mix airframe force/torque to rotor thrusts
        w = self.Minv @ tau

        # clip the rotor speeds to the range [minw, maxw]
        w = np.clip(w, self.minw, self.maxw)

        # convert required thrust to rotor speed
        w = np.sqrt(w)

        # flip the signs of alternating rotors
        w = self.signs * w

        return [w]


# ------------------------------------------------------------------------ #


class MultiRotorPlot(GraphicsBlock):
    """
    :blockname:`MULTIROTORPLOT`

    Displays/animates a multi-rotor flying vehicle.

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
            - dict
            - :math:`\mathit{x}`, vehicle state

    Animate a multi-rotor flying vehicle using Matplotlib graphics.  The
    rotors are shown as circles and their orientation includes rotor
    flapping which can be exagerated by ``flapscale``.

    .. figure:: ../figs/multirotorplot.png
        :width: 500px
        :alt: example of generated graphic

        Example of quad-rotor display.

    The input is a dictionary signal and the block requires the items:

            - ``x`` pose in the world frame as :math:`[x, y, z, \theta_Y, \theta_P, \theta_R]`
            - ``a1s`` rotor flap angle
            - ``b1s`` rotor flap angle

    The model is a dict with the following key/value pairs.

    ===========   ==========================================
    key           description
    ===========   ==========================================
    ``nrotors``   Number of rotors (even integer)
    ``h``         Height of rotors above CoG
    ``d``         Length of flyer arms
    ``r``         Rotor radius
    ===========   ==========================================

    .. note:: Based on MATLAB code developed by Pauline Pounds 2004.

    :seealso: :class:`MultiRotor` :class:`MultiRotorMixer`
    """

    nin = 1
    nout = 0
    inlabels = ("x",)

    # Based on code lovingly coded by Paul Pounds, first coded 17/4/02
    # version 2 2004 added scaling and ground display
    # version 3 2010 improved rotor rendering and fixed mirroring bug

    # Displays X-4 flyer position and attitude in a 3D plot.
    # GREEN ROTOR POINTS NORTH
    # BLUE ROTOR POINTS EAST

    # PARAMETERS
    # s defines the plot size in meters
    # swi controls flyer attitude plot; 1 = on, otherwise off.

    # INPUTS
    # 1 Center X position
    # 2 Center Y position
    # 3 Center Z position
    # 4 Yaw angle in rad
    # 5 Pitch angle in rad
    # 6 Roll angle in rad

    nin = 1
    nout = 0
    inlabels = ("x",)

    def __init__(
        self,
        model,
        scale=[-2, 2, -2, 2, 10],
        flapscale=1,
        projection="ortho",
        **blockargs,
    ):
        """
        :param model: A dictionary of vehicle geometric and inertial properties
        :type model: dict
        :param scale: dimensions of workspace: xmin, xmax, ymin, ymax, zmin, zmax, defaults to [-2,2,-2,2,10]
        :type scale: array_like, optional
        :param flapscale: exagerate flapping angle by this factor, defaults to 1
        :type flapscale: float
        :param projection: 3D projection, one of: 'ortho' [default], 'perspective'
        :type projection: str
        :param blockargs: |BlockOptions|
        :type blockargs: dict
        """
        if model is None:
            raise ValueError("no model provided")

        super().__init__(nin=1, **blockargs)
        self.type = "quadrotorplot"
        self.model = model
        self.scale = scale
        self.nrotors = model["nrotors"]
        self.projection = projection
        self.flapscale = flapscale

    def start(self, simstate):
        quad = self.model

        # vehicle dimensons
        d = quad["d"]
        # Hub displacement from COG
        r = quad["r"]
        # Rotor radius

        # C = np.zeros((3, self.nrotors))   ## WHERE USED?
        self.D = np.zeros((3, self.nrotors))

        for i in range(0, self.nrotors):
            theta = i / self.nrotors * 2 * pi
            #  Di      Rotor hub displacements (1x3)
            # first rotor is on the x-axis, clockwise order looking down from above
            self.D[:, i] = np.r_[
                quad["d"] * cos(theta), quad["d"] * sin(theta), quad["h"]
            ]

        # draw ground
        self.fig = self.create_figure(simstate)
        # no axes in the figure, create a 3D axes
        self.ax = self.fig.add_subplot(111, projection="3d", proj_type=self.projection)

        # ax.set_aspect('equal')
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("-Z (height above ground)")

        self.panel = self.ax.text2D(
            0.05,
            0.95,
            "",
            transform=self.ax.transAxes,
            fontsize=10,
            family="monospace",
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="black"),
        )

        # TODO allow user to set maximum height of plot volume
        self.ax.set_xlim(self.scale[0], self.scale[1])
        self.ax.set_ylim(self.scale[2], self.scale[3])
        self.ax.set_zlim(0, self.scale[4])

        # plot the ground boundaries and the big cross
        self.ax.plot(
            [self.scale[0], self.scale[1]], [self.scale[2], self.scale[3]], [0, 0], "b-"
        )
        self.ax.plot(
            [self.scale[0], self.scale[1]], [self.scale[3], self.scale[2]], [0, 0], "b-"
        )
        self.ax.grid(True)

        (self.shadow,) = self.ax.plot([0, 0], [0, 0], "k--")
        (self.groundmark,) = self.ax.plot([0], [0], [0], "kx")

        self.arm = []
        self.disk = []
        for i in range(0, self.nrotors):
            (h,) = self.ax.plot([0], [0], [0])
            self.arm.append(h)
            if i == 0:
                color = "b-"
            else:
                color = "g-"
            (h,) = self.ax.plot([0], [0], [0], color)
            self.disk.append(h)

        self.a1s = np.zeros((self.nrotors,))
        self.b1s = np.zeros((self.nrotors,))

        plt.draw()
        plt.show(block=False)

        super().start(simstate)

    def step(self, t, inports):
        def plot3(h, x, y, z):
            h.set_data_3d(x, y, z)
            # h.set_data(x, y)
            # h.set_3d_properties(np.r_[z])

        # read UAV output "bus"
        input = inports[0]
        z = input["x"][0:3]
        n = input["x"][3:6]

        # TODO, check input dimensions, 12 or 12+2N, deal with flapping

        a1s = input["a1s"]
        b1s = input["b1s"]

        quad = self.model

        # vehicle dimensons
        d = quad["d"]  # Hub displacement from COG
        r = quad["r"]  # Rotor radius

        # PREPROCESS ROTATION MATRIX
        phi, the, psi = n  # Euler angles

        # BBF > Inertial rotation matrix
        R = np.array(
            [
                [
                    cos(the) * cos(phi),
                    sin(psi) * sin(the) * cos(phi) - cos(psi) * sin(phi),
                    cos(psi) * sin(the) * cos(phi) + sin(psi) * sin(phi),
                ],
                [
                    cos(the) * sin(phi),
                    sin(psi) * sin(the) * sin(phi) + cos(psi) * cos(phi),
                    cos(psi) * sin(the) * sin(phi) - sin(psi) * cos(phi),
                ],
                [-sin(the), sin(psi) * cos(the), cos(psi) * cos(the)],
            ]
        )

        # Manual Construction
        # Q3 = [cos(psi) -sin(psi) 0;sin(psi) cos(psi) 0;0 0 1];   %Rotation mappings
        # Q2 = [cos(the) 0 sin(the);0 1 0;-sin(the) 0 cos(the)];
        # Q1 = [1 0 0;0 cos(phi) -sin(phi);0 sin(phi) cos(phi)];
        # R = Q3*Q2*Q1;    %Rotation matrix

        # CALCULATE FLYER TIP POSITONS USING COORDINATE FRAME ROTATION
        F = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

        # Draw flyer rotors
        theta = np.linspace(0, 2 * pi, 20)
        circle = np.zeros((3, 20))
        for j, t in enumerate(theta):
            circle[:, j] = np.r_[r * sin(t), r * cos(t), 0]

        hub = np.zeros((3, self.nrotors))
        tippath = np.zeros((3, 20, self.nrotors))
        for i in range(0, self.nrotors):
            hub[:, i] = F @ (z + R @ self.D[:, i])  # points in the inertial frame

            q = (
                self.flapscale
            )  # Flapping angle scaling for output display - makes it easier to see what flapping is occurring
            # Rotor -> Plot frame
            Rr = np.array(
                [
                    [
                        cos(q * a1s[i]),
                        sin(q * b1s[i]) * sin(q * a1s[i]),
                        cos(q * b1s[i]) * sin(q * a1s[i]),
                    ],
                    [0, cos(q * b1s[i]), -sin(q * b1s[i])],
                    [
                        -sin(q * a1s[i]),
                        sin(q * b1s[i]) * cos(q * a1s[i]),
                        cos(q * b1s[i]) * cos(q * a1s[i]),
                    ],
                ]
            )

            tippath[:, :, i] = F @ R @ Rr @ circle
            plot3(
                self.disk[i],
                hub[0, i] + tippath[0, :, i],
                hub[1, i] + tippath[1, :, i],
                hub[2, i] + tippath[2, :, i],
            )

        # Draw flyer
        hub0 = F @ z  # centre of vehicle
        for i in range(0, self.nrotors):
            # line from hub to centre plot3([hub(1,N) hub(1,S)],[hub(2,N) hub(2,S)],[hub(3,N) hub(3,S)],'-b')
            plot3(
                self.arm[i],
                [hub[0, i], hub0[0]],
                [hub[1, i], hub0[1]],
                [hub[2, i], hub0[2]],
            )

            # plot a circle at the hub itself
            # plot3([hub(1,i)],[hub(2,i)],[hub(3,i)],'o')

        # plot the vehicle's centroid on the ground plane
        plot3(self.shadow, [z[0], 0], [-z[1], 0], [0, 0])
        plot3(self.groundmark, [z[0]], [-z[1]], [0])

        textstr = f"t={t: .2f}\nh={z[2]: .2f}\nγ={n[0]: .2f}"
        self.panel.set_text(textstr)

        super().step(t, inports)


if __name__ == "__main__":

    from quad_model import quadrotor

    # m = MultiRotorMixer(model=quadrotor)
    # print(m.M)
    # print(m.Minv)
    # # print(m.Minv @ [0, 0, 0, -40])
    # m.inputs = [0, 0, 0, -40]
    # print(m.output(0.0))
    # m.inputs = [0, 0, 0, -50]
    # print(m.output(0.0))

    # m.inputs = [0, 0, 0.1, -40]
    # print(m.output(0.0))

    # m.inputs = [1, 0, 0, -40]
    # print(m.output(0.0))

    # m.inputs = [0, 1, 0, -40]
    # print(m.output(0.0))

    m = MultiRotor(model=quadrotor)

    def show(w):
        print()
        print(w[0] ** 2 + w[2] ** 2 - w[1] ** 2 - w[3] ** 2)
        print(w)

        x = np.r_[0.0, 0, -4, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        inputs = [np.r_[w]]

        dx = m.deriv(0.0, inputs, x)

        print("zdd", dx[8])
        print("wd", dx[9:12])

        x = dx
        x = m.output(0.0, inputs, x)[0]["X"]
        print("zd", x[8])
        print("ypr_dot", x[9:12])

    show([800.0, -800, 800, -800])

    # tau_y pitch
    z = np.sqrt((900**2 + 700**2) / 2)
    show([900.0, -z, 700, -z])
    show([700.0, -z, 900, -z])

    # tau_x roll
    show([z, -900, z, -700])
    show([z, -700, z, -900])

    # tau_z roll
    show([900, -800, 900, -800])
