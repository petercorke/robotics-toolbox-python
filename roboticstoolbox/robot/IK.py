#!/usr/bin/env python

"""
@author Jesse Haviland
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Union
import roboticstoolbox as rtb
from dataclasses import dataclass
from spatialmath import SE3
from roboticstoolbox.tools.types import ArrayLike

try:
    import qpsolvers as qp

    _qp = True
except ImportError:  # pragma nocover
    _qp = False


@dataclass
class IKSolution:
    """
    A dataclass for representing an IK solution

    Attributes
    ----------
    q
        The joint coordinates of the solution (ndarray). Note that these
        will not be valid if failed to find a solution
    success
        True if a valid solution was found
    iterations
        How many iterations were performed
    searches
        How many searches were performed
    residual
        The final error value from the cost function
    reason
        The reason the IK problem failed if applicable


    .. versionchanged:: 1.0.3
        Added IKSolution dataclass to replace the IKsolution named tuple

    """

    q: np.ndarray
    success: bool
    iterations: int = 0
    searches: int = 0
    residual: float = 0.0
    reason: str = ""

    def __iter__(self):
        return iter(
            (
                self.q,
                self.success,
                self.iterations,
                self.searches,
                self.residual,
                self.reason,
            )
        )

    def __str__(self):
        if self.q is not None:
            q_str = np.array2string(
                self.q,
                separator=", ",
                formatter={
                    "float": lambda x: "{:.4g}".format(0 if abs(x) < 1e-6 else x)
                },
            )  # np.round(self.q, 4)
        else:
            q_str = None

        if self.iterations == 0 and self.searches == 0:
            # Check for analytic
            if self.success:
                return f"IKSolution: q={q_str}, success=True"
            else:
                return f"IKSolution: q={q_str}, success=False, reason={self.reason}"
        else:
            # Otherwise it is a numeric solution
            if self.success:
                return (
                    f"IKSolution: q={q_str}, success=True,"
                    f" iterations={self.iterations}, searches={self.searches},"
                    f" residual={self.residual:.3g}"
                )
            else:
                return (
                    f"IKSolution: q={q_str}, success=False, reason={self.reason},"
                    f" iterations={self.iterations}, searches={self.searches},"
                    f" residual={np.round(self.residual, 4):.3g}"
                )


class IKSolver(ABC):
    """
    An abstract super class for numerical inverse kinematics (IK)

    This class provides basic functionality to perform numerical IK. Superclasses
    can inherit this class and implement the `solve` method and redefine any other
    methods necessary.

    Parameters
    ----------
    name
        The name of the IK algorithm
    ilimit
        How many iterations are allowed within a search before a new search
        is started
    slimit
        How many searches are allowed before being deemed unsuccessful
    tol
        Maximum allowed residual error E
    mask
        A 6 vector which assigns weights to Cartesian degrees-of-freedom
        error priority
    joint_limits
        Reject solutions with joint limit violations
    seed
        A seed for the private RNG used to generate random joint coordinate
        vectors

    See Also
    --------
    IK_NR
        Implements this class using the Newton-Raphson method
    IK_GN
        Implements this class using the Gauss-Newton method
    IK_LM
        Implements this class using the Levemberg-Marquadt method
    IK_QP
        Implements this class using a quadratic programming approach


    .. versionchanged:: 1.0.3
        Added the abstract super class IKSolver

    """

    def __init__(
        self,
        name: str = "IK Solver",
        ilimit: int = 30,
        slimit: int = 100,
        tol: float = 1e-6,
        mask: Union[ArrayLike, None] = None,
        joint_limits: bool = True,
        seed: Union[int, None] = None,
    ):
        # Solver parameters
        self.name = name
        self.slimit = slimit
        self.ilimit = ilimit
        self.tol = tol

        # Random number generator
        self._private_random = np.random.default_rng(seed=seed)

        if mask is None:
            mask = np.ones(6)

        self.We = np.diag(mask)  # type: ignore
        self.joint_limits = joint_limits

    def solve(
        self,
        ets: "rtb.ETS",
        Tep: Union[SE3, np.ndarray],
        q0: Union[ArrayLike, None] = None,
    ) -> IKSolution:
        """
        Solves the IK problem

        This method will attempt to solve the IK problem and obtain joint coordinates
        which result the the end-effector pose `Tep`.

        Parameters
        ----------
        ets
            The ETS representing the manipulators kinematics
        Tep
            The desired end-effector pose
        q0
            The initial joint coordinate vector

        Returns
        -------
        q
            The joint coordinates of the solution (ndarray). Note that these
            will not be valid if failed to find a solution
        success
            True if a valid solution was found
        iterations
            How many iterations were performed
        searches
            How many searches were performed
        residual
            The final error value from the cost function
        jl_valid
            True if q is inbounds of the robots joint limits
        reason
            The reason the IK problem failed if applicable

        """
        # Get the largest jindex in the ETS. If this is greater than ETS.n
        # then we need to pad the q vector with zeros
        max_jindex: int = 0

        for j in ets.joints():
            if j.jindex > max_jindex:  # type: ignore
                max_jindex = j.jindex  # type: ignore

        q0_method = np.zeros((self.slimit, max_jindex + 1))

        if q0 is None:
            q0_method[:, ets.jindices] = self._random_q(ets, self.slimit)

        elif not isinstance(q0, np.ndarray):
            q0 = np.array(q0)

        if q0 is not None and q0.ndim == 1:
            q0_method[:, ets.jindices] = self._random_q(ets, self.slimit)

            q0_method[0, ets.jindices] = q0

        if q0 is not None and q0.ndim == 2:
            q0_method[:, ets.jindices] = self._random_q(ets, self.slimit)

            q0_method[: q0.shape[0], ets.jindices] = q0

        q0 = q0_method

        traj = False

        methTep: np.ndarray

        if isinstance(Tep, SE3):
            if len(Tep) > 1:
                traj = True
                methTep = np.empty((len(Tep), 4, 4))

                for i, T in enumerate(Tep):
                    methTep[i] = T.A
            else:
                methTep = Tep.A
        elif Tep.ndim == 3:
            traj = True
            methTep = Tep
        elif Tep.shape != (4, 4):
            raise ValueError("Tep must be a 4x4 SE3 matrix")
        else:
            methTep = Tep

        if traj:
            q = np.empty((methTep.shape[0], ets.n))
            success = True
            interations = 0
            searches = 0
            residual = np.inf
            reason = ""

            for i, T in enumerate(methTep):
                sol = self._solve(ets, T, q0)
                q[i] = sol.q
                if not sol.success:
                    success = False
                    reason = sol.reason
                interations += sol.iterations
                searches += sol.searches

                if sol.residual < residual:
                    residual = sol.residual

            return IKSolution(
                q=q,
                success=success,
                iterations=interations,
                searches=searches,
                residual=residual,
                reason=reason,
            )

        else:
            sol = self._solve(ets, methTep, q0)

        return sol

    def _solve(self, ets: "rtb.ETS", Tep: np.ndarray, q0: np.ndarray) -> IKSolution:
        # Iteration count
        i = 0
        total_i = 0

        # Error flags
        found_with_limits = False
        linalg_error = 0

        # Initialise variables
        E = 0.0
        q = q0[0]

        for search in range(self.slimit):
            q = q0[search].copy()
            i = 0

            while i < self.ilimit:
                i += 1

                # Attempt a step
                try:
                    E, q[ets.jindices] = self.step(ets, Tep, q)

                except np.linalg.LinAlgError:
                    # Abandon search and try again
                    linalg_error += 1
                    break

                # Check if we have arrived
                if E < self.tol:
                    # Wrap q to be within +- 180 deg
                    # If your robot has larger than 180 deg range on a joint
                    # this line should be modified in incorporate the extra range
                    q = (q + np.pi) % (2 * np.pi) - np.pi

                    # Check if we have violated joint limits
                    jl_valid = self._check_jl(ets, q)

                    if not jl_valid and self.joint_limits:
                        # Abandon search and try again
                        found_with_limits = True
                        break
                    else:
                        return IKSolution(
                            q=q[ets.jindices],
                            success=True,
                            iterations=total_i + i,
                            searches=search + 1,
                            residual=E,
                            reason="Success",
                        )
            total_i += i

        # If we make it here, then we have failed
        reason = "iteration and search limit reached"

        if linalg_error:
            reason += f", {linalg_error} numpy.LinAlgError encountered"

        if found_with_limits:
            reason += ", solution found but violates joint limits"

        return IKSolution(
            q=q,
            success=False,
            iterations=total_i,
            searches=self.slimit,
            residual=E,
            reason=reason,
        )

    def error(self, Te: np.ndarray, Tep: np.ndarray) -> Tuple[np.ndarray, float]:
        r"""
        Calculates the error between Te and Tep

        Calculates the engle axis error between current end-effector pose Te and
        the desired end-effector pose Tep. Also calulates the quadratic error E
        which is weighted by the diagonal matrix We.

        .. math::

            E = \frac{1}{2} \vec{e}^{\top} \mat{W}_e \vec{e}

        where :math:`\vec{e} \in \mathbb{R}^6` is the angle-axis error.

        Parameters
        ----------
        Te
            The current end-effector pose
        Tep
            The desired end-effector pose

        Returns
        -------
        e
            angle-axis error (6 vector)
        E
            The quadratic error weighted by We

        """
        e = rtb.angle_axis(Te, Tep)
        E = 0.5 * e @ self.We @ e

        return e, E

    @abstractmethod
    def step(
        self, ets: "rtb.ETS", Tep: np.ndarray, q: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """
        Abstract step method

        Superclasses will implement this method to perform a step of the
        implemented IK algorithm

        Parameters
        ----------
        ets
            The ETS representing the manipulators kinematics
        Tep
            The desired end-effector pose
        q
            The current joint coordinate vector

        Raises
        ------
        numpy.LinAlgError
            If a step is impossible due to a linear algebra error

        Returns
        -------
        E
            The new error value
        q
            The new joint coordinate vector

        """
        pass  # pragma: nocover

    def _random_q(self, ets: "rtb.ETS", i: int = 1) -> np.ndarray:
        """
        Generate a random valid joint configuration using a private RNG

        Generates a random q vector within the joint limits defined by
        `ets.qlim`.

        Parameters
        ----------
        ets
            The ETS representing the manipulators kinematics
        i
            number of configurations to generate

        Returns
        -------
        q
            An `i x n` ndarray of random valid joint configurations, where n
            is the number of joints in the `ets`

        """

        if i == 1:
            q = np.zeros((1, ets.n))

            for i in range(ets.n):
                q[0, i] = self._private_random.uniform(ets.qlim[0, i], ets.qlim[1, i])

        else:
            q = np.zeros((i, ets.n))

            for j in range(i):
                for i in range(ets.n):
                    q[j, i] = self._private_random.uniform(
                        ets.qlim[0, i], ets.qlim[1, i]
                    )

        return q

    def _check_jl(self, ets: "rtb.ETS", q: np.ndarray) -> bool:
        """
        Checks if the joints are within their respective limits

        Parameters
        ----------
        ets
            the ETS
        q
            the current joint coordinate vector

        Returns
        -------
        True if joints within feasible limits otherwise False

        """

        # Loop through the joints in the ETS
        for i in range(ets.n):
            # Get the corresponding joint limits
            ql0 = ets.qlim[0, i]
            ql1 = ets.qlim[1, i]

            # Check if q exceeds the limits
            if q[i] < ql0 or q[i] > ql1:
                return False

        # If we make it here, all the joints are fine
        return True


def _null_Σ(ets: "rtb.ETS", q: np.ndarray, ps: float, pi: Union[np.ndarray, float]):
    """
    Formulates a relationship between joint limits and the joint velocity.
    When this is projected into the null-space of the differential kinematics
    to attempt to avoid exceeding joint limits

    :param q: The joint coordinates of the robot
    :param ps: The minimum angle/distance (in radians or metres) in which the joint is
        allowed to approach to its limit
    :param pi: The influence angle/distance (in radians or metres) in which the velocity
        damper becomes active

    :return: Σ
    """

    if isinstance(pi, float):
        pi = pi * np.ones(ets.n)

    # Add cost to going in the direction of joint limits, if they are within
    # the influence distance
    Σ = np.zeros((ets.n, 1))

    for i in range(ets.n):
        qi = q[i]
        ql0 = ets.qlim[0, i]
        ql1 = ets.qlim[1, i]

        if qi - ql0 <= pi[i]:
            Σ[i, 0] = -np.power(((qi - ql0) - pi[i]), 2) / np.power((ps - pi[i]), 2)
        if ql1 - qi <= pi[i]:
            Σ[i, 0] = np.power(((ql1 - qi) - pi[i]), 2) / np.power((ps - pi[i]), 2)

    return -Σ


def _calc_qnull(
    ets: "rtb.ETS",
    q: np.ndarray,
    J: np.ndarray,
    λΣ: float,
    λm: float,
    ps: float,
    pi: Union[np.ndarray, float],
):
    """
    Calculates the desired null-space motion according to the gains λΣ and λm.
    This is a helper method that is used within the `step` method of an IK solver

    :return: qnull - the desired null-space motion
    """

    qnull_grad = np.zeros(ets.n)
    qnull = np.zeros(ets.n)

    # Add the joint limit avoidance if the gain is above 0
    if λΣ > 0:
        Σ = _null_Σ(ets, q, ps, pi)
        qnull_grad += (1.0 / λΣ * Σ).flatten()

    # Add the manipulability maximisation if the gain is above 0
    if λm > 0:
        Jm = ets.jacobm(q)
        qnull_grad += (1.0 / λm * Jm).flatten()

    # Calculate the null-space motion
    if λΣ > 0 or λΣ > 0:
        null_space = np.eye(ets.n) - np.linalg.pinv(J) @ J
        qnull = null_space @ qnull_grad

    return qnull.flatten()


class IK_NR(IKSolver):
    """
    Newton-Raphson Numerical Inverse Kinematics Solver

    A class which provides functionality to perform numerical inverse kinematics (IK)
    using the Newton-Raphson method. See `step` method for mathematical description.

    Note
    ----
    When using this class with redundant robots (>6 DoF), `pinv` must be set to `True`

    Parameters
    ----------
    name
        The name of the IK algorithm
    ilimit
        How many iterations are allowed within a search before a new search
        is started
    slimit
        How many searches are allowed before being deemed unsuccessful
    tol
        Maximum allowed residual error E
    mask
        A 6 vector which assigns weights to Cartesian degrees-of-freedom
        error priority
    joint_limits
        Reject solutions with joint limit violations
    seed
        A seed for the private RNG used to generate random joint coordinate
        vectors
    pinv
        If True, will use the psuedoinverse in the `step` method instead of
        the normal inverse
    kq
        The gain for joint limit avoidance. Setting to 0.0 will remove this
        completely from the solution
    km
        The gain for maximisation. Setting to 0.0 will remove this completely
        from the solution
    ps
        The minimum angle/distance (in radians or metres) in which the joint is
        allowed to approach to its limit
    pi
        The influence angle/distance (in radians or metres) in null space motion
        becomes active

    Examples
    --------
    The following example gets the ``ets`` of a ``panda`` robot object, instantiates
    the IK_NR solver class using default parameters, makes a goal pose ``Tep``,
    and then solves for the joint coordinates which result in the pose ``Tep``
    using the ``solve`` method.

    .. runblock:: pycon
    >>> import roboticstoolbox as rtb
    >>> panda = rtb.models.Panda().ets()
    >>> solver = rtb.IK_NR(pinv=True)
    >>> Tep = panda.fkine([0, -0.3, 0, -2.2, 0, 2, 0.7854])
    >>> solver.solve(panda, Tep)

    Notes
    -----
    When using the NR method, the initial joint coordinates :math:`q_0`, should correspond
    to a non-singular manipulator pose, since it uses the manipulator Jacobian. When the
    the problem is solvable, it converges very quickly. However, this method frequently
    fails to converge on the goal.

    This class supports null-space motion to assist with maximising manipulability and
    avoiding joint limits. These are enabled by setting kq and km to non-zero values.

    References
    ----------
    - J. Haviland, and P. Corke. "Manipulator Differential Kinematics Part I:
      Kinematics, Velocity, and Applications." arXiv preprint arXiv:2207.01796 (2022).
    - J. Haviland, and P. Corke. "Manipulator Differential Kinematics Part II:
      Acceleration and Advanced Applications." arXiv preprint arXiv:2207.01794 (2022).

    See Also
    --------
    IKSolver
        An abstract super class for numerical IK solvers
    IK_GN
        Implements the IKSolver class using the Gauss-Newton method
    IK_LM
        Implements the IKSolver class using the Levemberg-Marquadt method
    IK_QP
        Implements the IKSolver class using a quadratic programming approach


    .. versionchanged:: 1.0.3
        Added the Newton-Raphson IK solver class

    """  # noqa

    def __init__(
        self,
        name: str = "IK Solver",
        ilimit: int = 30,
        slimit: int = 100,
        tol: float = 1e-6,
        mask: Union[ArrayLike, None] = None,
        joint_limits: bool = True,
        seed: Union[int, None] = None,
        pinv: bool = False,
        kq: float = 0.0,
        km: float = 0.0,
        ps: float = 0.0,
        pi: Union[np.ndarray, float] = 0.3,
        **kwargs,
    ):
        super().__init__(
            name=name,
            ilimit=ilimit,
            slimit=slimit,
            tol=tol,
            mask=mask,
            joint_limits=joint_limits,
            seed=seed,
            **kwargs,
        )

        self.pinv = pinv
        self.kq = kq
        self.km = km
        self.ps = ps
        self.pi = pi

        self.name = f"NR (pinv={pinv})"

        if self.kq > 0.0:
            self.name += " Σ"

        if self.km > 0.0:
            self.name += " Jm"

    def step(
        self, ets: "rtb.ETS", Tep: np.ndarray, q: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        r"""
        Performs a single iteration of the Newton-Raphson optimisation method

        .. math::

            \vec{q}_{k+1} = \vec{q}_k + {^0\mat{J}(\vec{q}_k)}^{-1} \vec{e}_k

        Parameters
        ----------
        ets
            The ETS representing the manipulators kinematics
        Tep
            The desired end-effector pose
        q
            The current joint coordinate vector

        Raises
        ------
        numpy.LinAlgError
            If a step is impossible due to a linear algebra error

        Returns
        -------
        E
            The new error value
        q
            The new joint coordinate vector

        """

        Te = ets.eval(q)
        e, E = self.error(Te, Tep)

        J = ets.jacob0(q)

        # Null-space motion
        qnull = _calc_qnull(
            ets=ets, q=q, J=J, λΣ=self.kq, λm=self.km, ps=self.ps, pi=self.pi
        )

        if self.pinv:
            q[ets.jindices] += np.linalg.pinv(J) @ e + qnull
        else:
            q[ets.jindices] += np.linalg.inv(J) @ e + qnull

        return E, q[ets.jindices]


class IK_LM(IKSolver):
    """
    Levemberg-Marquadt Numerical Inverse Kinematics Solver

    A class which provides functionality to perform numerical inverse kinematics (IK)
    using the Levemberg-Marquadt method. See ``step`` method for mathematical description.

    Parameters
    ----------
    name
        The name of the IK algorithm
    ilimit
        How many iterations are allowed within a search before a new search
        is started
    slimit
        How many searches are allowed before being deemed unsuccessful
    tol
        Maximum allowed residual error E
    mask
        A 6 vector which assigns weights to Cartesian degrees-of-freedom
        error priority
    joint_limits
        Reject solutions with joint limit violations
    seed
        A seed for the private RNG used to generate random joint coordinate
        vectors
    k
        Sets the gain value for the damping matrix Wn in the ``step`` method. See
        notes
    method
        One of "chan", "sugihara" or "wampler". Defines which method is used
        to calculate the damping matrix Wn in the ``step`` method
    kq
        The gain for joint limit avoidance. Setting to 0.0 will remove this
        completely from the solution
    km
        The gain for maximisation. Setting to 0.0 will remove this completely
        from the solution
    ps
        The minimum angle/distance (in radians or metres) in which the joint is
        allowed to approach to its limit
    pi
        The influence angle/distance (in radians or metres) in null space motion
        becomes active

    Examples
    --------
    The following example gets the ``ets`` of a ``panda`` robot object, instantiates
    the IK_LM solver class using default parameters, makes a goal pose ``Tep``,
    and then solves for the joint coordinates which result in the pose ``Tep``
    using the `solve` method.

    .. runblock:: pycon
    >>> import roboticstoolbox as rtb
    >>> panda = rtb.models.Panda().ets()
    >>> solver = rtb.IK_LM()
    >>> Tep = panda.fkine([0, -0.3, 0, -2.2, 0, 2, 0.7854])
    >>> solver.solve(panda, Tep)

    Notes
    -----
    The value for the ``k`` kwarg will depend on the ``method`` chosen and the arm you are
    using. Use the following as a rough guide ``chan, k = 1.0 - 0.01``,
    ``wampler, k = 0.01 - 0.0001``, and ``sugihara, k = 0.1 - 0.0001``

    When using the this method, the initial joint coordinates :math:`q_0`, should correspond
    to a non-singular manipulator pose, since it uses the manipulator Jacobian.

    This class supports null-space motion to assist with maximising manipulability and
    avoiding joint limits. These are enabled by setting kq and km to non-zero values.

    References
    ----------
    - J. Haviland, and P. Corke. "Manipulator Differential Kinematics Part I:
      Kinematics, Velocity, and Applications." arXiv preprint arXiv:2207.01796 (2022).
    - J. Haviland, and P. Corke. "Manipulator Differential Kinematics Part II:
      Acceleration and Advanced Applications." arXiv preprint arXiv:2207.01794 (2022).

    See Also
    --------
    IKSolver
        An abstract super class for numerical IK solvers
    IK_NR
        Implements the IKSolver class using the Newton-Raphson method
    IK_GN
        Implements the IKSolver class using the Gauss-Newton method
    IK_QP
        Implements the IKSolver class using a quadratic programming approach


    .. versionchanged:: 1.0.3
        Added the Levemberg-Marquadt IK solver class

    """  # noqa

    def __init__(
        self,
        name: str = "IK Solver",
        ilimit: int = 30,
        slimit: int = 100,
        tol: float = 1e-6,
        mask: Union[ArrayLike, None] = None,
        joint_limits: bool = True,
        seed: Union[int, None] = None,
        k: float = 1.0,
        method="chan",
        kq: float = 0.0,
        km: float = 0.0,
        ps: float = 0.0,
        pi: Union[np.ndarray, float] = 0.3,
        **kwargs,
    ):
        super().__init__(
            name=name,
            ilimit=ilimit,
            slimit=slimit,
            tol=tol,
            mask=mask,
            joint_limits=joint_limits,
            seed=seed,
            **kwargs,
        )

        if method.lower().startswith("sugi"):
            self.method = 1
            method_name = "Sugihara"
        elif method.lower().startswith("wamp"):
            self.method = 2
            method_name = "Wampler"
        else:
            self.method = 0
            method_name = "Chan"

        self.k = k
        self.kq = kq
        self.km = km
        self.ps = ps
        self.pi = pi

        self.name = f"LM ({method_name} λ={k})"

        if self.kq > 0.0:
            self.name += " Σ"

        if self.km > 0.0:
            self.name += " Jm"

    def step(self, ets: "rtb.ETS", Tep: np.ndarray, q: np.ndarray):
        r"""
        Performs a single iteration of the Levenberg-Marquadt optimisation

        The operation is defined by the choice of `method` when instantiating the class.

        The next step is deined as

        .. math::
            \vec{q}_{k+1}
            &=
            \vec{q}_k +
            \left(
                \mat{A}_k
            \right)^{-1}
            \bf{g}_k \\
            %
            \mat{A}_k
            &=
            {\mat{J}(\vec{q}_k)}^\top
            \mat{W}_e \
            {\mat{J}(\vec{q}_k)}
            +
            \mat{W}_n

        where :math:`\mat{W}_n = \text{diag}(\vec{w_n})(\vec{w_n} \in \mathbb{R}^n_{>0})` is a
        diagonal damping matrix. The damping matrix ensures that :math:`\mat{A}_k` is
        non-singular and positive definite. The performance of the LM method largely depends
        on the choice of :math:`\mat{W}_n`.

        **Chan's Method**

        Chan proposed

        .. math::
            \mat{W}_n
            =
            λ E_k \mat{1}_n

        where λ is a constant which reportedly does not have much influence on performance.
        Use the kwarg `k` to adjust the weighting term λ.

        **Sugihara's Method**

        Sugihara proposed

        .. math::
            \mat{W}_n
            =
            E_k \mat{1}_n + \text{diag}(\hat{\vec{w}}_n)

        where :math:`\hat{\vec{w}}_n \in \mathbb{R}^n`, :math:`\hat{w}_{n_i} = l^2 \sim 0.01 l^2`,
        and :math:`l` is the length of a typical link within the manipulator. We provide the
        variable `k` as a kwarg to adjust the value of :math:`w_n`.

        **Wampler's Method**

        Wampler proposed :math:`\vec{w_n}` to be a constant. This is set through the `k` kwarg.

        Parameters
        ----------
        ets
            The ETS representing the manipulators kinematics
        Tep
            The desired end-effector pose
        q
            The current joint coordinate vector

        Raises
        ------
        numpy.LinAlgError
            If a step is impossible due to a linear algebra error

        Returns
        -------
        E
            The new error value
        q
            The new joint coordinate vector

        """  # noqa

        Te = ets.eval(q)
        e, E = self.error(Te, Tep)

        if self.method == 1:
            # Sugihara's method
            Wn = E * np.eye(ets.n) + self.k * np.eye(ets.n)
        elif self.method == 2:
            # Wampler's method
            Wn = self.k * np.eye(ets.n)
        else:
            # Chan's method
            Wn = self.k * E * np.eye(ets.n)

        J = ets.jacob0(q)
        g = J.T @ self.We @ e

        # Null-space motion
        qnull = _calc_qnull(
            ets=ets, q=q, J=J, λΣ=self.kq, λm=self.km, ps=self.ps, pi=self.pi
        )

        q[ets.jindices] += np.linalg.inv(J.T @ self.We @ J + Wn) @ g + qnull

        return E, q[ets.jindices]


class IK_GN(IKSolver):
    """
    Gauss-Newton Numerical Inverse Kinematics Solver

    A class which provides functionality to perform numerical inverse kinematics (IK)
    using the Gauss-Newton method. See `step` method for mathematical description.

    Note
    ----
    When using this class with redundant robots (>6 DoF), ``pinv`` must be set to ``True``

    Parameters
    ----------
    name
        The name of the IK algorithm
    ilimit
        How many iterations are allowed within a search before a new search
        is started
    slimit
        How many searches are allowed before being deemed unsuccessful
    tol
        Maximum allowed residual error E
    mask
        A 6 vector which assigns weights to Cartesian degrees-of-freedom
        error priority
    joint_limits
        Reject solutions with joint limit violations
    seed
        A seed for the private RNG used to generate random joint coordinate
        vectors
    pinv
        If True, will use the psuedoinverse in the `step` method instead of
        the normal inverse
    kq
        The gain for joint limit avoidance. Setting to 0.0 will remove this
        completely from the solution
    km
        The gain for maximisation. Setting to 0.0 will remove this completely
        from the solution
    ps
        The minimum angle/distance (in radians or metres) in which the joint is
        allowed to approach to its limit
    pi
        The influence angle/distance (in radians or metres) in null space motion
        becomes active

    Examples
    --------
    The following example gets the ``ets`` of a ``panda`` robot object, instantiates
    the `IK_GN` solver class using default parameters, makes a goal pose ``Tep``,
    and then solves for the joint coordinates which result in the pose ``Tep``
    using the `solve` method.

    .. runblock:: pycon
    >>> import roboticstoolbox as rtb
    >>> panda = rtb.models.Panda().ets()
    >>> solver = rtb.IK_GN(pinv=True)
    >>> Tep = panda.fkine([0, -0.3, 0, -2.2, 0, 2, 0.7854])
    >>> solver.solve(panda, Tep)

    Notes
    -----
    When using the this method, the initial joint coordinates :math:`q_0`, should correspond
    to a non-singular manipulator pose, since it uses the manipulator Jacobian. When the
    the problem is solvable, it converges very quickly.

    This class supports null-space motion to assist with maximising manipulability and
    avoiding joint limits. These are enabled by setting kq and km to non-zero values.

    References
    ----------
    - J. Haviland, and P. Corke. "Manipulator Differential Kinematics Part I:
      Kinematics, Velocity, and Applications." arXiv preprint arXiv:2207.01796 (2022).
    - J. Haviland, and P. Corke. "Manipulator Differential Kinematics Part II:
      Acceleration and Advanced Applications." arXiv preprint arXiv:2207.01794 (2022).

    See Also
    --------
    IKSolver
        An abstract super class for numerical IK solvers
    IK_NR
        Implements IKSolver using the Newton-Raphson method
    IK_LM
        Implements IKSolver using the Levemberg-Marquadt method
    IK_QP
        Implements IKSolver using a quadratic programming approach


    .. versionchanged:: 1.0.3
        Added the Gauss-Newton IK solver class

    """  # noqa

    def __init__(
        self,
        name: str = "IK Solver",
        ilimit: int = 30,
        slimit: int = 100,
        tol: float = 1e-6,
        mask: Union[ArrayLike, None] = None,
        joint_limits: bool = True,
        seed: Union[int, None] = None,
        pinv: bool = False,
        kq: float = 0.0,
        km: float = 0.0,
        ps: float = 0.0,
        pi: Union[np.ndarray, float] = 0.3,
        **kwargs,
    ):
        super().__init__(
            name=name,
            ilimit=ilimit,
            slimit=slimit,
            tol=tol,
            mask=mask,
            joint_limits=joint_limits,
            seed=seed,
            **kwargs,
        )

        self.pinv = pinv
        self.kq = kq
        self.km = km
        self.ps = ps
        self.pi = pi

        self.name = f"GN (pinv={pinv})"

        if self.kq > 0.0:
            self.name += " Σ"

        if self.km > 0.0:
            self.name += " Jm"

    def step(
        self, ets: "rtb.ETS", Tep: np.ndarray, q: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        r"""
        Performs a single iteration of the Gauss-Newton optimisation method

        The next step is defined as

        .. math::

            \vec{q}_{k+1} &= \vec{q}_k +
            \left(
            {\mat{J}(\vec{q}_k)}^\top
            \mat{W}_e \
            {\mat{J}(\vec{q}_k)}
            \right)^{-1}
            \bf{g}_k \\
            \bf{g}_k &=
            {\mat{J}(\vec{q}_k)}^\top
            \mat{W}_e
            \vec{e}_k

        where :math:`\mat{J} = {^0\mat{J}}` is the base-frame manipulator Jacobian. If
        :math:`\mat{J}(\vec{q}_k)` is non-singular, and :math:`\mat{W}_e = \mat{1}_n`, then
        the above provides the pseudoinverse solution. However, if :math:`\mat{J}(\vec{q}_k)`
        is singular, the above can not be computed and the GN solution is infeasible.

        Parameters
        ----------
        ets
            The ETS representing the manipulators kinematics
        Tep
            The desired end-effector pose
        q
            The current joint coordinate vector

        Raises
        ------
        numpy.LinAlgError
            If a step is impossible due to a linear algebra error

        Returns
        -------
        E
            The new error value
        q
            The new joint coordinate vector

        """  # noqa

        Te = ets.eval(q)
        e, E = self.error(Te, Tep)

        J = ets.jacob0(q)

        # Null-space motion
        qnull = _calc_qnull(
            ets=ets, q=q, J=J, λΣ=self.kq, λm=self.km, ps=self.ps, pi=self.pi
        )

        if self.pinv:
            q[ets.jindices] += np.linalg.pinv(J) @ e + qnull
        else:
            q[ets.jindices] += np.linalg.inv(J) @ e + qnull

        return E, q[ets.jindices]


class IK_QP(IKSolver):
    """
    Quadratic Progamming Numerical Inverse Kinematics Solver

    A class which provides functionality to perform numerical inverse kinematics (IK)
    using a quadratic progamming approach. See `step` method for mathematical
    description.

    Parameters
    ----------
    name
        The name of the IK algorithm
    ilimit
        How many iterations are allowed within a search before a new search
        is started
    slimit
        How many searches are allowed before being deemed unsuccessful
    tol
        Maximum allowed residual error E
    mask
        A 6 vector which assigns weights to Cartesian degrees-of-freedom
        error priority
    joint_limits
        Reject solutions with joint limit violations
    seed
        A seed for the private RNG used to generate random joint coordinate
        vectors
    kj
        A gain for joint velocity norm minimisation
    ks
        A gain which adjusts the cost of slack (intentional error)
    kq
        The gain for joint limit avoidance. Setting to 0.0 will remove this
        completely from the solution
    km
        The gain for maximisation. Setting to 0.0 will remove this completely
        from the solution
    ps
        The minimum angle/distance (in radians or metres) in which the joint is
        allowed to approach to its limit
    pi
        The influence angle/distance (in radians or metres) in null space motion
        becomes active

    Raises
    ------
    ImportError
        If the package ``qpsolvers`` is not installed

    Examples
    --------
    The following example gets the ``ets`` of a ``panda`` robot object, instantiates
    the `IK_QP` solver class using default parameters, makes a goal pose ``Tep``,
    and then solves for the joint coordinates which result in the pose ``Tep``
    using the `solve` method.

    .. runblock:: pycon
    >>> import roboticstoolbox as rtb
    >>> panda = rtb.models.Panda().ets()
    >>> solver = rtb.IK_QP()
    >>> Tep = panda.fkine([0, -0.3, 0, -2.2, 0, 2, 0.7854])
    >>> solver.solve(panda, Tep)

    Notes
    -----
    When using the this method, the initial joint coordinates :math:`q_0`, should correspond
    to a non-singular manipulator pose, since it uses the manipulator Jacobian. When the
    the problem is solvable, it converges very quickly.

    References
    ----------
    - J. Haviland, and P. Corke. "Manipulator Differential Kinematics Part II:
      Acceleration and Advanced Applications." arXiv preprint arXiv:2207.01794 (2022).

    See Also
    --------
    IKSolver
        An abstract super class for numerical IK solvers
    IK_NR
        Implements IKSolver class using the Newton-Raphson method
    IK_GN
        Implements IKSolver class using the Gauss-Newton method
    IK_LM
        Implements IKSolver class using the Levemberg-Marquadt method


    .. versionchanged:: 1.0.3
        Added the Quadratic Programming IK solver class

    """  # noqa

    def __init__(
        self,
        name: str = "IK Solver",
        ilimit: int = 30,
        slimit: int = 100,
        tol: float = 1e-6,
        mask: Union[ArrayLike, None] = None,
        joint_limits: bool = True,
        seed: Union[int, None] = None,
        kj=0.01,
        ks=1.0,
        kq: float = 0.0,
        km: float = 0.0,
        ps: float = 0.0,
        pi: Union[np.ndarray, float] = 0.3,
        **kwargs,
    ):
        if not _qp:  # pragma: nocover
            raise ImportError(
                "the package qpsolvers is required for this class. \nInstall using 'pip"
                " install qpsolvers'"
            )

        super().__init__(
            name=name,
            ilimit=ilimit,
            slimit=slimit,
            tol=tol,
            mask=mask,
            joint_limits=joint_limits,
            seed=seed,
            **kwargs,
        )

        self.kj = kj
        self.ks = ks
        self.kq = kq
        self.km = km
        self.ps = ps
        self.pi = pi

        self.name = "QP)"

        if self.kq > 0.0:
            self.name += " Σ"

        if self.km > 0.0:
            self.name += " Jm"

    def step(
        self, ets: "rtb.ETS", Tep: np.ndarray, q: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        r"""
        Performs a single iteration of the Gauss-Newton optimisation method

        The next step is defined as

        .. math::

            \vec{q}_{k+1} = \vec{q}_{k} + \dot{\vec{q}}.

        where the QP is defined as

        .. math::

            \min_x \quad f_o(\vec{x}) &= \frac{1}{2} \vec{x}^\top \mathcal{Q} \vec{x}+ \mathcal{C}^\top \vec{x}, \\
            \text{subject to} \quad \mathcal{J} \vec{x} &= \vec{\nu},  \\
            \mathcal{A} \vec{x} &\leq \mathcal{B},  \\
            \vec{x}^- &\leq \vec{x} \leq \vec{x}^+

        with

        .. math::

            \vec{x} &=
            \begin{pmatrix}
                \dvec{q} \\ \vec{\delta}
            \end{pmatrix} \in \mathbb{R}^{(n+6)}  \\
            \mathcal{Q} &=
            \begin{pmatrix}
                \lambda_q \mat{1}_{n} & \mathbf{0}_{6 \times 6} \\ \mathbf{0}_{n \times n} & \lambda_\delta \mat{1}_{6}
            \end{pmatrix} \in \mathbb{R}^{(n+6) \times (n+6)} \\
            \mathcal{J} &=
            \begin{pmatrix}
                \mat{J}(\vec{q}) & \mat{1}_{6}
            \end{pmatrix} \in \mathbb{R}^{6 \times (n+6)} \\
            \mathcal{C} &=
            \begin{pmatrix}
                \mat{J}_m \\ \bf{0}_{6 \times 1}
            \end{pmatrix} \in \mathbb{R}^{(n + 6)} \\
            \mathcal{A} &=
            \begin{pmatrix}
                \mat{1}_{n \times n + 6} \\
            \end{pmatrix} \in \mathbb{R}^{(l + n) \times (n + 6)} \\
            \mathcal{B} &=
            \eta
            \begin{pmatrix}
                \frac{\rho_0 - \rho_s}
                        {\rho_i - \rho_s} \\
                \vdots \\
                \frac{\rho_n - \rho_s}
                        {\rho_i - \rho_s}
            \end{pmatrix} \in \mathbb{R}^{n} \\
            \vec{x}^{-, +} &=
            \begin{pmatrix}
                \dvec{q}^{-, +} \\
                \vec{\delta}^{-, +}
            \end{pmatrix} \in \mathbb{R}^{(n+6)},

        where :math:`\vec{\delta} \in \mathbb{R}^6` is the slack vector,
        :math:`\lambda_\delta \in \mathbb{R}^+` is a gain term which adjusts the
        cost of the norm of the slack vector in the optimiser,
        :math:`\dvec{q}^{-,+}` are the minimum and maximum joint velocities, and
        :math:`\dvec{\delta}^{-,+}` are the minimum and maximum slack velocities.

        Parameters
        ----------
        ets
            The ETS representing the manipulators kinematics
        Tep
            The desired end-effector pose
        q
            The current joint coordinate vector

        Raises
        ------
        numpy.LinAlgError
            If a step is impossible due to a linear algebra error

        Returns
        -------
        E
            The new error value
        q
            The new joint coordinate vector

        """  # noqa

        Te = ets.eval(q)
        e, E = self.error(Te, Tep)
        J = ets.jacob0(q)

        if isinstance(self.pi, float):
            self.pi = self.pi * np.ones(ets.n)

        # Quadratic component of objective function
        Q = np.eye(ets.n + 6)

        # Joint velocity component of Q
        Q[: ets.n, : ets.n] *= self.kj

        # Slack component of Q
        Q[ets.n :, ets.n :] = self.ks * (1 / np.sum(np.abs(e))) * np.eye(6)

        # The equality contraints
        Aeq = np.concatenate((J, np.eye(6)), axis=1)
        beq = e.reshape((6,))

        # The inequality constraints for joint limit avoidance
        if self.kq > 0.0:
            Ain = np.zeros((ets.n + 6, ets.n + 6))
            bin = np.zeros(ets.n + 6)

            # Form the joint limit velocity damper
            Ain_l = np.zeros((ets.n, ets.n))
            Bin_l = np.zeros(ets.n)

            for i in range(ets.n):
                ql0 = ets.qlim[0, i]
                ql1 = ets.qlim[1, i]

                if ql1 - q[i] <= self.pi[i]:
                    Bin_l[i] = ((ql1 - q[i]) - self.ps) / (self.pi[i] - self.ps)
                    Ain_l[i, i] = 1

                if q[i] - ql0 <= self.pi[i]:
                    Bin_l[i] = -(((ql0 - q[i]) + self.ps) / (self.pi[i] - self.ps))
                    Ain_l[i, i] = -1

            Ain[: ets.n, : ets.n] = Ain_l
            bin[: ets.n] = (1.0 / self.kq) * Bin_l
        else:
            Ain = None
            bin = None

        # Manipulability maximisation
        if self.km > 0.0:
            Jm = ets.jacobm(q).reshape((ets.n,))
            c = np.concatenate(((1.0 / self.km) * -Jm, np.zeros(6)))
        else:
            c = np.zeros(ets.n + 6)

        xd = qp.solve_qp(Q, c, Ain, bin, Aeq, beq, lb=None, ub=None, solver="quadprog")

        if xd is None:  # pragma: nocover
            raise np.linalg.LinAlgError("QP Unsolvable")

        q += xd[: ets.n]

        return E, q


if __name__ == "__main__":  # pragma nocover
    sol = IKSolution(
        np.array([1, 2, 3]), success=True, iterations=10, searches=100, residual=0.1
    )

    a, b, c, d, e = sol

    print(a, b, c, d, e)
