#!/usr/bin/env python
"""
@author Jesse Haviland
"""

# from __future__ import annotations
import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Union
import roboticstoolbox as rtb
from dataclasses import dataclass
from spatialmath import SE3

ArrayLike = Union[list, np.ndarray, tuple, set]


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
    """

    q: Union[np.ndarray, None]
    success: bool
    iterations: int = 0
    searches: int = 0
    residual: float = 0.0
    reason: str = ""

    def __str__(self):

        if self.q is not None:
            q_str = np.round(self.q, 4)
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
                return f"IKSolution: q={q_str}, success=True, iterations={self.iterations}, searches={self.searches}, residual={self.residual}"
            else:
                return f"IKSolution: q={q_str}, success=False, reason={self.reason}, iterations={self.iterations}, searches={self.searches}, residual={np.round(self.residual, 4)}"


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
    roboticstoolbox.robot.IK.IK_NR
        Implements this class using the Newton-Raphson method
    roboticstoolbox.robot.IK.IK_GN
        Implements this class using the Gauss-Newton method
    roboticstoolbox.robot.IK.IK_LM
        Implements this class using the Levemberg-Marquadt method

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

        self.We = np.diag(mask)
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
        q
            The current joint coordinate vector

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

        if q0 is None:
            q0 = self._random_q(ets, self.slimit)
        elif not isinstance(q0, np.ndarray):
            q0 = np.array(q0)

        if q0.ndim == 1:
            q0_new = self._random_q(ets, self.slimit)
            q0_new[0] = q0
            q0 = q0_new

        if isinstance(Tep, SE3):
            Tep = Tep.A

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
                    E, q = self.step(ets, Tep, q)

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
                            q=q,
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
            reason += f", {linalg_error} np.LinAlgError encountered"

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
        """
        Calculates the error between Te and Tep

        Calculates the engle axis error between current end-effector pose Te and
        the desired end-effector pose Tep. Also calulates the quadratic error E
        which is weighted by the diagonal matrix We.

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

        Returns
        -------
        E
            The new error value
        q
            The new joint coordinate vector

        """
        pass

    def _random_q(self, ets: "rtb.ETS", i: int = 1) -> np.ndarray:
        """
        Generate a random valid joint configuration using a private RNG

        Generates a random q vector within the joint limits defined by
        `self.qlim`.

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
            q = np.zeros(ets.n)

            for i in range(ets.n):
                q[i] = self._private_random.uniform(ets.qlim[0, i], ets.qlim[1, i])

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
    .. [1] J. Haviland, Jesse, and P. Corke. "Manipulator Differential Kinematics Part I:
           Kinematics, Velocity, and Applications." arXiv preprint arXiv:2207.01796 (2022).
    .. [2] J. Haviland, Jesse, and P. Corke. "Manipulator Differential Kinematics Part II:
           Acceleration and Advanced Applications." arXiv preprint arXiv:2207.01794 (2022).

    Examples
    --------
    The following example gets the `ets` of a `panda` robot object, instantiates
    the IK_NR solver class using default parameters, makes a goal pose `Tep`,
    and then solves for the joint coordinates which result in the pose `Tep`
    using the `solve` method.

    .. runblock:: pycon

        >>> import roboticstoolbox as rtb
        >>> panda = rtb.models.Panda().ets()
        >>> solver = rtb.IK_NR(pinv=True)
        >>> Tep = panda.fkine([0, -0.3, 0, -2.2, 0, 2, 0.7854])
        >>> solver.solve(panda, Tep)

    See Also
    --------
    roboticstoolbox.robot.IK.IKSolver
        An abstract super class for numerical IK solvers
    roboticstoolbox.robot.IK.IK_GN
        Implements this class using the Gauss-Newton method
    roboticstoolbox.robot.IK.IK_LM
        Implements this class using the Levemberg-Marquadt method

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

            \bf{q}_{k+1} = \bf{q}_k + {^0\bf{J}(\bf{q}_k)}^{-1} \bf{e}_k

        Parameters
        ----------
        ets
            The ETS representing the manipulators kinematics
        Tep
            The desired end-effector pose
        q
            The current joint coordinate vector

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
            q += np.linalg.pinv(J) @ e + qnull
        else:
            q += np.linalg.inv(J) @ e + qnull

        return E, q


class IK_LM(IKSolver):
    """
    Levemberg-Marquadt Numerical Inverse Kinematics Solver

    A class which provides functionality to perform numerical inverse kinematics (IK)
    using the Levemberg-Marquadt method. See `step` method for mathematical description.

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
        Sets the gain value for the damping matrix Wn in the `step` method. See
        notes
    method
        One of "chan", "sugihara" or "wampler". Defines which method is used
        to calculate the damping matrix Wn in the `step` method
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

    Notes
    -----
    When using the this method, the initial joint coordinates :math:`q_0`, should correspond
    to a non-singular manipulator pose, since it uses the manipulator Jacobian.

    This class supports null-space motion to assist with maximising manipulability and
    avoiding joint limits. These are enabled by setting kq and km to non-zero values.

    References
    ----------
    .. [1] J. Haviland, Jesse, and P. Corke. "Manipulator Differential Kinematics Part I:
           Kinematics, Velocity, and Applications." arXiv preprint arXiv:2207.01796 (2022).
    .. [2] J. Haviland, Jesse, and P. Corke. "Manipulator Differential Kinematics Part II:
           Acceleration and Advanced Applications." arXiv preprint arXiv:2207.01794 (2022).

    Examples
    --------
    The following example gets the `ets` of a `panda` robot object, instantiates
    the IK_LM solver class using default parameters, makes a goal pose `Tep`,
    and then solves for the joint coordinates which result in the pose `Tep`
    using the `solve` method.

    .. runblock:: pycon

        >>> import roboticstoolbox as rtb
        >>> panda = rtb.models.Panda().ets()
        >>> solver = rtb.IK_LM()
        >>> Tep = panda.fkine([0, -0.3, 0, -2.2, 0, 2, 0.7854])
        >>> solver.solve(panda, Tep)

    See Also
    --------
    roboticstoolbox.robot.IK.IKSolver
        An abstract super class for numerical IK solvers
    roboticstoolbox.robot.IK.IK_NR
        Implements this class using the Newton-Raphson method
    roboticstoolbox.robot.IK.IK_GN
        Implements this class using the Gauss-Newton method

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
            \bf{q}_{k+1} 
            &= 
            \bf{q}_k +
            \left(
                \bf{A}_k
            \right)^{-1}
            \bf{g}_k \\
            %
            \bf{A}_k
            &=
            {\bf{J}(\bf{q}_k)}^\top
            \bf{W}_e \
            {\bf{J}(\bf{q}_k)}
            +
            \bf{W}_n

        where :math:`\bf{W}_n = \text{diag}(\bf{w_n})(\bf{w_n} \in \mathbb{R}^n_{>0})` is a
        diagonal damping matrix. The damping matrix ensures that :math:`\bf{A}_k` is
        non-singular and positive definite. The performance of the LM method largely depends
        on the choice of :math:`\bf{W}_n`.

        **Chan's Method**

        Chan proposed

        .. math::
            \bf{W}_n
            =
            λ E_k \bf{1}_n

        where λ is a constant which reportedly does not have much influence on performance.
        Use the kwarg `k` to adjust the weighting term λ.

        **Sugihara's Method**

        Sugihara proposed

        .. math::
            \bf{W}_n
            =
            E_k \bf{1}_n + \text{diag}(\hat{\bf{w}}_n)

        where :math:`\hat{\bf{w}}_n \in \mathbb{R}^n`, :math:`\hat{w}_{n_i} = l^2 \sim 0.01 l^2`,
        and :math:`l` is the length of a typical link within the manipulator. We provide the
        variable `k` as a kwarg to adjust the value of :math:`w_n`.

        **Wampler's Method**

        Wampler proposed :math:`\bf{w_n}` to be a constant. This is set through the `k` kwarg.

        Parameters
        ----------
        ets
            The ETS representing the manipulators kinematics
        Tep
            The desired end-effector pose
        q
            The current joint coordinate vector

        Returns
        -------
        E
            The new error value
        q
            The new joint coordinate vector

        """
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

        q += np.linalg.inv(J.T @ self.We @ J + Wn) @ g + qnull

        return E, q


class IK_GN(IKSolver):
    """
    Gauss-Newton Numerical Inverse Kinematics Solver

    A class which provides functionality to perform numerical inverse kinematics (IK)
    using the Gauss-Newton method. See `step` method for mathematical description.

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

    Notes
    -----
    When using the this method, the initial joint coordinates :math:`q_0`, should correspond
    to a non-singular manipulator pose, since it uses the manipulator Jacobian. When the
    the problem is solvable, it converges very quickly. However, this method frequently
    fails to converge on the goal.

    This class supports null-space motion to assist with maximising manipulability and
    avoiding joint limits. These are enabled by setting kq and km to non-zero values.

    References
    ----------
    .. [1] J. Haviland, Jesse, and P. Corke. "Manipulator Differential Kinematics Part I:
           Kinematics, Velocity, and Applications." arXiv preprint arXiv:2207.01796 (2022).
    .. [2] J. Haviland, Jesse, and P. Corke. "Manipulator Differential Kinematics Part II:
           Acceleration and Advanced Applications." arXiv preprint arXiv:2207.01794 (2022).

    Examples
    --------
    The following example gets the `ets` of a `panda` robot object, instantiates
    the `IK_GN` solver class using default parameters, makes a goal pose `Tep`,
    and then solves for the joint coordinates which result in the pose `Tep`
    using the `solve` method.

    .. runblock:: pycon

        >>> import roboticstoolbox as rtb
        >>> panda = rtb.models.Panda().ets()
        >>> solver = rtb.IK_GN(pinv=True)
        >>> Tep = panda.fkine([0, -0.3, 0, -2.2, 0, 2, 0.7854])
        >>> solver.solve(panda, Tep)

    See Also
    --------
    roboticstoolbox.robot.IK.IKSolver
        An abstract super class for numerical IK solvers
    roboticstoolbox.robot.IK.IK_NR
        Implements this class using the Newton-Raphson method
    roboticstoolbox.robot.IK.IK_LM
        Implements this class using the Levemberg-Marquadt method

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

            \bf{q}_{k+1} &= \bf{q}_k +
            \left(
            {\bf{J}(\bf{q}_k)}^\top
            \bf{W}_e \
            {\bf{J}(\bf{q}_k)}
            \right)^{-1}
            \bf{g}_k \\
            \bf{g}_k &=
            {\bf{J}(\bf{q}_k)}^\top
            \bf{W}_e
            \bf{e}_k

        where :math:`\bf{J} = {^0\bf{J}}` is the base-frame manipulator Jacobian. If
        :math:`\bf{J}(\bf{q}_k)` is non-singular, and :math:`\bf{W}_e = \bf{1}_n`, then
        the above provides the pseudoinverse solution. However, if :math:`\bf{J}(\bf{q}_k)`
        is singular, the above can not be computed and the GN solution is infeasible.

        Parameters
        ----------
        ets
            The ETS representing the manipulators kinematics
        Tep
            The desired end-effector pose
        q
            The current joint coordinate vector

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
            q += np.linalg.pinv(J) @ e + qnull
        else:
            q += np.linalg.inv(J) @ e + qnull

        return E, q


# | Element    | Type    | Description                                                                                                     |
# |------------|---------|-----------------------------------------------------------------------------------------------------------------|
# | q          | ndarray | The joint coordinates of the solution (ndarray). Note that these will not be valid if failed to find a solution |
# | success    | bool    | True if a valid solution was found                                                                              |
# | iterations | int     | How many iterations were performed                                                                              |
# | searches   | int     | How many searches were performed                                                                                |
# | residual   | float   | The final error value from the cost function                                                                    |
# | reason     | str     | The reason the IK problem failed if applicable                                                                  |
