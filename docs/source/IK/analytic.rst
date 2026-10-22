.. _analytic-ik:

Analytic inverse kinematics
===========================

For a six-axis arm described by standard Denavit-Hartenberg parameters the inverse kinematics can be solved without iteration.  The equations are derived symbolically, once, for the particular robot.  After that a pose is solved by evaluating them, which gives **every** joint configuration that reaches it, or tells you that the pose cannot be reached.

The same solution can be written out as source code in Python, C, C++ or Rust, for use on a robot controller or in another program.  The generated code is released under the MIT licence, like the Toolbox, and needs neither the Toolbox nor SymPy.

.. note::

    Deriving the equations needs SymPy, which is not installed with the Toolbox by default::

        pip install roboticstoolbox-python[ik]

    The ``ik`` extra also installs ``python-flint``, which makes the derivation much faster.  Without it a warning is issued and SymPy is used instead, which gives the same results.  It is not part of the ``all`` extra.

.. versionadded:: 1.5.0

The notebook `ik-symbolic.ipynb <https://github.com/petercorke/robotics-toolbox-python/blob/main/docs/notebooks/ik-symbolic.ipynb>`_ works through the examples on this page, and more.

Setting up
----------

The solver is made by the robot.  The first call for a robot derives the equations, which takes a few seconds, and the solver is kept so that later calls are immediate.

.. runblock:: pycon

    >>> import numpy as np
    >>> import roboticstoolbox as rtb
    >>> puma = rtb.models.DH.Puma560()
    >>> ik = puma.analytic_ik()
    >>> print(ik)

The method is chosen from the structure of the arm, and each method suits only its own kind of arm:

=====================================================  ====================================================  ====================
Arm                                                    Method                                                Solver class
=====================================================  ====================================================  ====================
three consecutive joint axes meet at a point (a        closed form, derived symbolically                     ``ClosedFormIK``
spherical wrist) or are parallel
neither, for example the Kinova Jaco                   general 6R method of Raghavan and Roth [Raghavan93]_  ``General6RIK``
=====================================================  ====================================================  ====================

Solving
-------

To pose a problem we need a pose that can be reached, so we take the forward kinematics of a joint configuration that is within the joint limits.

.. runblock:: pycon

    >>> import numpy as np
    >>> import roboticstoolbox as rtb
    >>> puma = rtb.models.DH.Puma560()
    >>> q = np.array([0.2, 0.9, -0.8, 0.3, -0.5, 0.4])
    >>> T = puma.fkine(q)
    >>> sol = puma.analytic_ik().solve(T)
    >>> len(sol)
    >>> np.set_printoptions(precision=3, suppress=True)
    >>> sol.q

:meth:`~roboticstoolbox.robot.symbolic_ik_rr.AnalyticIK.solve` gives all the solutions as the rows of ``q`` in an :py:class:`~roboticstoolbox.robot.IK.IKSolution`, the same result type as the numerical solvers.  Each row is a different arm configuration, and every one reaches the pose.

Usually you want one solution.  The ``ikine_analytic`` method of the robot is the counterpart of ``ikine_LM`` and the other numerical methods.  It returns the solution nearest to ``q0``, and ``weights`` changes what nearest means: a weight of zero ignores a joint.

.. runblock:: pycon

    >>> import numpy as np
    >>> import roboticstoolbox as rtb
    >>> puma = rtb.models.DH.Puma560()
    >>> q = np.array([0.2, 0.9, -0.8, 0.3, -0.5, 0.4])
    >>> T = puma.fkine(q)
    >>> sol = puma.ikine_analytic(T, q0=q + 0.3)
    >>> sol

``iterations`` and ``searches`` are zero, since nothing is searched, and ``residual`` is the error of the forward kinematics of the solution.

A pose that can't be reached is not an exception.  The result is a failed ``IKSolution`` that says why, and ``q`` is NaN.

.. runblock:: pycon

    >>> import roboticstoolbox as rtb
    >>> from spatialmath import SE3
    >>> puma = rtb.models.DH.Puma560()
    >>> puma.ikine_analytic(SE3(5, 5, 5))

Joint limits
------------

Solutions that violate the joint limits are discarded, as the numerical solvers do with ``joint_limits=True``.  Revolute angles are first shifted by whole turns, since :math:`q` and :math:`q \pm 2\pi` are the same configuration, so a solution is rejected only if no equivalent angle is within the limits.  Use ``joint_limits=False`` to get everything that the equations allow.

Trajectories
------------

A trajectory is an :py:class:`~spatialmath.pose3d.SE3` with N poses, or an array of shape (N, 4, 4), and the solution has one row of ``q`` for each pose.  The first pose is solved nearest to ``q0`` and each later pose nearest to the solution of the one before, so the path stays on one branch for as long as the arm can.  This differs from the numerical solvers, which solve every pose from the same ``q0``.  If a pose can't be solved ``success`` is false, the row for that pose is NaN, and ``reason`` names the first such pose.

.. runblock:: pycon

    >>> import numpy as np
    >>> import roboticstoolbox as rtb
    >>> puma = rtb.models.DH.Puma560()
    >>> qa = np.array([0.2, 0.9, -0.8, 0.3, -0.5, 0.4])
    >>> qb = qa + [0.5, -0.3, 0.4, 0.6, -0.2, 0.5]
    >>> Ts = rtb.ctraj(puma.fkine(qa), puma.fkine(qb), 50)
    >>> sol = puma.ikine_analytic(Ts, q0=qa)
    >>> sol.success, sol.q.shape

Generating code
---------------

``generate`` writes the solution as source code.  The first argument is the language and the second is the name of the function.  Every file starts with a header that says where it came from, and records the parameters of the robot.

=========  ===============================  =========================================================
Language   Closed form and general 6R       Needs
=========  ===============================  =========================================================
Python     both                             NumPy for the general method, and SciPy if installed
C          both                             ``math.h`` and ``string.h`` only
C++        both                             the C++17 standard library only
Rust       closed form only                 the standard library only
=========  ===============================  =========================================================

.. runblock:: pycon

    >>> import roboticstoolbox as rtb
    >>> puma = rtb.models.DH.Puma560()
    >>> source = puma.analytic_ik().generate("c", "puma_ik")
    >>> lines = source.splitlines()
    >>> len(lines)
    >>> print("\n".join(lines[:9]))

The generated C is written with embedded use in mind: no dynamic memory allocation, no input or output, no recursion, no variable-length arrays, and working arrays of fixed size, with the solutions written to an array that the caller provides.  It compiles without warnings as C99 with ``-pedantic -Wall -Wextra -Wvla``.  It has **not** been checked against a coding standard such as MISRA C, see `#839 <https://github.com/petercorke/robotics-toolbox-python/issues/839>`_.  The C++ and Rust code return their solutions in a ``Vec`` or ``std::vector``, which allocates.

The general 6R method needs to find the roots of a polynomial matrix, so its C and C++ code includes a small linear algebra kernel of about 300 lines: LU, Householder QR, a one-sided Jacobi SVD and an eigenvalue solver, in plain C99.  It was written for this project from the standard algorithms, with the references in the source, and is tested against NumPy, but it has had far less use than LAPACK.

Limitations
-----------

* six joints and standard DH parameters, not modified DH
* an arm that needs the general method and has a prismatic joint is not supported, and ``analytic_ik`` raises ``NotImplementedError``
* the general method is less well conditioned than the closed form, it is slower, and near a singularity its accuracy falls; every solution is checked against the forward kinematics before it is returned
* Rust code is generated for the closed form only

For a robot that this can't handle, the numerical methods :meth:`~roboticstoolbox.robot.Robot.Robot.ikine_LM` and its relatives still work.

The methods
-----------

.. autosummary::
    :toctree: stubs
    :nosignatures:

    ~roboticstoolbox.robot.DHRobot.DHRobot.analytic_ik
    ~roboticstoolbox.robot.DHRobot.DHRobot.ikine_analytic

The solver classes
------------------

.. autoclass:: roboticstoolbox.robot.symbolic_ik_rr.AnalyticIK
    :members: solve, ikine, generate, validate
    :member-order: bysource

.. autoclass:: roboticstoolbox.robot.symbolic_ik_rr.ClosedFormIK
    :no-members:
    :show-inheritance:

.. autoclass:: roboticstoolbox.robot.symbolic_ik_rr.General6RIK
    :no-members:
    :show-inheritance:

.. [Raghavan93] M. Raghavan and B. Roth, "Inverse kinematics of the general 6R manipulator and related linkages," *Journal of Mechanical Design*, vol. 115, no. 3, pp. 502–508, September 1993.
