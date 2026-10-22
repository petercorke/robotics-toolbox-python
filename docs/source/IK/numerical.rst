.. _numerical-ik:

Numerical inverse kinematics
============================

The numerical solvers are methods of the :py:class:`~roboticstoolbox.ets.ETS.ETS` class, and a :py:class:`~roboticstoolbox.robot.Robot.Robot` has methods of the same names that call the ``ETS`` of the robot.  The ``ikine_XX`` methods are written in Python and the ``ik_XX`` methods in C++, where ``XX`` is the algorithm, for example ``LM``.  The C++ methods have :ref:`limits <ik-cpp-limits>` that the Python ones do not.  Each solver adds arguments of its own, so check its documentation for the complete list.

Common arguments
----------------

Besides ``Tep`` the numerical solvers share these arguments.

.. list-table:: Arguments common to the numerical solvers
    :header-rows: 1
    :widths: 18 82

    * - Argument
      - Meaning
    * - ``q0``
      - initial joint coordinates, a vector (``n``) for the first search or a matrix (``slimit``, ``n``) with one row for each search
    * - ``ilimit``
      - the number of iterations allowed in one search
    * - ``slimit``
      - the number of searches allowed before the problem is deemed unsolvable
    * - ``tol``
      - the error tolerance for a solution to count as successful
    * - ``mask``
      - six weights, for translation and rotation about X, Y and Z, of how much each Cartesian degree of freedom matters; 0 ignores it
    * - ``joint_limits``
      - if true, reject solutions that violate the joint limits

The sections below describe each argument in detail.

.. rubric:: ilimit

The ``ilimit`` specifies how many iterations are allowed within a single search. After ``ilimit`` is reached, either, a new attempt is made or the IK solution has failed depending on ``slimit``

.. rubric:: slimit

The ``slimit`` specifies how many searches are allowed before the problem is deemed unsolvable. The maximum number of iterations allowed is therefore ``ilimit`` x ``slimit``. By having ``slimit`` > 1, a global search is performed. Since finding a solution with numerical IK heavily depends on the initial choice of ``q0``, performing a global search where ``slimit`` >> 1 will provide a far greater chance of success.

.. rubric:: q0

``q0`` is the initial joint coordinate vector. If ``q0`` is 1 dimensional (, ``n``), then ``q0`` is only used for the first attempt, after which a new random valid initial joint coordinate vector will be generated. If ``q0`` is 2 dimensional (``slimit``, ``n``), then the next vector within ``q0`` will be used for the next search.

.. rubric:: tol

``tol`` sets the error tolerance before the solution is deemed successful. The error is typically set by some quadratic error function

.. math::

    E = \frac{1}{2} \vec{e}^{\top} \mat{W}_e \vec{e}

where :math:`\vec{e} \in \mathbb{R}^6` is the angle-axis error, and :math:`\mat{W}_e` assigns weights to Cartesian degrees-of-freedom

.. rubric:: mask

``mask`` is a (,6) array that sets :math:`\mat{W}_e` in error equation above. The vector has six elements that correspond to translation in X, Y and Z, and rotation about X, Y and Z respectively. The value can be 0 (for ignore) or above to assign a priority relative to other Cartesian DoF. 

For the case where the manipulator has fewer than 6 DoF the solution space has more dimensions than can be spanned by the manipulator joint coordinates.

In this case we use the ``mask`` option where the ``mask`` vector specifies the Cartesian DOF that will be ignored in reaching a solution. The number of non-zero elements must equal the number of manipulator DOF.

For example when using a 3 DOF manipulator tool orientation might be unimportant, in which case use the option ``mask=[1, 1, 1, 0, 0, 0]``.

.. rubric:: joint_limits

setting ``joint_limits = True`` will reject solutions with joint limit violations. Note that finding a solution with valid joint coordinates is likely to take longer than without.

After convergence, prismatic joint coordinates are left unchanged. Revolute
coordinates already within their limits are preserved, including limits outside
the principal interval of :math:`[-\pi, \pi]`. Other revolute coordinates are
reduced to the principal angle and, if necessary, shifted by whole turns towards
their limits. If no equivalent angle is within the limits, the solution is
rejected when ``joint_limits = True``. Setting ``joint_limits = False`` disables
this rejection; it does not change a prismatic distance into an angle.

.. rubric:: Others

There are other arguments which may be unique to the solver, so check the documentation of the solver you wish to use for a complete list and explanation of arguments.


.. _ik-cpp-limits:

C++ Solvers
-----------

The ``ik_XX`` methods are written in high performance C++ and wrapped in Python.  They are extraordinarily fast, typically about 10 µs, but are hard to extend or modify.  There are ``ik_LM``, ``ik_GN`` and ``ik_NR``, but no ``ik_QP``.

These methods have been written purely for speed so they do not contain the niceties of the Python alternative.  They handle a single pose only, ``Tep`` must not be a trajectory, and they give no failure ``reason``, which is always empty.  The lengths of ``q0`` (``n``) and of ``mask`` (6) are checked, and a wrong length raises ``ValueError``, but the other arguments are passed to the C++ code as they are, so it is important that you understand each of the parameters and that they are of the correct type.  To solve for a trajectory use the Python solvers.

.. rubric:: When the C++ solvers cannot be used

The ``ik_XX`` methods need the compiled C++ extension and raise ``RuntimeError`` if it cannot be used.  This is so:

- in the pure-Python wheel and in the browser under Pyodide/JupyterLite, which have no compiled code
- if the sequence has a **SymPy parameter**, a symbol or expression as the parameter of a transform, even one with a numeric value such as ``sympy.pi/2``
- if the sequence has a joint with a nonzero :attr:`~roboticstoolbox.ets.ET.ET.offset`, which is what merging a constant into a joint produces

Everything else in the Toolbox, such as ``fkine`` and ``jacob0``, falls back to Python in these cases, but the C++ solvers do not.  Use the ``ikine_XX`` methods, which work in all of them as long as every parameter has a numeric value, are not much harder to call, and take the same arguments with the exceptions noted in their documentation.

.. rubric:: ETS C++ IK Methods

.. autosummary::
    :nosignatures:

    ~roboticstoolbox.ETS.ik_LM
    ~roboticstoolbox.ETS.ik_GN
    ~roboticstoolbox.ETS.ik_NR

.. rubric:: Robot C++ IK Methods

.. autosummary::
    :toctree: stubs

    ~roboticstoolbox.robot.Robot.Robot.ik_LM
    ~roboticstoolbox.robot.Robot.Robot.ik_GN
    ~roboticstoolbox.robot.Robot.Robot.ik_NR

In the following example, we create a :py:class:`~roboticstoolbox.models.URDF.Panda` robot and use one of the fast IK solvers available within the :py:class:`~roboticstoolbox.robot.Robot.Robot` class.

.. runblock:: pycon

    >>> import roboticstoolbox as rtb
    >>> panda = rtb.models.Panda()  # Make a Panda robot
    >>> Tep = panda.fkine([0, -0.3, 0, -2.2, 0, 2, 0.7854])  # Make a goal pose
    >>> panda.ik_LM(Tep)  # Solve the IK problem

In the following example, we create a :py:class:`~roboticstoolbox.models.URDF.Panda` robot and then get the :py:class:`~roboticstoolbox.ets.ETS.ETS` representation. Subsequently, we use one of the fast IK solvers available within the :py:class:`~roboticstoolbox.ets.ETS.ETS` class.

.. runblock:: pycon

    >>> import roboticstoolbox as rtb
    >>> panda = rtb.models.Panda()  # Make a Panda robot
    >>> ets = panda.ets()  # Get the ETS
    >>> Tep = ets.fkine([0, -0.3, 0, -2.2, 0, 2, 0.7854])  # Make a goal pose
    >>> ets.ik_LM(Tep)  # Solve the IK problem





Python Solvers
--------------

The ``ikine_XX`` methods are built on Python classes, one for each algorithm, which extend the abstract base class :py:class:`~roboticstoolbox.robot.IK.IKSolver`, and the :py:meth:`~roboticstoolbox.robot.IK.IKSolver.solve` method of each returns an :py:class:`~roboticstoolbox.robot.IK.IKSolution`.  They are slower than the C++ solvers, typically about 1 ms, but are easy to extend and modify.  They also handle trajectories and give a failure ``reason``.

.. rubric:: The Abstract Base Class 

.. toctree::
    :maxdepth: 1
    :hidden:

    iksolver

The :py:class:`~roboticstoolbox.robot.IK.IKSolver` provides basic functionality for performing numerical IK. Subclasses inherit from this class and must implement the :py:meth:`~roboticstoolbox.robot.IK.IKSolver.solve` method. Additionally a subclass can redefine any other methods necessary such as :py:meth:`~roboticstoolbox.robot.IK.IKSolver.error` to provide a custom error function.

.. rubric:: The Solution DataClass

.. toctree::
    :maxdepth: 1
    :hidden:

    iksolution

The :py:class:`~roboticstoolbox.robot.IK.IKSolution` is a :py:func:`dataclasses.dataclass`, described at the top of this page.

.. rubric:: Solving for a trajectory

If ``Tep`` is a trajectory of N poses, each pose is solved independently, starting from the same ``q0`` (and the same random restarts) for every pose.  Consecutive solutions are therefore not guaranteed to be on the same IK branch.  The members of the returned :py:class:`~roboticstoolbox.robot.IK.IKSolution` are:

* ``q`` has shape (N, n), one row per pose
* ``success`` is True only if every pose was solved
* ``iterations`` and ``searches`` are summed over all poses
* ``residual`` is the maximum over all poses, the worst case
* ``reason`` is the reason given by the last pose that failed, if any

.. versionchanged:: 1.5.0
    For a trajectory, ``residual`` is the maximum over the poses, it was the minimum.

.. rubric:: The Implemented IK Solvers

These solvers can be identified as classes with a name starting with ``IK_``.

==================  ============================  =============================================================
Class               Method                        Notes
==================  ============================  =============================================================
:doc:`ik_lm`        Levenberg-Marquardt           the solver used in the examples below
:doc:`ik_gn`        Gauss-Newton                  set ``pinv=True`` for a robot with more than 6 joints
:doc:`ik_nr`        Newton-Raphson                set ``pinv=True`` for a robot with more than 6 joints
:doc:`ik_qp`        quadratic programming         needs the ``qp`` extra
==================  ============================  =============================================================

.. toctree::
    :maxdepth: 1
    :hidden:

    ik_lm
    ik_qp
    ik_gn
    ik_nr

.. rubric:: Examples

In the following example, we create an IK Solver class and pass an :py:class:`~roboticstoolbox.ets.ETS.ETS` to it to solve the problem. This style may be preferable to experiments where you wish to compare the same solver on different robots.

.. runblock:: pycon

    >>> import roboticstoolbox as rtb
    >>> panda = rtb.models.Panda()  # Make a Panda robot
    >>> ets = panda.ets()  # Get the ETS of the Panda
    >>> solver = rtb.IK_LM()  # Make an IK solver
    >>> Tep = panda.fkine([0, -0.3, 0, -2.2, 0, 2, 0.7854])  # Make a goal pose
    >>> solver.solve(ets, Tep)  # Solve the IK problem



Additionally, these class-based solvers have been implemented as methods within the :py:class:`~roboticstoolbox.ets.ETS.ETS` and :py:class:`~roboticstoolbox.robot.Robot.Robot` classes. The method names start with ``ikine_``.


.. rubric:: ETS Python IK Methods


.. autosummary::
    :nosignatures:

    ~roboticstoolbox.ETS.ikine_LM
    ~roboticstoolbox.ETS.ikine_QP
    ~roboticstoolbox.ETS.ikine_GN
    ~roboticstoolbox.ETS.ikine_NR


.. rubric:: Robot Python IK Methods

.. autosummary::
    :toctree: stubs
    
    ~roboticstoolbox.robot.Robot.Robot.ikine_LM
    ~roboticstoolbox.robot.Robot.Robot.ikine_QP
    ~roboticstoolbox.robot.Robot.Robot.ikine_GN
    ~roboticstoolbox.robot.Robot.Robot.ikine_NR


.. rubric:: Examples

In the following example, we create a :py:class:`~roboticstoolbox.models.URDF.Panda` robot and use one of the IK solvers available within the :py:class:`~roboticstoolbox.robot.Robot.Robot` class. This style is far more convenient than the above example.

.. runblock:: pycon

    >>> import roboticstoolbox as rtb
    >>> panda = rtb.models.Panda()  # Make a Panda robot
    >>> Tep = panda.fkine([0, -0.3, 0, -2.2, 0, 2, 0.7854])  # Make a goal pose
    >>> panda.ikine_LM(Tep)  # Solve the IK problem

In the following example, we create a :py:class:`~roboticstoolbox.models.URDF.Panda` robot and then get the :py:class:`~roboticstoolbox.ets.ETS.ETS` representation. Subsequently, we use one of the IK solvers available within the :py:class:`~roboticstoolbox.ets.ETS.ETS` class.

.. runblock:: pycon

    >>> import roboticstoolbox as rtb
    >>> panda = rtb.models.Panda()  # Make a Panda robot
    >>> ets = panda.ets()  # Get the ETS
    >>> Tep = ets.fkine([0, -0.3, 0, -2.2, 0, 2, 0.7854])  # Make a goal pose
    >>> ets.ikine_LM(Tep)  # Solve the IK problem
