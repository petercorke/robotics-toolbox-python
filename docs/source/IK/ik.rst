.. _IK:


Inverse Kinematics
==================

Inverse kinematics (IK) finds the joint coordinates :math:`\vec{q}` that put the end-effector at a desired pose :math:`\mat{T}`.  It is the reverse of forward kinematics, and is harder: there may be no solution, one, several or infinitely many, and which one you get can matter.

Every IK method in the Toolbox takes the same input and returns the same result.

.. rubric:: The desired pose, ``Tep``

``Tep`` is the desired end-effector pose in the robot's base frame, an :py:class:`~spatialmath.pose3d.SE3` (a homogeneous transformation, a 4x4 matrix).  The name follows the Toolbox convention: **T** is an SE(3) pose, *e* is the end-effector, *p* is "prime", the desired value, and with no frame to the left of the **T** the world or base frame is implied.  Most methods also accept a **trajectory** of poses, an ``SE3`` object containing N poses or an array with shape (N, 4, 4).

.. rubric:: The result, ``IKSolution``

The result is an :py:class:`~roboticstoolbox.robot.IK.IKSolution`.

==============   =====================================================================================================
Member           Meaning
==============   =====================================================================================================
``q``            the joint coordinates, shape (n,) for a pose or (N, n) for a trajectory; not valid if there was no solution
``success``      true if a valid solution was found
``iterations``   how many iterations were performed, zero for an analytic solution
``searches``     how many searches were performed, zero for an analytic solution
``residual``     the final error of the solution
``reason``       why it failed, if it did
==============   =====================================================================================================

An ``IKSolution`` is a sequence of the rows of ``q``, so ``len(sol)`` is the number of poses and ``for q in sol`` visits them, and ``bool(sol)`` is ``success``.

The Toolbox has three ways to find a solution, shown in the figure.  **Numerical** solvers start from a guess for :math:`\vec{q}` and iterate until the pose is reached.  They work for any robot, and are written in Python, or in C++ for speed.  **Analytic** solvers derive the equations of the inverse kinematics once, symbolically, and then solve them directly for any pose, with no guess and no iteration, giving all the solutions.  They work for six-axis arms, and can write the solution out as source code with no dependency on the Toolbox.

.. graphviz:: ik_overview.dot
    :alt: The ways to solve inverse kinematics in the Toolbox: analytic, numerical in Python, and numerical in C++.
    :caption: The three mechanisms.  The numerical methods ``ikine_XX`` and ``ik_XX`` are methods of the :py:class:`~roboticstoolbox.ets.ETS.ETS` class, and a robot has methods of the same names that call its ETS.  The analytic methods belong to :py:class:`~roboticstoolbox.robot.DHRobot.DHRobot`.

For a **tutorial** on numerical IK, see `here <https://bit.ly/3ak5GDi>`_.

.. list-table:: Which one to use
    :header-rows: 1
    :widths: 24 25 25 26

    * -
      - Analytic
      - Numerical, Python
      - Numerical, C++
    * - Robots
      - six-joint arms with standard DH parameters
      - any, from an :py:class:`~roboticstoolbox.ets.ETS.ETS`
      - any, from an :py:class:`~roboticstoolbox.ets.ETS.ETS`
    * - Solutions
      - all of them, or the one nearest ``q0``
      - one, which depends on ``q0``
      - one, which depends on ``q0``
    * - Initial guess
      - not needed
      - needed
      - needed
    * - Trajectory
      - yes
      - yes
      - one pose only
    * - Time for one pose
      - about 0.2 ms, or microseconds as generated C
      - about 1 ms
      - 30 to 90 µs
    * - Methods
      - ``ikine_analytic``, ``analytic_ik`` of a DH robot
      - ``ikine_XX`` of an ``ETS`` and of a robot
      - ``ik_XX`` of an ``ETS`` and of a robot
    * - Extra to install
      - ``ik``, see :ref:`analytic-ik`
      - none
      - none

The times are indicative, for a laptop and a six-axis arm.  Where an analytic solution exists it is the right choice: it is exact, finds every configuration, and can be generated as code for a controller.  Otherwise use a numerical solver, and the C++ one if you only need a single pose quickly.

.. toctree::
    :maxdepth: 1

    analytic


Numerical solvers
-----------------

The numerical solvers are methods of the :py:class:`~roboticstoolbox.ets.ETS.ETS` class, and a :py:class:`~roboticstoolbox.robot.Robot.Robot` has methods of the same names that call the ``ETS`` of the robot.  The ``ikine_XX`` methods are written in Python and the ``ik_XX`` methods in C++, where ``XX`` is the algorithm, for example ``LM``.  Besides ``Tep`` they share these arguments, and each solver adds some of its own, so check its documentation for the complete list.

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


C++ Solvers
-----------

These solvers are written in high performance C++ and wrapped in Python methods. The methods are made available within the :py:class:`~roboticstoolbox.ets.ETS.ETS` and :py:class:`~roboticstoolbox.robot.Robot.Robot` classes. Being written in C++, these solvers are extraordinarily fast and typically take 30 to 90 µs. However, these solvers are hard to extend or modify.

These methods have been written purely for speed so they do not contain the niceties of the Python alternative, such as a failure ``reason`` or trajectory handling. The lengths of ``q0`` (``n``) and of ``mask`` (6) are checked, and a wrong length raises ``ValueError``, but the other arguments are passed to the C++ code as they are, so it is important that you understand each of the parameters and that they are of the correct type.

The C++ solvers return an :py:class:`~roboticstoolbox.robot.IK.IKSolution` with the following members (``reason`` is always empty, these solvers do not produce a failure reason string):

==============   =========   =====================================================================================================
Element          Type        Description                                                                                                    
==============   =========   =====================================================================================================
``q``            `ndarray`   The joint coordinates of the solution. Note that these will not be valid if failed to find a solution
``success``      `bool`      True if a valid solution was found                                                                             
``iterations``   `int`       How many iterations were performed                                                                             
``searches``     `int`       How many searches were performed                                                                               
``residual``     `float`     The final error value from the cost function                                                                   
==============   =========   =====================================================================================================

The C++ solvers can be identified as methods which start with ``ik_``.

These solvers handle a single pose only, ``Tep`` must not be a trajectory.  To solve for a trajectory use the Python solvers.

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
    >>> # Make a Panda robot
    >>> panda = rtb.models.Panda()
    >>> # Make a goal pose
    >>> Tep = panda.fkine([0, -0.3, 0, -2.2, 0, 2, 0.7854])
    >>> # Solve the IK problem
    >>> panda.ik_LM(Tep)

In the following example, we create a :py:class:`~roboticstoolbox.models.URDF.Panda` robot and then get the :py:class:`~roboticstoolbox.ets.ETS.ETS` representation. Subsequently, we use one of the fast IK solvers available within the :py:class:`~roboticstoolbox.ets.ETS.ETS` class.

.. runblock:: pycon

    >>> import roboticstoolbox as rtb
    >>> # Make a Panda robot
    >>> panda = rtb.models.Panda()
    >>> # Get the ETS
    >>> ets = panda.ets()
    >>> # Make a goal pose
    >>> Tep = ets.fkine([0, -0.3, 0, -2.2, 0, 2, 0.7854])
    >>> # Solve the IK problem
    >>> ets.ik_LM(Tep)





Python Solvers
--------------

These solvers are Python classes which extend the abstract base class :py:class:`~roboticstoolbox.robot.IK.IKSolver` and the :py:meth:`~roboticstoolbox.robot.IK.IKSolver.solve` method returns an :py:class:`~roboticstoolbox.robot.IK.IKSolution`. These solvers are slow and will typically take 100 - 1000 ms.  However, these solvers are easy to extend and modify.

.. rubric:: The Abstract Base Class 

.. toctree::
    :maxdepth: 1

    iksolver

The :py:class:`~roboticstoolbox.robot.IK.IKSolver` provides basic functionality for performing numerical IK. Subclasses inherit from this class and must implement the :py:meth:`~roboticstoolbox.robot.IK.IKSolver.solve` method. Additionally a subclass can redefine any other methods necessary such as :py:meth:`~roboticstoolbox.robot.IK.IKSolver.error` to provide a custom error function.

.. rubric:: The Solution DataClass

.. toctree::
    :maxdepth: 1

    iksolution

The :py:class:`~roboticstoolbox.robot.IK.IKSolution` is a :py:func:`dataclasses.dataclass` instance with the following members.  It is also a sequence of the rows of ``q``, one per pose: ``len(sol)`` is the number of poses, ``for q in sol`` and ``sol[i]`` give the joint coordinates of each pose, and ``bool(sol)`` is ``success``.  Iterating over the six members was removed in 1.4.5, use the attributes or ``sol.astuple()``.

==============   =========   =====================================================================================================
Element          Type        Description                                                                                                    
==============   =========   =====================================================================================================
``q``            `ndarray`   The joint coordinates of the solution. Note that these will not be valid if failed to find a solution
``success``      `bool`      True if a valid solution was found                                                                             
``iterations``   `int`       How many iterations were performed                                                                             
``searches``     `int`       How many searches were performed                                                                               
``residual``     `float`     The final error value from the cost function                                                                   
``reason``       `str`       The reason the IK problem failed if applicable   
==============   =========   =====================================================================================================

.. rubric:: Solving for a trajectory

If ``Tep`` is a trajectory of N poses, each pose is solved independently, starting from the same ``q0`` (and the same random restarts) for every pose.  Consecutive solutions are therefore not guaranteed to be on the same IK branch.  The members of the returned :py:class:`~roboticstoolbox.robot.IK.IKSolution` are:

* ``q`` has shape (N, n), one row per pose
* ``success`` is True only if every pose was solved
* ``iterations`` and ``searches`` are summed over all poses
* ``residual`` is the maximum over all poses, the worst case
* ``reason`` is the reason given by the last pose that failed, if any

.. versionchanged:: 1.4.5
    For a trajectory, ``residual`` is the maximum over the poses, it was the minimum.

.. rubric:: The Implemented IK Solvers

These solvers can be identified as classes with a name starting with ``IK_``.

.. toctree::
    :maxdepth: 1

    ik_lm
    ik_qp
    ik_gn
    ik_nr

.. rubric:: Example 

In the following example, we create an IK Solver class and pass an :py:class:`~roboticstoolbox.ets.ETS.ETS` to it to solve the problem. This style may be preferable to experiments where you wish to compare the same solver on different robots.

.. runblock:: pycon

    >>> import roboticstoolbox as rtb
    >>> # Make a Panda robot
    >>> panda = rtb.models.Panda()
    >>> # Get the ETS of the Panda
    >>> ets = panda.ets()
    >>> # Make an IK solver
    >>> solver = rtb.IK_LM()
    >>> # Make a goal pose
    >>> Tep = panda.fkine([0, -0.3, 0, -2.2, 0, 2, 0.7854])
    >>> # Solve the IK problem
    >>> solver.solve(ets, Tep)



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


.. rubric:: Example 

In the following example, we create a :py:class:`~roboticstoolbox.models.URDF.Panda` robot and use one of the IK solvers available within the :py:class:`~roboticstoolbox.robot.Robot.Robot` class. This style is far more convenient than the above example.

.. runblock:: pycon

    >>> import roboticstoolbox as rtb
    >>> # Make a Panda robot
    >>> panda = rtb.models.Panda()
    >>> # Make a goal pose
    >>> Tep = panda.fkine([0, -0.3, 0, -2.2, 0, 2, 0.7854])
    >>> # Solve the IK problem
    >>> panda.ikine_LM(Tep)

In the following example, we create a :py:class:`~roboticstoolbox.models.URDF.Panda` robot and then get the :py:class:`~roboticstoolbox.ets.ETS.ETS` representation. Subsequently, we use one of the IK solvers available within the :py:class:`~roboticstoolbox.ets.ETS.ETS` class.

.. runblock:: pycon

    >>> import roboticstoolbox as rtb
    >>> # Make a Panda robot
    >>> panda = rtb.models.Panda()
    >>> # Get the ETS
    >>> ets = panda.ets()
    >>> # Make a goal pose
    >>> Tep = ets.fkine([0, -0.3, 0, -2.2, 0, 2, 0.7854])
    >>> # Solve the IK problem
    >>> ets.ikine_LM(Tep)
