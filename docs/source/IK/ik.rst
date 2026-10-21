.. _IK:


Inverse Kinematics
==================

Inverse kinematics (IK) finds the joint coordinates :math:`\vec{q}` that put the end-effector at a desired pose :math:`\mat{T}`.  It is the reverse of forward kinematics, and is harder: there may be no solution, one, several or infinitely many, and which one you get can matter.

Every IK method in the Toolbox takes the same input and returns the same result.

.. rubric:: The desired pose, ``Tep``

``Tep`` is the desired end-effector pose in the robot's base frame, an :py:class:`~spatialmath.pose3d.SE3` or a 4x4 NumPy array (a homogeneous transformation).  The name follows the Toolbox convention: **T** is an SE(3) pose, *e* is the end-effector, *p* is "prime", the desired value, and with no frame to the left of the **T** the world or base frame is implied.  Most methods also accept a **trajectory** of poses, an ``SE3`` object containing N poses or an array with shape (N, 4, 4), that is one pose for each index of the first axis, not (4, 4, N) as in MATLAB.

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

.. versionchanged:: 1.4.5
    Iterating over the six members was removed, use the attributes or ``sol.astuple()``.

The Toolbox has *three ways* to find a solution, shown in the figure.  **Numerical** solvers start from a guess for :math:`\vec{q}` and iterate until the pose is reached.  They work for any robot, and are written in Python, or in C++ for speed.  **Analytic** solvers (see :doc:`analytic`) derive the equations of the inverse kinematics once, symbolically, and then solve them directly for any pose, with no guess and no iteration, giving all the solutions.  They work for six-axis arms, and can write the solution out as source code with no dependency on the Toolbox.

.. graphviz:: ik_overview.dot
    :alt: The ways to solve inverse kinematics in the Toolbox: analytic, numerical in Python, and numerical in C++.
    :caption: The three mechanisms (click to enlarge).  The numerical methods ``ikine_XX`` and ``ik_XX`` are methods of the :py:class:`~roboticstoolbox.ets.ETS.ETS` class, and a robot has methods of the same names that call its ETS.  The analytic methods belong *only* to :py:class:`~roboticstoolbox.robot.DHRobot.DHRobot`.

The numerical algorithms have names that are shown ending in ``_XX``, where ``XX`` identifies a particular numerical algorithm:

=======  =============================================
``XX``   Algorithm
=======  =============================================
``LM``   Levenberg-Marquardt
``GN``   Gauss-Newton
``NR``   Newton-Raphson
``QP``   quadratic programming (not for C++ solvers)
=======  =============================================

For a **tutorial** on numerical IK, see `here <https://bit.ly/3ak5GDi>`_.

.. list-table:: Which one to use
    :header-rows: 1
    :widths: 24 25 25 26

    * -
      - :doc:`Analytic <analytic>`
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
      - no, one pose only
    * - Time for one pose
      - about 0.2 ms, or microseconds as generated C
      - about 1 ms
      - about 10 µs
    * - Methods
      - ``ikine_analytic``, ``analytic_ik`` of a DH robot
      - ``ikine_XX`` of an ``ETS`` and of a robot
      - ``ik_XX`` of an ``ETS`` and of a robot
    * - Extra to install
      - ``ik``, see :ref:`analytic-ik`
      - none
      - none
    * - In the browser (JupyterLite)
      - yes
      - yes
      - no, it needs the compiled extension
    * - SymPy parameters, joint offsets
      - not applicable
      - yes, if every parameter has a numeric value
      - no, see :ref:`ik-cpp-limits`

The times are indicative, for a six-axis arm on a recent laptop (an Apple M4).  The script ``examples/benchmark_ik.py`` in the repository measures the numerical solvers on your own machine.  Where an analytic solution exists it is the right choice: it is exact, finds every configuration, and can be generated as code for a controller.  Otherwise use a numerical solver, and the C++ one if you only need a single pose quickly and its :ref:`limits <ik-cpp-limits>` do not matter to you.

.. rubric:: Analytic

The analytic solvers are methods of a :py:class:`~roboticstoolbox.robot.DHRobot.DHRobot`.  ``analytic_ik()`` makes a solver for the robot, which gives every solution for a pose and can write the solution out as source code, and ``ikine_analytic()`` is the one-step counterpart of the ``ikine_XX`` methods.  They need the ``ik`` extra, which installs SymPy.  See :doc:`analytic` for how to use them, and the notebook `ik-symbolic.ipynb <https://github.com/petercorke/robotics-toolbox-python/blob/main/docs/notebooks/ik-symbolic.ipynb>`_ for worked examples with a Puma 560, the Stanford arm and a Kinova Jaco, including trajectories and code generation.

.. rubric:: Numerical

The numerical solvers start from a guess and iterate.  They work for any robot and are written in C++ and in Python, but the C++ ones need the compiled extension and have :ref:`limits <ik-cpp-limits>`.  See :doc:`numerical` for the arguments that they share, and for the C++ and Python solvers, the classes that implement them and examples.

.. toctree::
    :maxdepth: 1
    :hidden:

    analytic
    numerical
