.. _IK:


Inverse Kinematics
==================


The Robotics Toolbox supports an extensive set of numerical inverse kinematics (IK) tools and we will demonstrate the different ways these IK tools can be interacted with in this document.

For a **tutorial** on numerical IK, see `here <https://bit.ly/3ak5GDi>`_.

Within the Toolbox, we have two sets of solvers: solvers written in C++ and solvers written in Python. However, within all of our solvers there are several common arguments:

.. rubric:: Tep

``Tep`` represent the desired end-effector pose. 

A note on the semantics of the above variable:

* **T** represents an SE(3) (a homogeneous tranformation matrix in 3 dimensions, a 4x4 matrix) 
* *e* is short for end-effector referring to the end of the kinematic chain
* *p* is short for prime or desired
* Since there is no letter to the left of the **T**, the world or base reference frame is implied

Therefore, ``Tep`` refers to the desired end-effector pose in the base robot frame represented as an SE(3).

.. rubric:: ilimit

The ``ilimit`` specifies how many iterations are allowed within a single search. After ``ilimit`` is reached, either, a new attempt is made or the IK solution has failed depending on ``slimit``

.. rubric:: slimit

The ``slimit`` specifies how many searches are allowed before the problem is deemed unsolvable. The maximum number of iterations allowed is therefore ``ilimit`` x ``slimit``. By having ``slimit`` > 1, a global search is performed. Since finding a solution with numerical IK heavily depends on the initial choice of ``q0``, performing a global search where ``slimit`` >> 1 will provide a far greater chance of success.

.. rubric:: q0

``q0`` is the inital joint coordinate vector. If ``q0`` is 1 dimensional (, ``n``), then ``q0`` is only used for the first attempt, after which a new random valid initial joint coordinate vector will be generated. If ``q0`` is 2 dimensional (``slimit``, ``n``), then the next vector within ``q0`` will be used for the next search.

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

.. rubric:: Others

There are other arguments which may be unique to the solver, so check the documentation of the solver you wish to use for a complete list and explanation of arguments.

C++ Solvers
-----------

These solvers are written in high performance C++ and wrapped in Python methods. The methods are made available within the :py:class:`~roboticstoolbox.robot.ETS.ETS` and :py:class:`~roboticstoolbox.robot.Robot.Robot` classes. Being written in C++, these solvers are extraordinarily fast and typically take 30 to 90 µs. However, these solvers are hard to extend or modify.

These methods have been written purely for speed so they do not contain the niceties of the Python alternative. For example, if you give the incorrect length for the ``q0`` vector, you could end up with a ``seg-fault`` or other undetermined behaviour. Therefore, when using these methods it is very important that you understand each of the parameters and the parameters passed are of the correct type and length.

The C++ solvers return a tuple with the following members:

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

.. rubric:: ETS C++ IK Methods

.. autosummary::
    :toctree: stubs
    
    ~roboticstoolbox.robot.ETS.ETS.ik_LM
    ~roboticstoolbox.robot.ETS.ETS.ik_GN
    ~roboticstoolbox.robot.ETS.ETS.ik_NR

.. rubric:: Robot C++ IK Methods

.. autosummary::
    :toctree: stubs

    ~roboticstoolbox.robot.Robot.Robot.ik_LM
    ~roboticstoolbox.robot.Robot.Robot.ik_GN
    ~roboticstoolbox.robot.Robot.Robot.ik_NR

In the following example, we create a :py:class:`~roboticstoolbox.models.URDF.Panda` robot and one of the fast IK solvers available within the :py:class:`~roboticstoolbox.robot.Robot.Robot` class.

.. runblock:: pycon

    >>> import roboticstoolbox as rtb
    >>> # Make a Panda robot
    >>> panda = rtb.models.Panda()
    >>> # Make a goal pose
    >>> Tep = panda.fkine([0, -0.3, 0, -2.2, 0, 2, 0.7854])
    >>> # Solve the IK problem
    >>> panda.ik_LM(Tep)

In the following example, we create a :py:class:`~roboticstoolbox.models.URDF.Panda` robot and and then get the :py:class:`~roboticstoolbox.robot.ETS.ETS` representation. Subsequently, we use one of the fast IK solvers available within the :py:class:`~roboticstoolbox.robot.ETS.ETS` class.

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

The :py:class:`~roboticstoolbox.robot.IK.IKSolver` provides basic functionality for performing numerical IK. Superclasses can inherit this class and must implement the :py:meth:`~roboticstoolbox.robot.IK.IKSolver.solve` method. Additionally a superclass redefine any other methods necessary such as :py:meth:`~roboticstoolbox.robot.IK.IKSolver.error` to provide a custom error function.

.. rubric:: The Solution DataClass

.. toctree::
    :maxdepth: 1

    iksolution

The :py:class:`~roboticstoolbox.robot.IK.IKSolution` is a :py:class:`dataclasses.dataclass` instance with the following members.

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

.. rubric:: The Implemented IK Solvers

These solvers can be identified as a :py:class:`Class` starting with ``IK_``.

.. toctree::
    :maxdepth: 1

    ik_lm
    ik_qp
    ik_gn
    ik_nr

.. rubric:: Example 

In the following example, we create an IK Solver class and pass an :py:class:`~roboticstoolbox.robot.ETS.ETS` to it to solve the problem. This style may be preferable to experiments where you wish to compare the same solver on different robots.

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



.. IK Solvers Available with an ETS
.. ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Additionally, these :py:class:`Class` based solvers have been implemented as methods within the :py:class:`~roboticstoolbox.robot.ETS.ETS` and :py:class:`~roboticstoolbox.robot.Robot.Robot` classes. The method names start with ``ikine_``.


.. toctree:
   :caption: IK Solvers from an ETS


.. rubric:: ETS Python IK Methods


.. autosummary::
    :toctree: stubs
    
    ~roboticstoolbox.robot.ETS.ETS.ikine_LM
    ~roboticstoolbox.robot.ETS.ETS.ikine_QP
    ~roboticstoolbox.robot.ETS.ETS.ikine_GN
    ~roboticstoolbox.robot.ETS.ETS.ikine_NR


.. rubric:: Robot Python IK Methods

.. autosummary::
    :toctree: stubs
    
    ~roboticstoolbox.robot.Robot.Robot.ikine_LM
    ~roboticstoolbox.robot.Robot.Robot.ikine_QP
    ~roboticstoolbox.robot.Robot.Robot.ikine_GN
    ~roboticstoolbox.robot.Robot.Robot.ikine_NR


.. rubric:: Example 

In the following example, we create a :py:class:`~roboticstoolbox.models.URDF.Panda` robot and one of the IK solvers available within the :py:class:`~roboticstoolbox.robot.Robot.Robot` class. This style is far more convenient than the above example.

.. runblock:: pycon

    >>> import roboticstoolbox as rtb
    >>> # Make a Panda robot
    >>> panda = rtb.models.Panda()
    >>> # Make a goal pose
    >>> Tep = panda.fkine([0, -0.3, 0, -2.2, 0, 2, 0.7854])
    >>> # Solve the IK problem
    >>> panda.ikine_LM(Tep)

In the following example, we create a :py:class:`~roboticstoolbox.models.URDF.Panda` robot and and then get the :py:class:`~roboticstoolbox.robot.ETS.ETS` representation. Subsequently, we use one of the IK solvers available within the :py:class:`~roboticstoolbox.robot.ETS.ETS` class.

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
