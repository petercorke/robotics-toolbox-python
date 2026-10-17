ET and ETS: 3D
==============

An :class:`~roboticstoolbox.ets.ET.ET` is one elementary :math:`\SE{3}`
transform: a rotation about, or a translation along, the x-, y- or z-axis.  Its
amount is either a constant or, if it is left out, a variable, which makes the
transform a joint.  An :class:`~roboticstoolbox.ets.ETS.ETS` is a sequence of
them.  The two-dimensional equivalents are on the :doc:`ets_2d` page.

.. runblock:: pycon

   >>> from roboticstoolbox import ET
   >>> ET.Rz()                      # a revolute joint
   >>> ET.tx(0.5)                   # a constant translation
   >>> ET.Ry(90, 'deg')             # a constant rotation
   >>> ET.tz(flip=True, qlim=[0, 1])  # a prismatic joint, flipped, with limits
   >>> e = ET.Rz() * ET.tx(1) * ET.Ry() * ET.tz()
   >>> e.structure                  # the joint types
   >>> e.n, e.m                     # joints, and transforms

ET
--

Creating an elementary transform
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The constructors ``Rx``, ``Ry``, ``Rz``, ``tx``, ``ty`` and ``tz`` take the
same arguments.  The first, ``param``, is what makes the transform a constant
or a joint.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Call
     - Result
   * - ``ET.Rz()``
     - a joint, its value is supplied at evaluation time
   * - ``ET.Rz(0.3)``
     - a constant, the angle in radians
   * - ``ET.Rz(30, 'deg')``
     - a constant, the angle in degrees (rotations only)
   * - ``ET.tx(0.5)``
     - a constant, the distance in the units of the robot
   * - ``ET.Rz(flip=True)``
     - a joint that moves in the opposite sense
   * - ``ET.Rz(jindex=2)``
     - a joint with an explicit index in the joint vector of a robot
   * - ``ET.Rz(qlim=[-1, 1])``
     - a joint with joint limits
   * - ``ET.Rz("q2")``
     - a joint described by a string, see :ref:`ets-descriptors`
   * - ``ET.SE3(T)``
     - a constant, an arbitrary :math:`\SE{3}` transform

.. runblock:: pycon

   >>> from roboticstoolbox import ET
   >>> ET.Rz()                         # a joint
   >>> ET.Rz(0.3)                      # a constant, radians
   >>> ET.Rz(30, 'deg')                # a constant, degrees
   >>> ET.tz(flip=True, qlim=[0, 1])   # a flipped prismatic joint with limits
   >>> ET.Rz(jindex=2).jindex          # an explicit joint index
   >>> from spatialmath import SE3
   >>> ET.SE3(SE3(1, 2, 3))            # a constant arbitrary transform

.. autosummary::
   :nosignatures:

   ~roboticstoolbox.ets.ET.ET.Rx
   ~roboticstoolbox.ets.ET.ET.Ry
   ~roboticstoolbox.ets.ET.ET.Rz
   ~roboticstoolbox.ets.ET.ET.tx
   ~roboticstoolbox.ets.ET.ET.ty
   ~roboticstoolbox.ets.ET.ET.tz
   ~roboticstoolbox.ets.ET.ET.SE3

Information
^^^^^^^^^^^

.. autosummary::
   :nosignatures:

   ~roboticstoolbox.ets.ET.ET.kind
   ~roboticstoolbox.ets.ET.ET.axis
   ~roboticstoolbox.ets.ET.ET.ax
   ~roboticstoolbox.ets.ET.ET.param
   ~roboticstoolbox.ets.ET.ET.isjoint
   ~roboticstoolbox.ets.ET.ET.isflip
   ~roboticstoolbox.ets.ET.ET.isrotation
   ~roboticstoolbox.ets.ET.ET.istranslation
   ~roboticstoolbox.ets.ET.ET.iselementary
   ~roboticstoolbox.ets.ET.ET.jindex
   ~roboticstoolbox.ets.ET.ET.qlim

Kinematics
^^^^^^^^^^

.. autosummary::
   :nosignatures:

   ~roboticstoolbox.ets.ET.ET.A
   ~roboticstoolbox.ets.ET.ET.inv
   ~roboticstoolbox.ets.ET.ET.s

.. _ets-descriptors:

Joint descriptors
^^^^^^^^^^^^^^^^^

The amount of a transform can also be a *string* that names a joint.  This is
convenient when a robot is written down from a datasheet or a URDF, where joints
have names such as ``q1``, ``theta2`` or ``A3``.  The string is parsed once, when
the transform is created, into the joint index, whether the joint is flipped,
and the name that is displayed.

.. runblock:: pycon

   >>> from roboticstoolbox import ET
   >>> e = ET.Rz("q2")              # the number is the joint index
   >>> e, e.jindex, e.isflip
   >>> e = ET.Rz("-q1")             # a leading minus flips the joint
   >>> e, e.jindex, e.isflip
   >>> e = ET.Rz("theta2")          # any name, the first number is the index
   >>> e, e.jindex
   >>> e = ET.Rz("A1_")             # trailing underscore: counts from one
   >>> e, e.jindex
   >>> ET.Rz("joint").jindex        # no number, numbered automatically
   >>> ET.Rz("1.5")                 # a number, it is a constant

A whole robot can be written with names that match its documentation.

.. runblock:: pycon

   >>> from roboticstoolbox import ET
   >>> e = ET.tz(0.4) * ET.Rz("A1_") * ET.Ry("-A2_") * ET.tx(0.5) * ET.Ry("A3_")
   >>> e
   >>> e.jindices
   >>> e.structure

A descriptor string must not be combined with ``jindex`` or ``flip``: ``ET.Rz("q1", jindex=2)`` raises ``ValueError``.  The complete rules are:

.. autoclass:: roboticstoolbox.ets._ET.BaseET
   :noindex:

Details
^^^^^^^

.. autoclass:: roboticstoolbox.ets.ET.ET
   :members: Rx, Ry, Rz, tx, ty, tz, SE3, kind, axis, ax, param, eta, isjoint, isflip, isrotation, istranslation, iselementary, jindex, qlim, A, inv, s
   :show-inheritance:
   :member-order: bysource

ETS
---

An ETS is a list-like object.  It supports ``len``, indexing and slicing, and
the list methods ``append``, ``extend``, ``pop``, ``remove``, ``reverse``,
``index`` and ``count``.  Multiplying elementary transforms, or sequences,
gives a new sequence.

Information
^^^^^^^^^^^

.. autosummary::
   :nosignatures:

   ~roboticstoolbox.ets.ETS.ETS.n
   ~roboticstoolbox.ets.ETS.ETS.m
   ~roboticstoolbox.ets.ETS.ETS.structure
   ~roboticstoolbox.ets.ETS.ETS.joints
   ~roboticstoolbox.ets.ETS.ETS.joint_idx
   ~roboticstoolbox.ets.ETS.ETS.jindices
   ~roboticstoolbox.ets.ETS.ETS.jindex_set
   ~roboticstoolbox.ets.ETS.ETS.qlim
   ~roboticstoolbox.ets.ETS.ETS.random_q

Modification
^^^^^^^^^^^^

.. autosummary::
   :nosignatures:

   ~roboticstoolbox.ets.ETS.ETS.insert
   ~roboticstoolbox.ets.ETS.ETS.inv
   ~roboticstoolbox.ets.ETS.ETS.split
   ~roboticstoolbox.ets.ETS.ETS.swap
   ~roboticstoolbox.ets.ETS.ETS.merge
   ~roboticstoolbox.ets.ETS.ETS.compile

Kinematics
^^^^^^^^^^

.. autosummary::
   :nosignatures:

   ~roboticstoolbox.ets.ETS.ETS.fkine
   ~roboticstoolbox.ets.ETS.ETS.eval
   ~roboticstoolbox.ets.ETS.ETS.jacob0
   ~roboticstoolbox.ets.ETS.ETS.jacobe
   ~roboticstoolbox.ets.ETS.ETS.jacob0_analytical
   ~roboticstoolbox.ets.ETS.ETS.hessian0
   ~roboticstoolbox.ets.ETS.ETS.hessiane
   ~roboticstoolbox.ets.ETS.ETS.partial_fkine0
   ~roboticstoolbox.ets.ETS.ETS.manipulability
   ~roboticstoolbox.ets.ETS.ETS.jacobm

Inverse kinematics
^^^^^^^^^^^^^^^^^^

There are three ways to solve the numerical inverse kinematics problem for a
sequence.  All are described, with examples and the shared arguments, on the
:ref:`IK <IK>` page.  The first two are methods of the ETS, and a robot has
the same methods, which call them on its own sequence.

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Method
     - Poses
     - Notes
   * - :meth:`~roboticstoolbox.ets.ETS.ETS.ik_LM`,
       :meth:`~roboticstoolbox.ets.ETS.ETS.ik_NR`,
       :meth:`~roboticstoolbox.ets.ETS.ETS.ik_GN`
     - one
     - fast, written in C++, not easy to extend
   * - :meth:`~roboticstoolbox.ets.ETS.ETS.ikine_LM`,
       :meth:`~roboticstoolbox.ets.ETS.ETS.ikine_NR`,
       :meth:`~roboticstoolbox.ets.ETS.ETS.ikine_GN`,
       :meth:`~roboticstoolbox.ets.ETS.ETS.ikine_QP`
     - one, or a trajectory
     - written in Python, so easy to extend, and has the quadratic
       programming solver
   * - :class:`~roboticstoolbox.robot.IK.IK_LM`,
       :class:`~roboticstoolbox.robot.IK.IK_NR`,
       :class:`~roboticstoolbox.robot.IK.IK_GN`,
       :class:`~roboticstoolbox.robot.IK.IK_QP`
     - one
     - solver objects, ``IK_LM().solve(ets, Tep)``, that take an ETS rather
       than a robot, convenient for comparing solvers on different robots

Each returns an :class:`~roboticstoolbox.robot.IK.IKSolution`.

Other
^^^^^

.. autosummary::
   :nosignatures:

   ~roboticstoolbox.ets.ETS.ETS.plot
   ~roboticstoolbox.ets.ETS.ETS.teach

Details
^^^^^^^

.. autoclass:: roboticstoolbox.ets.ETS.ETS
   :members: n, m, structure, joints, joint_idx, jindices, jindex_set, qlim, random_q, insert, inv, split, swap, merge, compile, fkine, eval, jacob0, jacobe, jacob0_analytical, hessian0, hessiane, partial_fkine0, manipulability, jacobm, ik_LM, ik_NR, ik_GN, ikine_LM, ikine_NR, ikine_GN, ikine_QP, plot, teach
   :show-inheritance:
   :member-order: bysource
