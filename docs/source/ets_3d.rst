ETS for 3D space
=================

An :class:`~roboticstoolbox.ET` is one elementary :math:`\SE{3}`
transform: a rotation about, or a translation along, the x-, y- or z-axis.  The
angle or distance is either a constant or, if it is left out, a variable, which
makes the transform a joint.  An :class:`~roboticstoolbox.ETS` is a sequence of
them.  The two-dimensional equivalents are on the :doc:`ets_2d` page.

We can use :class:`~roboticstoolbox.ETS` to describes a planar arm that moves in 3D space.

.. runblock:: pycon

   >>> from roboticstoolbox import ET
   >>> ET.Rz()                      # a revolute joint
   >>> ET.tx(0.5)                   # a constant translation
   >>> ET.Ry(90, 'deg')             # a constant rotation
   >>> ET.tz(flip=True, qlim=[0, 1])  # a prismatic joint, flipped, with limits
   >>> e = ET.Rz() * ET.tx(1) * ET.Ry() * ET.tz()
   >>> e.structure                  # the joint types
   >>> e.n, e.m                     # joints, and transforms

ET: Elementary Transform
-------------------------

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

The arguments are:

``param``
    the angle (radians) or distance of a constant transform, left out for a
    joint, or a string that names a joint, see :ref:`ets-descriptors`
``unit``
    ``'rad'`` (default) or ``'deg'``, the unit of the angle of a constant
    rotation
``flip``
    if True the joint moves in the opposite sense, ``q`` is negated
``jindex``
    the index of this joint in the joint vector of a robot, by default the
    joints are numbered in the order they appear
``qlim``
    the joint limits, a 2-vector ``[lower, upper]``

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

   ~roboticstoolbox.ET.Rx
   ~roboticstoolbox.ET.Ry
   ~roboticstoolbox.ET.Rz
   ~roboticstoolbox.ET.tx
   ~roboticstoolbox.ET.ty
   ~roboticstoolbox.ET.tz
   ~roboticstoolbox.ET.SE3

Information
^^^^^^^^^^^

.. autosummary::
   :nosignatures:

   ~roboticstoolbox.ET.kind
   ~roboticstoolbox.ET.axis
   ~roboticstoolbox.ET.ax
   ~roboticstoolbox.ET.param
   ~roboticstoolbox.ET.isjoint
   ~roboticstoolbox.ET.isflip
   ~roboticstoolbox.ET.offset
   ~roboticstoolbox.ET.isrotation
   ~roboticstoolbox.ET.istranslation
   ~roboticstoolbox.ET.iselementary
   ~roboticstoolbox.ET.jindex
   ~roboticstoolbox.ET.qlim

Kinematics
^^^^^^^^^^

.. autosummary::
   :nosignatures:

   ~roboticstoolbox.ET.A
   ~roboticstoolbox.ET.inv
   ~roboticstoolbox.ET.s

.. _ets-descriptors:

Joint descriptors
^^^^^^^^^^^^^^^^^

The parameter of a transform can also be a *string* that names a joint.  This is
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

The rules are:

- the first number in the string is the joint index, so ``"q0"`` is joint 0
  and ``"theta2"`` is joint 2; ``"q(3)"`` and ``"θ_3"`` are joint 3
- a leading ``-`` (or the Unicode minus sign ``−``) negates the joint
  variable, ``flip``, so ``"-q1"`` is joint 1, flipped.  A leading ``+`` is
  ignored
- joint indices count from zero.  If the names count from one, as ``"A1"`` to
  ``"A6"`` of a Kuka arm do, end the name with an underscore: ``"A1_"`` is
  joint 0, and ``"-A1_"`` is joint 0, flipped.  The underscore is not part of
  the name that is displayed
- a string with no number, such as ``"joint"``, is a variable joint that is
  numbered automatically
- a string that is a number, such as ``"1.5"``, is a constant
- ``jindex`` and ``flip`` must not also be given, ``ET.Rz("q1", jindex=2)`` raises ``ValueError``

The name is shown when the transform is displayed, ``Rz(A1)``.

.. note:: A robot built from names ``"A1"`` to ``"A6"``, without the
    underscores, has joint indices 1 to 6, and is rejected as the index 6 is
    out of range for 6 joints.

.. versionchanged:: 1.5.0
    A trailing underscore, and the Unicode minus sign, are accepted.

Details
^^^^^^^

.. autoclass:: roboticstoolbox.ET
   :members: Rx, Ry, Rz, tx, ty, tz, SE3, kind, axis, ax, param, eta, isjoint, isflip, offset, isrotation, istranslation, iselementary, jindex, qlim, A, inv, s
   :member-order: bysource

ETS: Elementary Transform Sequence
-------------------------------------

An ETS is a list-like object.  It supports ``len``, indexing and slicing, iteration, and
the list methods ``append``, ``extend``, ``pop``, ``remove``, ``reverse``,
``index`` and ``count``.  Multiplying elementary transforms, or sequences,
gives a new sequence.

Information
^^^^^^^^^^^

.. autosummary::
   :nosignatures:

   ~roboticstoolbox.ETS.n
   ~roboticstoolbox.ETS.m
   ~roboticstoolbox.ETS.structure
   ~roboticstoolbox.ETS.joints
   ~roboticstoolbox.ETS.joint_idx
   ~roboticstoolbox.ETS.jindices
   ~roboticstoolbox.ETS.jindex_set
   ~roboticstoolbox.ETS.qlim
   ~roboticstoolbox.ETS.random_q

Modification
^^^^^^^^^^^^

.. autosummary::
   :nosignatures:

   ~roboticstoolbox.ETS.insert
   ~roboticstoolbox.ETS.inv
   ~roboticstoolbox.ETS.fixjoint
   ~roboticstoolbox.ETS.split
   ~roboticstoolbox.ETS.swap
   ~roboticstoolbox.ETS.swapped
   ~roboticstoolbox.ETS.merge
   ~roboticstoolbox.ETS.merged
   ~roboticstoolbox.ETS.replace
   ~roboticstoolbox.ETS.replaced
   ~roboticstoolbox.ETS.compile

List operations
^^^^^^^^^^^^^^^

A sequence is a list of elementary transforms, so it has the methods of a list.  Those that change it do so in place.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Method
     - Does
   * - :meth:`~roboticstoolbox.ETS.append`
     - adds an elementary transform to the end
   * - :meth:`~roboticstoolbox.ETS.extend`
     - adds the transforms of another sequence to the end
   * - ``pop``
     - removes and returns the transform at an index, by default the last
   * - :meth:`~roboticstoolbox.ETS.remove`
     - removes the first transform that is equal to the given one
   * - :meth:`~roboticstoolbox.ETS.reverse`
     - reverses the order of the transforms, which is not the inverse of the sequence, see :meth:`~roboticstoolbox.ETS.inv`
   * - ``clear``
     - removes all the transforms
   * - ``count``
     - the number of transforms equal to the given one
   * - ``index``
     - the index of the first transform equal to the given one

Kinematics
^^^^^^^^^^

.. autosummary::
   :nosignatures:

   ~roboticstoolbox.ETS.fkine
   ~roboticstoolbox.ETS.eval
   ~roboticstoolbox.ETS.jacob0
   ~roboticstoolbox.ETS.jacobe
   ~roboticstoolbox.ETS.jacob0_analytical
   ~roboticstoolbox.ETS.hessian0
   ~roboticstoolbox.ETS.hessiane
   ~roboticstoolbox.ETS.partial_fkine0
   ~roboticstoolbox.ETS.manipulability
   ~roboticstoolbox.ETS.jacobm

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
   * - :meth:`~roboticstoolbox.ETS.ik_LM`,
       :meth:`~roboticstoolbox.ETS.ik_NR`,
       :meth:`~roboticstoolbox.ETS.ik_GN`
     - one
     - fast, written in C++, not easy to extend; they need the compiled extension, so not in the browser, and no SymPy parameters or joint offsets, see :ref:`ik-cpp-limits`
   * - :meth:`~roboticstoolbox.ETS.ikine_LM`,
       :meth:`~roboticstoolbox.ETS.ikine_NR`,
       :meth:`~roboticstoolbox.ETS.ikine_GN`,
       :meth:`~roboticstoolbox.ETS.ikine_QP`
     - one, or a trajectory
     - methods that use the solver objects in the next row, and add the
       quadratic programming solver
   * - :class:`~roboticstoolbox.robot.IK.IK_LM`,
       :class:`~roboticstoolbox.robot.IK.IK_NR`,
       :class:`~roboticstoolbox.robot.IK.IK_GN`,
       :class:`~roboticstoolbox.robot.IK.IK_QP`
     - one, or a trajectory
     - solver objects, written in Python so easy to extend,
       ``IK_LM().solve(ets, Tep)``, that take an ETS rather than a robot,
       convenient for comparing solvers on different robots

Each returns an :class:`~roboticstoolbox.robot.IK.IKSolution`.

Other
^^^^^

.. autosummary::
   :nosignatures:

   ~roboticstoolbox.ETS.plot
   ~roboticstoolbox.ETS.teach

Details
^^^^^^^

.. autoclass:: roboticstoolbox.ETS
   :members: n, m, structure, joints, joint_idx, jindices, jindex_set, qlim, random_q, insert, append, extend, remove, reverse, inv, fixjoint, split, swap, swapped, merge, merged, replace, replaced, compile, fkine, eval, jacob0, jacobe, jacob0_analytical, hessian0, hessiane, partial_fkine0, manipulability, jacobm, ik_LM, ik_NR, ik_GN, ikine_LM, ikine_NR, ikine_GN, ikine_QP, plot, teach
   :member-order: bysource
