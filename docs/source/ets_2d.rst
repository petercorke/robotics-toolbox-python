ETS for 2D space
=================

An :class:`~roboticstoolbox.ET2` is one elementary :math:`\SE{2}`
transform: a rotation about, or a translation along, the x- or y-axis.  The
angle or distance is either a constant or, if it is left out, a variable, which
makes the transform a joint.  An :class:`~roboticstoolbox.ETS2` is a sequence of
them.  The three-dimensional equivalents are on the :doc:`ets_3d` page.

We can use :class:`~roboticstoolbox.ETS` to describes a planar arm that moves in 3D space.

.. runblock:: pycon

   >>> from roboticstoolbox import ET2
   >>> e = ET2.R() * ET2.tx(1) * ET2.R() * ET2.tx(1)
   >>> print(e)
   >>> e.fkine([0.3, 0.4])

.. note:: ``ETS2`` has no inverse kinematics, Hessian or manipulability
   methods yet, see `Issue 824 <https://github.com/petercorke/robotics-toolbox-python/issues/824>`_.

ET2: Elementary Transform
-------------------------

Creating an elementary transform
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The constructors ``R``, ``tx`` and ``ty`` take the same arguments as their
3D equivalents on the :doc:`ets_3d` page: no argument for a joint, a number for
a constant, ``'deg'`` for a rotation in degrees, ``flip``, ``jindex`` and
``qlim``.  The angle or distance can also be a joint descriptor string, with the same
rules, see :ref:`ets-descriptors`.

.. runblock:: pycon

   >>> from roboticstoolbox import ET2
   >>> ET2.R()                        # a joint
   >>> ET2.R(30, 'deg')               # a constant, degrees
   >>> ET2.tx(flip=True, qlim=[0, 1]) # a flipped prismatic joint with limits
   >>> e = ET2.R("q0") * ET2.tx(1) * ET2.R("-q1") * ET2.tx(1)
   >>> e, e.jindices

.. autosummary::
   :nosignatures:

   ~roboticstoolbox.ET2.R
   ~roboticstoolbox.ET2.tx
   ~roboticstoolbox.ET2.ty
   ~roboticstoolbox.ET2.SE2

Information
^^^^^^^^^^^

.. autosummary::
   :nosignatures:

   ~roboticstoolbox.ET2.kind
   ~roboticstoolbox.ET2.axis
   ~roboticstoolbox.ET2.ax
   ~roboticstoolbox.ET2.param
   ~roboticstoolbox.ET2.isjoint
   ~roboticstoolbox.ET2.isflip
   ~roboticstoolbox.ET2.offset
   ~roboticstoolbox.ET2.isrotation
   ~roboticstoolbox.ET2.istranslation
   ~roboticstoolbox.ET2.iselementary
   ~roboticstoolbox.ET2.jindex
   ~roboticstoolbox.ET2.qlim

Kinematics
^^^^^^^^^^

.. autosummary::
   :nosignatures:

   ~roboticstoolbox.ET2.A
   ~roboticstoolbox.ET2.inv
   ~roboticstoolbox.ET2.s

Details
^^^^^^^

.. autoclass:: roboticstoolbox.ET2
   :members: R, tx, ty, SE2, kind, axis, ax, param, eta, isjoint, isflip, offset, isrotation, istranslation, iselementary, jindex, qlim, A, inv, s
   :member-order: bysource

ETS: Elementary Transform Sequence
------------------------------------

An ETS2 is a list-like object, with the same list methods as the
:class:`~roboticstoolbox.ETS`.  Joint descriptors are the same as for
the 3D case, see :ref:`ets-descriptors`.

Information
^^^^^^^^^^^

.. autosummary::
   :nosignatures:

   ~roboticstoolbox.ETS2.n
   ~roboticstoolbox.ETS2.m
   ~roboticstoolbox.ETS2.structure
   ~roboticstoolbox.ETS2.joints
   ~roboticstoolbox.ETS2.joint_idx
   ~roboticstoolbox.ETS2.jindices
   ~roboticstoolbox.ETS2.jindex_set
   ~roboticstoolbox.ETS2.qlim
   ~roboticstoolbox.ETS2.random_q

Modification
^^^^^^^^^^^^

.. autosummary::
   :nosignatures:

   ~roboticstoolbox.ETS2.insert
   ~roboticstoolbox.ETS2.inv
   ~roboticstoolbox.ETS2.fixjoint
   ~roboticstoolbox.ETS2.split
   ~roboticstoolbox.ETS2.swap
   ~roboticstoolbox.ETS2.swapped
   ~roboticstoolbox.ETS2.merge
   ~roboticstoolbox.ETS2.merged
   ~roboticstoolbox.ETS2.compile

List operations
^^^^^^^^^^^^^^^

A sequence is a list of elementary transforms, so it has the methods of a list.  Those that change it do so in place.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Method
     - Does
   * - :meth:`~roboticstoolbox.ETS2.append`
     - adds an elementary transform to the end
   * - :meth:`~roboticstoolbox.ETS2.extend`
     - adds the transforms of another sequence to the end
   * - ``pop``
     - removes and returns the transform at an index, by default the last
   * - :meth:`~roboticstoolbox.ETS2.remove`
     - removes the first transform that is equal to the given one
   * - :meth:`~roboticstoolbox.ETS2.reverse`
     - reverses the order of the transforms, which is not the inverse of the sequence, see :meth:`~roboticstoolbox.ETS2.inv`
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

   ~roboticstoolbox.ETS2.fkine
   ~roboticstoolbox.ETS2.eval
   ~roboticstoolbox.ETS2.jacob0
   ~roboticstoolbox.ETS2.jacobe

Other
^^^^^

.. autosummary::
   :nosignatures:

   ~roboticstoolbox.ETS2.plot
   ~roboticstoolbox.ETS2.teach

Details
^^^^^^^

.. autoclass:: roboticstoolbox.ETS2
   :members: n, m, structure, joints, joint_idx, jindices, jindex_set, qlim, random_q, insert, append, extend, remove, reverse, inv, fixjoint, split, swap, swapped, merge, merged, compile, fkine, eval, jacob0, jacobe, plot, teach
   :member-order: bysource
