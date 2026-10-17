ET2 and ETS2: 2D
================

The planar equivalents of :doc:`ets_3d`.  An :class:`~roboticstoolbox.ets.ET2.ET2`
is one elementary :math:`\SE{2}` transform: a rotation, or a translation along
the x- or y-axis.  An :class:`~roboticstoolbox.ets.ETS2.ETS2` is a sequence of
them, and describes a planar arm.

.. runblock:: pycon

   >>> from roboticstoolbox import ET2
   >>> e = ET2.R() * ET2.tx(1) * ET2.R() * ET2.tx(1)
   >>> print(e)
   >>> e.fkine([0.3, 0.4])

.. note:: ``ETS2`` has no inverse kinematics, Hessian or manipulability
   methods yet, see `Issue 824 <https://github.com/petercorke/robotics-toolbox-python/issues/824>`_.

ET2
---

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

   ~roboticstoolbox.ets.ET2.ET2.R
   ~roboticstoolbox.ets.ET2.ET2.tx
   ~roboticstoolbox.ets.ET2.ET2.ty
   ~roboticstoolbox.ets.ET2.ET2.SE2

Information
^^^^^^^^^^^

.. autosummary::
   :nosignatures:

   ~roboticstoolbox.ets.ET2.ET2.kind
   ~roboticstoolbox.ets.ET2.ET2.axis
   ~roboticstoolbox.ets.ET2.ET2.ax
   ~roboticstoolbox.ets.ET2.ET2.param
   ~roboticstoolbox.ets.ET2.ET2.isjoint
   ~roboticstoolbox.ets.ET2.ET2.isflip
   ~roboticstoolbox.ets.ET2.ET2.isrotation
   ~roboticstoolbox.ets.ET2.ET2.istranslation
   ~roboticstoolbox.ets.ET2.ET2.iselementary
   ~roboticstoolbox.ets.ET2.ET2.jindex
   ~roboticstoolbox.ets.ET2.ET2.qlim

Kinematics
^^^^^^^^^^

.. autosummary::
   :nosignatures:

   ~roboticstoolbox.ets.ET2.ET2.A
   ~roboticstoolbox.ets.ET2.ET2.inv
   ~roboticstoolbox.ets.ET2.ET2.s

Details
^^^^^^^

.. autoclass:: roboticstoolbox.ets.ET2.ET2
   :members: R, tx, ty, SE2, kind, axis, ax, param, eta, isjoint, isflip, isrotation, istranslation, iselementary, jindex, qlim, A, inv, s
   :show-inheritance:
   :member-order: bysource

ETS2
----

An ETS2 is a list-like object, with the same list methods as the
:class:`~roboticstoolbox.ets.ETS.ETS`.  Joint descriptors are the same as for
the 3D case, see :ref:`ets-descriptors`.

Information
^^^^^^^^^^^

.. autosummary::
   :nosignatures:

   ~roboticstoolbox.ets.ETS2.ETS2.n
   ~roboticstoolbox.ets.ETS2.ETS2.m
   ~roboticstoolbox.ets.ETS2.ETS2.structure
   ~roboticstoolbox.ets.ETS2.ETS2.joints
   ~roboticstoolbox.ets.ETS2.ETS2.joint_idx
   ~roboticstoolbox.ets.ETS2.ETS2.jindices
   ~roboticstoolbox.ets.ETS2.ETS2.jindex_set
   ~roboticstoolbox.ets.ETS2.ETS2.qlim
   ~roboticstoolbox.ets.ETS2.ETS2.random_q

Modification
^^^^^^^^^^^^

.. autosummary::
   :nosignatures:

   ~roboticstoolbox.ets.ETS2.ETS2.insert
   ~roboticstoolbox.ets.ETS2.ETS2.inv
   ~roboticstoolbox.ets.ETS2.ETS2.split
   ~roboticstoolbox.ets.ETS2.ETS2.swap
   ~roboticstoolbox.ets.ETS2.ETS2.merge
   ~roboticstoolbox.ets.ETS2.ETS2.compile

Kinematics
^^^^^^^^^^

.. autosummary::
   :nosignatures:

   ~roboticstoolbox.ets.ETS2.ETS2.fkine
   ~roboticstoolbox.ets.ETS2.ETS2.eval
   ~roboticstoolbox.ets.ETS2.ETS2.jacob0
   ~roboticstoolbox.ets.ETS2.ETS2.jacobe

Other
^^^^^

.. autosummary::
   :nosignatures:

   ~roboticstoolbox.ets.ETS2.ETS2.plot
   ~roboticstoolbox.ets.ETS2.ETS2.teach

Details
^^^^^^^

.. autoclass:: roboticstoolbox.ets.ETS2.ETS2
   :members: n, m, structure, joints, joint_idx, jindices, jindex_set, qlim, random_q, insert, inv, split, swap, merge, compile, fkine, eval, jacob0, jacobe, plot, teach
   :show-inheritance:
   :member-order: bysource
