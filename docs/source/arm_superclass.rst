Superclasses
============


Robot
-------

.. inheritance-diagram:: roboticstoolbox.ERobot
   :top-classes: roboticstoolbox.Robot
   :parts: 2

The various models :ref:`E Models` all subclass this class.

Collision checking
^^^^^^^^^^^^^^^^^^

``robot.iscollided(q, shape)`` checks the robot against an external shape.
``robot.isselfcollided(q)`` checks its own non-adjacent links, including gripper
links; ``robot.iscollided(q, robot)`` is equivalent. Shapes on the same link,
neighbouring links, and links within the same gripper are excluded. Links without
collision geometry are transparent when determining neighbours, so each link's
nearest ancestor with collision geometry is excluded. Links in separate branches
are checked unless they belong to the same gripper, and touching counts as a
collision for the pairs that are checked.

The checks use collision geometry, not visual geometry, and require the optional
``coal`` dependency. A model without collision geometry raises ``ValueError``
for self-collision checking. Links without collision geometry are omitted, so
ensure the model represents all the parts that need checking. Many DH models
provide kinematics and dynamics without collision geometry.

Omitting ``q`` from ``isselfcollided`` uses the robot's current configuration.
Grippers use their own current ``q``. The stored joint configurations are not
changed. ``skip=True`` reuses existing world transforms and should only be used
when the caller has already updated them.

.. automodule:: roboticstoolbox.robot.Robot
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :special-members: __getitem__

Link
----

The ``ERobot`` is defined by a tree of ``ELink`` subclass objects.

.. inheritance-diagram:: roboticstoolbox.Link
   :top-classes: roboticstoolbox.robot.Link
   :parts: 2

.. automodule:: roboticstoolbox.robot.Link
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
