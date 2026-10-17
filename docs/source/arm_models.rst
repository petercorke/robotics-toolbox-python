Manipulator models
==================

Dynamic parameters
------------------

Dynamics computations require a model with inertial data. The default
``rtb.models.Panda()`` and ``rtb.models.ETS.Puma560()`` are kinematics-only
models with no dynamic parameters. Use ``rtb.models.DH.Panda()`` or
``rtb.models.DH.Puma560()`` for models that include dynamics. Other models,
including those loaded from URDF, depend on the inertial data supplied by
their source.

Check ``robot.has_dynamics`` before computing dynamics. It is true if any
link has non-zero mass, so individual massless links, such as the first link
of the DH Puma560, are allowed. A model with all link masses zero raises a
``ValueError`` from dynamics computations, naming the model and suggesting
a model with dynamics or setting ``m``, ``r`` and ``I`` on its links.
Changing those link parameters updates availability immediately.

This applies to joint and operational space inertia, Coriolis and gravity
loads, inverse and forward dynamics, friction, joint actuator dynamics,
inertial torque and payload capacity. Kinematic computations and setting or
displaying dynamic parameters do not require inertial data.

Listing available models
-------------------------

.. automodule:: roboticstoolbox.models.catalog
   :members:

.. toctree::
   :maxdepth: 2

   arm_dh
   arm_ets
   arm_erobot
   arm_superclass
