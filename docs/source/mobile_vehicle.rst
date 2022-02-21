Mobile robot kinematic models
=============================

These vehicle kinematic classes have methods to:

* predict new configuration based on odometry
* compute configuration derivative
* simulate and animate motion
* compute Jacobians

Bicycle model
^^^^^^^^^^^^^

  .. autoclass:: roboticstoolbox.mobile.Bicycle
   :members:
   :undoc-members:
   :inherited-members:
   :show-inheritance:
   :special-members: __init__

Unicycle model
^^^^^^^^^^^^^^

.. autoclass:: roboticstoolbox.mobile.Unicycle
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :special-members: __init__

 
Superclass
^^^^^^^^^^

.. autoclass:: roboticstoolbox.mobile.VehicleBase
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :special-members: __init__