Localization and Mapping
========================

These classes support simulation of vehicle and map estimation in a simple
planar world with point landmarks.


State estimation
----------------

Two state estimators are included.

Extended Kalman filter
^^^^^^^^^^^^^^^^^^^^^^

The EKF is capable of vehicle localization, map estimation or SLAM.

.. autoclass:: roboticstoolbox.mobile.EKF
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :special-members: __init__

Particle filter
^^^^^^^^^^^^^^^

The particle filter is capable of map-based vehicle localization.

.. autoclass:: roboticstoolbox.mobile.ParticleFilter
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :special-members: __init__

Sensor models
-------------

.. autoclass:: roboticstoolbox.mobile.RangeBearingSensor
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :special-members: __init__

.. autoclass:: roboticstoolbox.mobile.SensorBase
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :special-members: __init__

Map models
----------

.. automodule:: roboticstoolbox.mobile.landmarkmap
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :special-members: __init__


