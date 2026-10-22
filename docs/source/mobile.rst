*************
Mobile robots
*************

The Toolbox supports mobile robots that move on a plane: models of how a vehicle moves, drivers that make it move, maps and sensors to perceive its world, and algorithms to plan a path through it, to navigate by reacting to obstacles, and to estimate where the vehicle is.  Each is a class, and they are meant to be combined: a vehicle driven by an agent, in a world of landmarks, observed by a sensor, and tracked by a filter.

.. toctree::
   :maxdepth: 2
   :hidden:

   mobile-vehicle
   mobile-vehicle-animation
   mobile-drivers
   mobile-planner
   mobile-reactive
   mobile-SLAM
   mobile-workspace

.. rubric:: Vehicles

The motion model of a vehicle says how its configuration changes with its speed and steering.  The three models share a superclass, and are described in :doc:`mobile-vehicle`.

.. autosummary::
    :nosignatures:

    ~roboticstoolbox.mobile.Bicycle
    ~roboticstoolbox.mobile.Unicycle
    ~roboticstoolbox.mobile.DiffSteer

A vehicle can be shown moving, by a marker, a polygon or an image, see :doc:`mobile-vehicle-animation`.

.. autosummary::
    :nosignatures:

    ~roboticstoolbox.mobile.VehicleMarker
    ~roboticstoolbox.mobile.VehiclePolygon
    ~roboticstoolbox.mobile.VehicleIcon

.. rubric:: Drivers

A driver is an agent that makes a vehicle move around, which is useful for testing, see :doc:`mobile-drivers`.

.. autosummary::
    :nosignatures:

    ~roboticstoolbox.mobile.drivers.RandomPath
    ~roboticstoolbox.mobile.drivers.PurePursuit

.. rubric:: Maps and sensors

A map is a model of the world, and a sensor measures it from the vehicle.  Maps are used by the planners and by the estimators, see :doc:`mobile-SLAM`.

.. autosummary::
    :nosignatures:

    ~roboticstoolbox.mobile.landmarkmap.LandmarkMap
    ~roboticstoolbox.mobile.OccupancyGrid
    ~roboticstoolbox.mobile.BinaryOccupancyGrid
    ~roboticstoolbox.mobile.PolygonMap
    ~roboticstoolbox.mobile.RangeBearingSensor

.. rubric:: Path planning

A planner finds a path from a start to a goal in a map, see :doc:`mobile-planner` for how to choose and use one.

.. autosummary::
    :nosignatures:

    ~roboticstoolbox.mobile.DistanceTransformPlanner
    ~roboticstoolbox.mobile.DstarPlanner
    ~roboticstoolbox.mobile.PRMPlanner
    ~roboticstoolbox.mobile.RRTPlanner
    ~roboticstoolbox.mobile.LatticePlanner
    ~roboticstoolbox.mobile.DubinsPlanner
    ~roboticstoolbox.mobile.ReedsSheppPlanner
    ~roboticstoolbox.mobile.QuinticPolyPlanner
    ~roboticstoolbox.mobile.CurvaturePolyPlanner

.. rubric:: Reactive navigation

Reactive navigation drives the vehicle to a goal without a plan, by reacting to the obstacles it senses, see :doc:`mobile-reactive`.

.. autosummary::
    :nosignatures:

    ~roboticstoolbox.mobile.Bug2

.. rubric:: State estimation

An estimator works out where the vehicle is, and where the landmarks are, from the odometry of the vehicle and the measurements of a sensor, see :doc:`mobile-SLAM`.

.. autosummary::
    :nosignatures:

    ~roboticstoolbox.mobile.EKF
    ~roboticstoolbox.mobile.ParticleFilter
    ~roboticstoolbox.mobile.PoseGraph

Everything that has a region of the plane to work in takes the same ``workspace`` argument, which is described on :doc:`mobile-workspace`.
