*************
Mobile robots
*************


The Robotics Toolbox supports kino-dynamic vehicle models, animation, planning
and state estimation.

.. toctree::
   :maxdepth: 2

   mobile-vehicle
   mobile-vehicle-animation
   mobile-drivers
   mobile-planner
   mobile-reactive
   mobile-SLAM

.. _mobile-workspace:

The workspace
=============

Mobile robots live in a rectangular region of the plane, the *workspace*,
written as ``[xmin, xmax, ymin, ymax]``.  It is the extent of a map, the limits of
a plot, and the region from which random goals, landmarks and particles are drawn.

Many classes have a ``workspace`` argument: the vehicles, the drivers
(:class:`~roboticstoolbox.mobile.drivers.RandomPath`), the maps
(:class:`~roboticstoolbox.mobile.landmarkmap.LandmarkMap`,
:class:`~roboticstoolbox.mobile.OccGrid.BinaryOccupancyGrid`,
:class:`~roboticstoolbox.mobile.OccGrid.PolygonMap`) and the estimators
(:class:`~roboticstoolbox.mobile.EKF.EKF`,
:class:`~roboticstoolbox.mobile.ParticleFilter.ParticleFilter`).
All of them accept the same things:

==================  ==============================  ==========================
``workspace``       meaning                         x-range, y-range
==================  ==============================  ==========================
``A`` (scalar)      a square about the origin       ``-A:A``, ``-A:A``
``[A, B]``          the same range on both axes     ``A:B``, ``A:B``
``[A, B, C, D]``    both ranges                     ``A:B``, ``C:D``
an object           anything with a ``workspace``   that object's workspace
==================  ==============================  ==========================

so a vehicle can be told to move in the world of a map, or a map can be made to
fit a vehicle, by passing one to the other::

    map = LandmarkMap(20, workspace=10)
    veh = Bicycle(workspace=map)
    veh.control = RandomPath(workspace=map)

Precedence
----------

When more than one workspace is available, the one that is used is, in order:

#. an explicit ``workspace`` argument;
#. a map that already has a concrete extent (an occupancy grid, a landmark map);
#. the vehicle's workspace;
#. otherwise there is none, which is an error for the classes that need one.

The principle is that *something with a concrete extent beats something that is
declared*: a map loaded from a file cannot change its size, so the vehicle
conforms to it, whereas a map that is being generated can be made to fit the
vehicle by giving it the vehicle as its ``workspace``.  For example, a
:class:`~roboticstoolbox.mobile.drivers.RandomPath` with no workspace of its own
chooses waypoints inside the workspace of the vehicle it is attached to, and an
:class:`~roboticstoolbox.mobile.EKF.EKF` that is animated shows the vehicle in the
workspace of its map.

Classes never modify the workspace of another object other than through its
public ``workspace`` property.

.. autofunction:: roboticstoolbox.mobile.workspace.resolve_workspace
