Mobile robot path planning
**************************

A set of path planners for robots operating in planar environments with
configuration :math:`\vec{q} \in \mathbb{R}^2` or :math:`\vec{q} \in \SE{2}`.
All inherit from :class:`PlannerBase`.

Some planners are based on code from the PathPlanning category of
`PythonRobotics <https://github.com/AtsushiSakai/PythonRobotics>`_ by Atsushi Sakai.

.. inheritance-diagram:: roboticstoolbox.mobile.DistanceTransformPlanner roboticstoolbox.mobile.DstarPlanner roboticstoolbox.mobile.DubinsPlanner roboticstoolbox.mobile.ReedsSheppPlanner roboticstoolbox.mobile.QuinticPolyPlanner roboticstoolbox.mobile.CurvaturePolyPlanner roboticstoolbox.mobile.RRTPlanner
    :parts: 1

=========================================================   ====================   ===================   ===================
Planner                                                     Plans in               Discrete/Continuous   Obstacle avoidance
=========================================================   ====================   ===================   ===================
:class:`~roboticstoolbox.mobile.Bug2`                       :math:`\mathbb{R}^2`   discrete              yes
:class:`~roboticstoolbox.mobile.DistanceTransformPlanner`   :math:`\mathbb{R}^2`   discrete              yes
:class:`~roboticstoolbox.mobile.DstarPlanner`               :math:`\mathbb{R}^2`   discrete              yes
:class:`~roboticstoolbox.mobile.PRMPlanner`                 :math:`\mathbb{R}^2`   continuous            yes
:class:`~roboticstoolbox.mobile.LatticePlanner`             :math:`\SE{2}`         discrete              yes
:class:`~roboticstoolbox.mobile.DubinsPlanner`              :math:`\SE{2}`         continuous            no
:class:`~roboticstoolbox.mobile.ReedsSheppPlanner`          :math:`\SE{2}`         continuous            no
:class:`~roboticstoolbox.mobile.CurvaturePolyPlanner`       :math:`\SE{2}`         continuous            no
:class:`~roboticstoolbox.mobile.QuinticPolyPlanner`         :math:`\SE{2}`         continuous            no
:class:`~roboticstoolbox.mobile.RRTPlanner`                 :math:`\SE{2}`         continuous            yes
=========================================================   ====================   ===================   ===================



Discrete (Grid-based) planners
==============================


Distance transform planner
--------------------------

.. autoclass:: roboticstoolbox.mobile.DistanceTransformPlanner
   :members: 
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :exclude-members: random, random_init, progress_start, progress_end, progress_next, message

D* planner
----------

.. autoclass:: roboticstoolbox.mobile.DstarPlanner
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :exclude-members: next, isoccupied, random, random_init, progress_start, progress_end, progress_next, message

PRM planner
-----------

.. autoclass:: roboticstoolbox.mobile.PRMPlanner
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :exclude-members: isoccupied, random, random_init, progress_start, progress_end, progress_next, message, plot_bg


Lattice planner
---------------

.. autoclass:: roboticstoolbox.mobile.LatticePlanner
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :exclude-members: isoccupied, random, random_init, progress_start, progress_end, progress_next, message, plot_bg

Continuous configuration-space planners
=======================================

These planners do not support planning around obstacles, but allow for the
start and goal configuration :math:`(x, y, \theta)` to be specified.

Dubins path planner
-------------------

.. autoclass:: roboticstoolbox.mobile.DubinsPlanner
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :exclude-members: isoccupied, random, random_init, progress_start, progress_end, progress_next, message, plot_bg


Reeds-Shepp path planner
------------------------

.. autoclass:: roboticstoolbox.mobile.ReedsSheppPlanner
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :exclude-members: isoccupied, random, random_init, progress_start, progress_end, progress_next, message, plot_bg

Curvature-polynomial planner
----------------------------

.. autoclass:: roboticstoolbox.mobile.CurvaturePolyPlanner
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :exclude-members: isoccupied, random, random_init, progress_start, progress_end, progress_next, message, plot_bg

Quintic-polynomial planner
--------------------------

.. autoclass:: roboticstoolbox.mobile.QuinticPolyPlanner
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :exclude-members: isoccupied, random, random_init, progress_start, progress_end, progress_next, message, plot_bg

RRT planner
-----------

.. autoclass:: roboticstoolbox.mobile.RRTPlanner
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :exclude-members: isoccupied, random, random_init, progress_start, progress_end, progress_next, message, plot_bg

Map classes
===========

Occupancy grid classes
----------------------

Binary Occupancy grid
^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: roboticstoolbox.mobile.BinaryOccupancyGrid
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :special-members: __init__
   

Occupancy grid
^^^^^^^^^^^^^^

.. autoclass:: roboticstoolbox.mobile.OccupancyGrid
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :special-members: __init__

Polygon map
-----------

.. autoclass:: roboticstoolbox.mobile.PolygonMap
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :special-members: __init__


Supporting classes
==================

Planner superclass
------------------

.. autoclass:: roboticstoolbox.mobile.PlannerBase
   :members:
   :undoc-members:
   :show-inheritance:


Occupancy grid base classes
---------------------------

.. autoclass:: roboticstoolbox.mobile.OccGrid.BaseMap
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: roboticstoolbox.mobile.OccGrid.BaseOccupancyGrid
   :members:
   :undoc-members:
   :show-inheritance:


