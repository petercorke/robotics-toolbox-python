Mobile robot path planning
==========================

A set of path planners that all inherit from :class:`Planner`.

Some planners are based on code from the PathPlanning category of
`PythonRobotics <https://github.com/AtsushiSakai/PythonRobotics>`_ by Atsushi Sakai.

.. py:module:: roboticstoolbox.mobile

.. inheritance-diagram:: Bug2Planner DistanceTransformPlanner DstarPlanner DubinsPlanner ReedsSheppPlanner CurvaturePolyPlanner QuinticPolyPlanner
    :parts: 1
    :top-classes: roboticstoolbox.mobile.Planner


========================   ====================   ===================
Planner                    Plans in               Obstacle avoidance
========================   ====================   ===================
Bug2Planner                :math:`\mathbb{R}^2`   yes
DistanceTransformPlanner   :math:`\mathbb{R}^2`   yes
DstarPlanner               :math:`\mathbb{R}^2`   yes
PRMPlanner                 :math:`\mathbb{R}^2`   yes
LatticePlanner             :math:`\mathbb{R}^2`   yes
DubinsPlanner              :math:`\SE{2}`         no
ReedsSheppPlanner          :math:`\SE{2}`         no
CurvaturePolyPlanner       :math:`\SE{2}`         no
QuinticPolyPlanner         :math:`\SE{2}`         no
========================   ====================   ===================

Grid-based planners
-------------------



Distance transform planner
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: roboticstoolbox.mobile.DistanceTransformPlanner
   :members: plan, query, plot
   :undoc-members:
   :show-inheritance:


D* planner
^^^^^^^^^^

.. autoclass:: roboticstoolbox.mobile.DstarPlanner
   :members: plan, query, plot
   :undoc-members:
   :show-inheritance:

PRM planner
^^^^^^^^^^^

.. autoclass:: roboticstoolbox.mobile.PRMPlanner
   :members: plan, query, plot
   :undoc-members:
   :show-inheritance:

Lattice planner
^^^^^^^^^^^^^^^

.. autoclass:: roboticstoolbox.mobile.LatticePlanner
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :exclude-members: isoccupied, validate_point, occgrid, plan, random, message, plot_bg

Configuration-space planners
----------------------------

These planners do not support planning around obstacles, but allow for the
start and goal configuration :math:`(x, y, \theta)` to be specified.

Dubins path planner
^^^^^^^^^^^^^^^^^^^

.. autoclass:: roboticstoolbox.mobile.DubinsPlanner
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :exclude-members: isoccupied, validate_point, occgrid, plan, random, message, plot_bg


Reeds-Shepp path planner
^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: roboticstoolbox.mobile.ReedsSheppPlanner
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :exclude-members: isoccupied, validate_point, occgrid, plan, random, message, plot_bg

Curvature-polynomial planner
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: roboticstoolbox.mobile.CurvaturePolyPlanner
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :exclude-members: isoccupied, validate_point, occgrid, plan, random, message, plot_bg

Quintic-polynomial planner
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: roboticstoolbox.mobile.QuinticPolyPlanner
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :exclude-members: isoccupied, validate_point, occgrid, plan, random, message, plot_bg


Supporting classes
------------------

Planner superclass
^^^^^^^^^^^^^^^^^^

.. autoclass:: roboticstoolbox.mobile.Planner
   :members:
   :undoc-members:
   :show-inheritance:

Occupancy grid class
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: roboticstoolbox.mobile.OccGrid
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__