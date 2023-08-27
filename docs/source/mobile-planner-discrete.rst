Discrete (Grid-based) planners
==============================


=========================================================   ====================   ===================   ===================
Planner                                                     Plans in               Discrete/Continuous   Obstacle avoidance
=========================================================   ====================   ===================   ===================
:class:`~roboticstoolbox.mobile.Bug2`                       :math:`\mathbb{R}^2`   discrete              yes
:class:`~roboticstoolbox.mobile.DistanceTransformPlanner`   :math:`\mathbb{R}^2`   discrete              yes
:class:`~roboticstoolbox.mobile.DstarPlanner`               :math:`\mathbb{R}^2`   discrete              yes
=========================================================   ====================   ===================   ===================


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
