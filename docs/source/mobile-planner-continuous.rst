Continuous configuration-space planners
=======================================


=========================================================   ====================   ===================   ===================
Planner                                                     Plans in               Discrete/Continuous   Obstacle avoidance
=========================================================   ====================   ===================   ===================
:class:`~roboticstoolbox.mobile.DubinsPlanner`              :math:`\SE{2}`         continuous            no
:class:`~roboticstoolbox.mobile.ReedsSheppPlanner`          :math:`\SE{2}`         continuous            no
:class:`~roboticstoolbox.mobile.CurvaturePolyPlanner`       :math:`\SE{2}`         continuous            no
:class:`~roboticstoolbox.mobile.QuinticPolyPlanner`         :math:`\SE{2}`         continuous            no
:class:`~roboticstoolbox.mobile.RRTPlanner`                 :math:`\SE{2}`         continuous            yes
=========================================================   ====================   ===================   ===================


These planners do not support planning around obstacles, but allow for the
start and goal configuration :math:`(x, y, \theta)` to be specified.

PRM planner
-----------

.. autoclass:: roboticstoolbox.mobile.PRMPlanner
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :exclude-members: isoccupied, random, random_init, progress_start, progress_end, progress_next, message, plot_bg


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

