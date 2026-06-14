Path planning
*************

A set of path planners for robots operating in planar environments with
configuration :math:`\vec{q} \in \mathbb{R}^2` or :math:`\vec{q} \in \mathbb{R}^2 \times S^1 \sim \SE{2}`.
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


.. toctree::
   :maxdepth: 2

   mobile-planner-continuous
   mobile-planner-discrete
   mobile-planner-map
   mobile-planner-base