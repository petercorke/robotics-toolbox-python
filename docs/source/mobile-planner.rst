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


Using a planner
===============

The planners do not share one call signature, because the algorithms need
different things at different times.  They fall into four families, and the
verbs a planner has are the ones that make sense for it.

.. list-table::
   :header-rows: 1
   :widths: 24 38 18 20

   * - Family
     - Planners
     - ``plan``
     - ``query`` or ``run``
   * - Plan for a goal, then query many starts
     - :class:`~roboticstoolbox.mobile.DistanceTransformPlanner`,
       :class:`~roboticstoolbox.mobile.DstarPlanner`,
       :class:`~roboticstoolbox.mobile.RRTPlanner`
     - ``plan(goal)``
     - ``query(start)``, Dstar also takes a ``sensor`` function
   * - Build a map of the space, then query both ends
     - :class:`~roboticstoolbox.mobile.PRMPlanner`,
       :class:`~roboticstoolbox.mobile.LatticePlanner`
     - ``plan(npoints)`` or ``plan(iterations)``
     - ``query(start, goal)``
   * - Nothing to plan
     - :class:`~roboticstoolbox.mobile.DubinsPlanner`,
       :class:`~roboticstoolbox.mobile.ReedsSheppPlanner`,
       :class:`~roboticstoolbox.mobile.QuinticPolyPlanner`,
       :class:`~roboticstoolbox.mobile.CurvaturePolyPlanner`
     - none
     - ``query(start, goal)``
   * - Reactive
     - :class:`~roboticstoolbox.mobile.Bug2`
     - none
     - ``run(start, goal)``

Failure
-------

- ``ValueError``: the arguments are invalid, for example the start is inside an
  obstacle or has the wrong length.
- :class:`~roboticstoolbox.mobile.PlannerBase.NoPathError`: the problem is valid
  but has no solution, for example the goal is unreachable or the robot is
  trapped.  It is a ``RuntimeError``.
- ``RuntimeError``: ``query`` was called before ``plan``.

Animation
---------

Planners that can animate take ``animate`` and ``movie`` arguments at the
constructor, which set the default for the planner and can be overridden by the
same arguments of the method that runs the animation.  A ``movie`` implies
``animate``.


.. toctree::
   :maxdepth: 2

   mobile-planner-continuous
   mobile-planner-discrete
   mobile-planner-map
   mobile-planner-base