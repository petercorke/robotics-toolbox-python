Trajectories
============

A trajectory is how something moves from one place to another: the value of each coordinate at each instant of time.  For a robot it is the way to get an arm to make a smooth motion instead of jumping to its goal.  A trajectory can be of the joint coordinates, in which case the robot's joints follow it, or of the pose of the end-effector, which the robot follows by solving inverse kinematics for each point of it.

The Toolbox makes trajectories as arrays, one row for each time step, so that they can be given directly to the methods of a robot.  The scalar and joint-space functions return a :py:class:`~roboticstoolbox.tools.trajectory.Trajectory` object, which holds the position ``q``, velocity ``qd`` and acceleration ``qdd``, and can plot itself.  The Cartesian function returns a sequence of poses.

.. runblock:: pycon

    >>> import roboticstoolbox as rtb
    >>> puma = rtb.models.DH.Puma560()
    >>> tg = rtb.jtraj(puma.qz, puma.qr, 50)  # 50 steps from one configuration to another
    >>> tg.q.shape
    >>> T = puma.fkine(tg.q)  # the pose of the end-effector at each step
    >>> len(T)

.. rubric:: Functions that make trajectories

.. autosummary::
    :nosignatures:

    ~roboticstoolbox.tools.trajectory.jtraj
    ~roboticstoolbox.tools.trajectory.ctraj
    ~roboticstoolbox.tools.trajectory.mtraj
    ~roboticstoolbox.tools.trajectory.mstraj
    ~roboticstoolbox.tools.trajectory.trapezoidal
    ~roboticstoolbox.tools.trajectory.quintic
    ~roboticstoolbox.tools.trajectory.trapezoidal_func
    ~roboticstoolbox.tools.trajectory.quintic_func

``jtraj`` is the joint-space trajectory between two configurations, which a robot also has as the method :meth:`~roboticstoolbox.robot.Robot.Robot.jtraj`, and ``ctraj`` is the Cartesian trajectory between two poses.  ``mtraj`` makes a multi-axis trajectory by applying a scalar function, ``trapezoidal`` or ``quintic``, to each axis, and ``mstraj`` passes through a sequence of via points.  The ``_func`` versions return a function of time instead of an array.

.. rubric:: Functions and methods that use trajectories

.. autosummary::
    :nosignatures:

    ~roboticstoolbox.robot.Robot.Robot.fkine
    ~roboticstoolbox.robot.Robot.Robot.ikine_LM
    roboticstoolbox.robot.Robot.Robot.plot
    ~roboticstoolbox.robot.Robot.Robot.rne
    roboticstoolbox.tools.trajectory.Trajectory.plot
    ~roboticstoolbox.tools.plot.xplot

A joint trajectory can be given to ``fkine`` to get the pose of the end-effector at each step, to ``plot`` to animate the robot, and, with its velocity and acceleration, to ``rne`` to get the joint torques needed to follow it.  A pose trajectory is given to the ``ikine_XX`` methods, see :doc:`Inverse Kinematics <IK/ik>`, to get the joint trajectory that follows it.  ``plot`` and ``xplot`` plot a trajectory against time.

Reference
---------

.. automodule:: roboticstoolbox.tools.trajectory
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :special-members:
   :exclude-members: __dict__, __module__, __weakref__

.. autofunction:: roboticstoolbox.tools.plot.xplot
