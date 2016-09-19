# robotics-toolbox-python
Robotics Toolbox for Python

This is an old first attempt at creating an open Python version of the venerable Robotics Toolbox for MATLAB.  
The MATLAB toolbox has support for:

* mobile robots
  - vehicle kinematic models and controllers
  - path planners (distance xform, D*, PRM, lattice, RRT)
  - dead-reckoning, localization, mapping, SLAM
  
* robot manipulator arms
  - kinematics forward and inverse
  - Jacobians
  - rigid-body dynamics
  
* common datastructures for
  - SO2/SE2 planar rotations and rigid-body motion
  - SO3/SE3 3D and rigid-body motion
  - quaternions
  - twists, Plucker lines
  
With matplotlib, scipy, numpy and jupyter it should be possible to create a very effective open robotics environment.
