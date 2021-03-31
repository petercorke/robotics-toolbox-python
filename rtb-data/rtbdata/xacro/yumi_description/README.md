Urdf & Gazebo files for the ABB YuMi (IRB 14000) robot

### Dependencies
This package depends on ros-industrial which is not released on kinetic yet. So skip step 2 if you have already install ros-industrial.

#### Step 1
Install all of these:
```
sudo apt-get install ros-kinetic-control-toolbox ros-kinetic-controller-interface ros-kinetic-controller-manager ros-kinetic-joint-limits-interface ros-kinetic-transmission-interface ros-kinetic-moveit-core ros-kinetic-moveit-planners ros-kinetic-moveit-ros-planning
```

#### Step 2
Build the industrial_core package from source. To do that, clone OrebroUniversity's fork of the package from: https://github.com/OrebroUniversity/industrial_core.git into your ros workspace and run `catkin_make`.

#### Step 3
If your industrial_core is in a different workspace, source the workspace containing industrial_core.
Finally, `catkin_make` the workspace containing the clone of this package.
