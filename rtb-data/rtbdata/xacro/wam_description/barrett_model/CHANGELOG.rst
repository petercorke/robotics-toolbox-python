^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package barrett_model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

0.5.1 (2014-12-10)
------------------
* fixing collada up-axes
* Merge branch 'master' into devel
* fixing orientation of bhand inertia
* Contributors: Jonathan Bohren

0.5.0 (2014-12-08)
------------------
* adding tf frame for fingertips, fixing naming of the first two fingers
* updating damping to have a global scale factor
* Merge branch 'master' of github.com:jhu-lcsr/barrett_model
* adding cleaner top-level xacro file for barrett system
* reducing friction and disabling collision for some links
* decreasing damping for gazebo 2.2.3
* adding a single xacro file with a bunch of options
* Merge branch 'master' of github.com:jhu-lcsr/barrett_model
* updating palm limits
* fixing stl headers
* Contributors: Jonathan Bohren

0.4.0 (2014-08-29)
------------------
* increasing max effort for wrist joints
* updating ball urdf to have a frame in the middle of the ball
* Contributors: Jonathan Bohren

0.3.0 (2014-07-22)
------------------
* adding additional frames for use as moveit end-effectors
* decreasing joint damping for simulation
* Contributors: Jonathan Bohren

0.2.0 (2014-07-15)
------------------
* decreasing damping for wam joints to increase performance
* increasing damping for simulation stability
* Contributors: Christopher Paxton, Jonathan Bohren

0.1.0 (2014-04-17)
------------------
* fixing joint limits
* more realistic damping
* updating wam ball
* bball
* fixing palm link, forearm, and upper arm inertia (palm link was in the wrong place)
* run dep
* Merge branch 'master' of github.com:jhu-lcsr/barrett_model
* decreasing damping
* Update README.md
* Update README.md
* Update README.md
* adding scrots
* adding simplified collision geometry
* adding realistic velocity limits, damping
* Tweaking limits
* fixing velocity limits which were messing up high-performance motions in gazebo
* increasing bhand damping in simulation
* moving gz tags to common and adding them to the hand
* Using new implicitSpringDamper tag instead of cfmDamping tag
* Adding better damping characterization
* updating for new xacro include tag
* adding in damping that gazebo actually reads
* updating doc
* Major cleanup
* Cleaning up robot urdfs
* updating wrist pitch limits
* Removing old dynamics tags, replacing with new gazebo xacro
* Adding missing mass to link, this was angering gazebo
* updating inertial properties
* Updating inertial parameters from those observed in the lab. Adding stubby 4-dof WAM.
* Fixing inertial properties
* fixing top-level xacro files
* adding better export frame
* gh formatting
* gh formatting
* gh formatting
* gh formatting
* fixing base link coordinate frame
* Create README.md
* adding cad files
* cleaning up...
* new and improved
* catkinizing
* add in bench lattice to urdf
* Removing big wall in base link, fixing damping-induced explosion when adding the last link
* fixing wrist link
* Fixing more stuff in urdf, making base link fixed to the world
* Updates, getting things running in gazebo
* Adding base to wam urdf
* Fixing two arm test, non-FT urdf wrist
* wam sim working with new launchfiles
* lots of launchfile refactoring
* updating wam stub test
* adding 4dof wam for testing
* adding single 7dof wam for testing
* adding ready-to-use models
* streamlining model
* renaming barrett_urdf back to bard_urdf
* renaming bard_urdf to barrett_urdf
* smooth gains, happy IK
* fixed joint effort limits, control switching could be made a bit less complex, joint traj controller is smooth, but could use more testing, experimentation with derivative gains needed
* working ik pose controller
* fixing urdf
* adding tesr urdf
* updating launchfile and fixing urdf
* lots of updates
* functioning grav comp
* working grav comp
* kdl chain solver takes link names and not joint names as arguments
* updating bard urdf to have non-ft stuff
* about to fix the wrist stuff
* adding non-ft barrett wrist
* renaming root to example, and separating the wam arm with and without a hand
* adding this, will be gone soon
* renaming darpa_arm to wam_arm
* removing center bar
* fixing normals
* fixing normals
* adding popeye bench lattice, will move soon
* adding base back in
* removing WAM from urdf path name
* adding bard urdf
* Contributors: Jonathan Bohren
