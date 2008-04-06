# Copyright (C) 1993-2002, by Peter I. Corke

# $Log: rtikdemo.m,v $
# Revision 1.3  2002/04/02 12:26:49  pic
# Handle figures better, control echo at end of each script.
# Fix bug in calling ctraj.
#
# Revision 1.2  2002/04/01 11:47:17  pic
# General cleanup of code: help comments, see also, copyright, remnant dh/dyn
# references, clarification of functions.
#
# $Revision: 1.3 $
figure(2)
echo on
#
# Inverse kinematics is the problem of finding the robot joint coordinates,
# given a homogeneous transform representing the last link of the manipulator.
# It is very useful when the path is planned in Cartesian space, for instance 
# a straight line path as shown in the trajectory demonstration.
#
# First generate the transform corresponding to a particular joint coordinate,
    q = [0 -pi/4 -pi/4 0 pi/8 0]
    T = fkine(p560, q);
#
# Now the inverse kinematic procedure for any specific robot can be derived 
# symbolically and in general an efficient closed-form solution can be 
# obtained.  However we are given only a generalized description of the 
# manipulator in terms of kinematic parameters so an iterative solution will 
# be used. The procedure is slow, and the choice of starting value affects 
# search time and the solution found, since in general a manipulator may 
# have several poses which result in the same transform for the last
# link. The starting point for the first point may be specified, or else it
# defaults to zero (which is not a particularly good choice in this case)
    qi = ikine(p560, T);
    qi'
#
# Compared with the original value
    q
#
# A solution is not always possible, for instance if the specified transform 
# describes a point out of reach of the manipulator.  As mentioned above 
# the solutions are not necessarily unique, and there are singularities 
# at which the manipulator loses degrees of freedom and joint coordinates 
# become linearly dependent.
pause % any key to continue
#
# To examine the effect at a singularity lets repeat the last example but for a
# different pose.  At the `ready' position two of the Puma's wrist axes are 
# aligned resulting in the loss of one degree of freedom.
    T = fkine(p560, qr);
    qi = ikine(p560, T);
    qi'
#
# which is not the same as the original joint angle
    qr
pause % any key to continue
#
# However both result in the same end-effector position
    fkine(p560, qi) - fkine(p560, qr)
pause % any key to continue
    
# Inverse kinematics may also be computed for a trajectory.
# If we take a Cartesian straight line path
    t = [0:.056:2]; 		% create a time vector
    T1 = transl(0.6, -0.5, 0.0) % define the start point
    T2 = transl(0.4, 0.5, 0.2)	% and destination
    T = ctraj(T1, T2, length(t)); 	% compute a Cartesian path

#
# now solve the inverse kinematics.  When solving for a trajectory, the 
# starting joint coordinates for each point is taken as the result of the 
# previous inverse solution.
#
    tic
    q = ikine(p560, T); 
    toc
#
# Clearly this approach is slow, and not suitable for a real robot controller 
# where an inverse kinematic solution would be required in a few milliseconds.
#
# Let's examine the joint space trajectory that results in straightline 
# Cartesian motion
    subplot(3,1,1)
    plot(t,q(:,1))
    xlabel('Time (s)');
    ylabel('Joint 1 (rad)')
    subplot(3,1,2)
    plot(t,q(:,2))
    xlabel('Time (s)');
    ylabel('Joint 2 (rad)')
    subplot(3,1,3)
    plot(t,q(:,3))
    xlabel('Time (s)');
    ylabel('Joint 3 (rad)')

pause % hit any key to continue
    
# This joint space trajectory can now be animated
    plot(p560, q)
pause % any key to continue
echo off
