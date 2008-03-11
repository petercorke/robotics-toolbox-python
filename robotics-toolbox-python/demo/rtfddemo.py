# Copyright (C) 1993-2002, by Peter I. Corke

# $Log: rtfddemo.m,v $
# Revision 1.4  2002/04/02 12:26:48  pic
# Handle figures better, control echo at end of each script.
# Fix bug in calling ctraj.
#
# Revision 1.3  2002/04/01 11:47:17  pic
# General cleanup of code: help comments, see also, copyright, remnant dh/dyn
# references, clarification of functions.
#
# $Revision: 1.4 $
figure(2)
echo on
#
# Forward dynamics is the computation of joint accelerations given position and
# velocity state, and actuator torques.  It is useful in simulation of a robot
# control system.
#
# Consider a Puma 560 at rest in the zero angle pose, with zero applied joint 
# torques. The joint acceleration would be given by
    accel(p560, qz, zeros(1,6), zeros(1,6))
pause % any key to continue
#
# To be useful for simulation this function must be integrated.  fdyn() uses the
# MATLAB function ode45() to integrate the joint acceleration.  It also allows 
# for a user written function to compute the joint torque as a function of 
# manipulator state.
#
# To simulate the motion of the Puma 560 from rest in the zero angle pose 
# with zero applied joint torques
    tic
    [t q qd] = fdyn(nofriction(p560), 0, 10);
    toc
#
# and the resulting motion can be plotted versus time
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
#
# Clearly the robot is collapsing under gravity, but it is interesting to 
# note that rotational velocity of the upper and lower arm are exerting 
# centripetal and Coriolis torques on the waist joint, causing it to rotate.

pause % hit any key to continue
#
# This can be shown in animation also
    clf
    plot(p560, q)
pause % hit any key to continue
echo off
