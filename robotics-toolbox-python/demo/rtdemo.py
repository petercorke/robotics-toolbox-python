#RTDEMO 	Robot toolbox demonstrations
#
# Displays popup menu of toolbox demonstration scripts that illustrate:
#   * homogeneous transformations
#   * trajectories
#   * forward kinematics
#   * inverse kinematics
#   * robot animation
#   * inverse dynamics
#   * forward dynamics
#
# The scripts require the user to periodically hit <Enter> in order to move
# through the explanation.  Set PAUSE OFF if you want the scripts to run
# completely automatically.

# $Log: rtdemo.m,v $
# Revision 1.3  2002/04/02 12:26:48  pic
# Handle figures better, control echo at end of each script.
# Fix bug in calling ctraj.
#
# Revision 1.2  2002/04/01 11:47:17  pic
# General cleanup of code: help comments, see also, copyright, remnant dh/dyn
# references, clarification of functions.
#
# $Revision: 1.3 $
# Copyright (C) 1993-2002, by Peter I. Corke

echo off
clear all
delete( get(0, 'Children') );

puma560
while 1,
 selection = menu('Robot Toolbox demonstrations', ...
 	'Transformations', ...
 	'Trajectory', ...
 	'Forward kinematics', ...
 	'Animation', ...
 	'Inverse kinematics', ...
 	'Jacobians', ...
 	'Inverse dynamics', ...
 	'Forward dynamics', ...
 	'Exit');

 switch selection,
 case 1,
 	rttrdemo
 case 2,
 	rttgdemo
 case 3,
 	rtfkdemo
 case 4,
 	rtandemo
 case 5,
 	rtikdemo
 case 6,
 	rtjademo
 case 7,
 	rtidemo
 case 8,
 	rtfddemo
 case 9,
	delete( get(0, 'Children') );
 	break;
 end
end
