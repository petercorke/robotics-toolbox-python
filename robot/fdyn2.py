#FDYN2  private function called by FDYN
#
#	XDD = FDYN2(T, X, FLAG, ROBOT, TORQUEFUN)
#
# Called by FDYN to evaluate the robot velocity and acceleration for
# forward dynamics.  T is the current time, X = [Q QD] is the state vector,
# ROBOT is the object being integrated, and TORQUEFUN is the string name of
# the function to compute joint torques and called as
#
#       TAU = TORQUEFUN(T, X)
#
# if not given zero joint torques are assumed.
#
# The result is XDD = [QD QDD].
#
# See also: FDYN

# MOD HISTORY
# 	4/99 add object support
# $Log: fdyn2.m,v $
# Revision 1.2  2002/04/14 10:14:04  pic
# Added support for extra command line arguments passed to torqfun.
# Update comments.
#
# Revision 1.1  2002/04/01 11:47:12  pic
# General cleanup of code: help comments, see also, copyright, remnant dh/dyn
# references, clarification of functions.
#
# $Revision: 1.2 $
# Copyright (C) 1999-2002, by Peter I. Corke

function xd = fdyn2(t, x, flag, robot, torqfun, varargin)

	n = robot.n;

	q = x(1:n);
	qd = x(n+1:2*n);

	% evaluate the torque function if one is given
	if isstr(torqfun)
		tau = feval(torqfun, t, q, qd, varargin{:});
	else
		tau = zeros(n,1);
	end
	
	qdd = accel(robot, x(1:n,1), x(n+1:2*n,1), tau);
	xd = [x(n+1:2*n,1); qdd];
