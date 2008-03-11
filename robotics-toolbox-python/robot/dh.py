#DH Matrix representation of manipulator kinematics
#
# The original robot toolbox functions used a DH matrix to describes the
# kinematics of a manipulator in a general way.
#
# For an n-axis manipulator, DH is an nx4 or nx5 matrix, whose rows 
# comprise
# 
# 	1	alpha	link twist angle
# 	2	A	link length
# 	3	theta	link rotation angle
# 	4	D	link offset distance
# 	5	sigma	joint type, 0 for revolute, non-zero for prismatic
#
# If the last column is not given the manipulator is all-revolute.
#
# The first 5 columns of a DYN matrix contain the kinematic parameters
# and maybe used anywhere that a DH kinematic matrix is required -- the
# dynamic data is ignored.
#
# The functionality of the DH matrix has been replaced by the ROBOT object.
#
# See also: ROBOT, DYN.

# MOD.HISTORY
# 	1/95	reverse labels on A & D
# $Log: dh.m,v $
# Revision 1.2  2002/04/01 11:47:11  pic
# General cleanup of code: help comments, see also, copyright, remnant dh/dyn
# references, clarification of functions.
#
# $Revision: 1.2 $
# Copyright (C) 1993-2002, by Peter I. Corke
