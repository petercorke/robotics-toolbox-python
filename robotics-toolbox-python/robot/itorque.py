#ITORQUE Compute the manipulator inertia torque
#
#	TAUI = ITORQUE(ROBOT, Q, QDD)
#
# Returns the n-element inertia torque vector at the specified pose and 
# acceleration, that is,
# 	TAUI = INERTIA(Q)*QDD
#
# ROBOT describes the manipulator dynamics and kinematics.
# If Q and QDD are row vectors, the result is a row vector of joint torques.
# If Q and QDD are matrices, each row is interpretted as a joint state 
# vector, and the result is a matrix each row being the corresponding joint 
# torques.
# 
# If ROBOT contains non-zero motor inertia then this will included in the
# result.
#
# See also: RNE, CORIOLIS, INERTIA, GRAVLOAD.


# $Log: itorque.m,v $
# Revision 1.3  2002/04/14 10:17:08  pic
# Update comments.
#
# Revision 1.2  2002/04/01 11:47:14  pic
# General cleanup of code: help comments, see also, copyright, remnant dh/dyn
# references, clarification of functions.
#
# $Revision: 1.3 $
# Copyright (C) 1993-2002, by Peter I. Corke

from rne import *
from numpy import *

def itorque(robot, q, qdd):
    return rne(robot, q, zeros(shape(q)), qdd, [[0],[0],[0]])
