#MANIPLTY Manipulability measure
#
#	M = MANIPLTY(ROBOT, Q)
#	M = MANIPLTY(ROBOT, Q, WHICH)
#
# Computes the manipulability index for the manipulator at the given pose.
#
# For an n-axis manipulator Q may be an n-element vector, or an m x n
# joint space trajectory.
#
# If Q is a vector MANIPLTY returns a scalar manipulability index.
# If Q is a matrix MANIPLTY returns a column vector of  manipulability 
# indices for each pose specified by Q.
#
# The argument WHICH can be either 'yoshikawa' (default) or 'asada' and
# selects one of two manipulability measures.
# Yoshikawa's manipulability measure gives an indication of how far 
# the manipulator is from singularities and thus able to move and 
# exert forces uniformly in all directions.
#
# Asada's manipulability measure is based on the manipulator's
# Cartesian inertia matrix.  An n-dimensional inertia ellipsoid
# 	X' M(q) X = 1
# gives an indication of how well the manipulator can accelerate
# in each of the Cartesian directions.  The scalar measure computed
# here is the ratio of the smallest/largest ellipsoid axis.  Ideally
# the ellipsoid would be spherical, giving a ratio of 1, but in
# practice will be less than 1.
#
# See also: INERTIA, JACOB0.

# MOD HISTORY
# 4/99	object support, matlab local functions
# 6/99	change switch to allow abbreviations of measure type
# $Log: maniplty.m,v $
# Revision 1.2  2002/04/01 11:47:14  pic
# General cleanup of code: help comments, see also, copyright, remnant dh/dyn
# references, clarification of functions.
#
# $Revision: 1.2 $

# Copyright (C) 1993-2002, by Peter I. Corke

from numpy import *
from jacob0 import *
from numpy.linalg import inv,eig,det
from inertia import *
from numrows import *

def maniplty(robot, q, which = 'yoshikawa'):
        n = robot.n
        q = mat(q)
        w = array([])
        if which == 'yoshikawa' or which == 'yoshi' or which == 'y':
                if numrows(q)==1:
                        return yoshi(robot,q)
                for Q in q:
                        w = concatenate((w,array([yoshi(robot,Q)])))
        if which == 'asada' or which == 'a':
                if numrows(q)==1:
                        return asada(robot,q)
                for Q in q:
                        w = concatenate((w,array([asada(robot,Q)])))
        return mat(w)

def yoshi(robot,q):
        J = jacob0(robot,q)
        return sqrt(det(J*J.T))

def asada(robot,q):
        J = jacob0(robot,q)
        Ji = inv(J)
        M = inertia(robot,q)
        Mx = Ji.T*M*Ji
        e = eig(Mx)[0]
        return e.min(0)/e.max(0)
