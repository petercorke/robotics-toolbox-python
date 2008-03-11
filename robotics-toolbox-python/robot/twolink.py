#TWOLINK Load kinematic and dynamic data for a simple 2-link mechanism
#
#	TWOLINK
#
# Defines the object 'tl' in the current workspace which describes the 
# kinematic and dynamic characterstics of a simple planar 2-link mechanism.
#
# Example based on Fig 3-6 (p73) of Spong and Vidyasagar (1st edition).  
# It is a planar mechanism operating in the XY (horizontal) plane and is 
# therefore not affected by gravity.
#
# Assume unit length links with all mass (unity) concentrated at the joints.
#
# Also define the vector qz = [0 0] which corresponds to the zero joint
# angle configuration.
#
# See also: DH, DYN, PUMA560, PUMA560AKB, STANFORD.

# $Log: twolink.m,v $
# Revision 1.2  2002/04/01 11:47:20  pic
# General cleanup of code: help comments, see also, copyright, remnant dh/dyn
# references, clarification of functions.
#
# $Revision: 1.2 $
# Copyright (C) 2000-2002, by Peter I. Corke

from numpy import *
from Robot import *
from Link import *

L = []

L.append(Link(A=1))
L.append(Link(A=1))

L[0].m = 1
L[1].m = 1

L[0].r = mat([1,0,0])
L[1].r = mat([1,0,0])

L[0].I = mat([0,0,0,0,0,0])
L[1].I = mat([0,0,0,0,0,0])

L[0].Jm = 0
L[1].Jm = 0

L[0].G = 1
L[1].G = 1

qz = [0,0]

tl = Robot(links=L,name='Simple two link')

