#STANFORD Load kinematic and dynamic data for Stanford arm
#
# Defines the object 'stanford' in the current workspace which describes the 
# kinematic and dynamic characterstics of the Stanford (Scheinman) arm.
#
# Kinematic data from "Modelling, Trajectory calculation and Servoing of 
# a computer controlled arm".  Stanford AIM-177.  Figure 2.3
# Dynamic data from "Robot manipulators: mathematics, programming and control"
# Paul 1981, Tables 6.4, 6.6
# 
# Note: gear ratios not currently known, though reflected armature inertia 
# is known, so gear ratios set to 1.
#
# Also define the vector qz which corresponds to the zero joint
# angle configuration.
#
# See also: ROBOT, PUMA560, PUMA560AKB, TWOLINK.

# $Log: stanford.m,v $
# Revision 1.2  2002/04/01 11:47:18  pic
# General cleanup of code: help comments, see also, copyright, remnant dh/dyn
# references, clarification of functions.
#
# $Revision: 1.2 $
# Copyright (C) 1990-2002, by Peter I. Corke

from numpy import *
from Link import *
from Robot import *

L = []
L.append(Link(alpha=-pi/2, A=0, theta=0, D=0.412, sigma=0))
L.append(Link(alpha=pi/2, A=0, theta=0, D=0.154, sigma=0))
L.append(Link(alpha=0, A=0, theta=-pi/2, D=0, sigma=1))
L.append(Link(alpha=-pi/2, A=0, theta=0, D=0, sigma=0))
L.append(Link(alpha=pi/2, A=0, theta=0, D=0, sigma=0))
L.append(Link(alpha=0, A=0, theta=0, D=0.263, sigma=0))

L[0].m = 9.29
L[1].m = 5.01
L[2].m = 4.25
L[3].m = 1.08
L[4].m = 0.63
L[5].m = 0.51

L[0].r = mat([0,	.0175,	-0.1105])
L[1].r = mat([0,	-1.054,	0])
L[2].r = mat([0,	0,	-6.447])
L[3].r = mat([0,	0.092,	-0.054])
L[4].r = mat([0,	0,	0.566])
L[5].r = mat([0,	0,	1.554])

L[0].I = mat([0.276,	0.255,	0.071,	0,	0,	0])
L[1].I = mat([0.108,	0.018,	0.100,	0,	0,	0])
L[2].I = mat([2.51,	2.51,	0.006,	0,	0,	0])
L[3].I = mat([0.002,	0.001,	0.001,	0,	0,	0])
L[4].I = mat([0.003,	0.003,	0.0004,	0,	0,	0])
L[5].I = mat([0.013,	0.013,	0.0003,	0,	0,	0])

L[0].Jm = 0.953
L[1].Jm = 2.193
L[2].Jm = 0.782
L[3].Jm = 0.106
L[4].Jm = 0.097
L[5].Jm = 0.020

L[0].G = 1
L[1].G = 1
L[2].G = 1
L[3].G = 1
L[4].G = 1
L[5].G = 1

qz = [0,0,0,0,0,0]

stanf = Robot(links=L, name='Stanford arm')
