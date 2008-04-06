'''
PUMA560 Load kinematic and dynamic data for a Puma 560 manipulator

	from robot.puma560 import *

Defines the object 'p560' in the current workspace which describes the 
kinematic and dynamic % characterstics of a Unimation Puma 560 manipulator
using standard DH conventions.
The model includes armature inertia and gear ratios.

Also define the vector qz which corresponds to the zero joint
angle configuration, qr which is the vertical 'READY' configuration,
and qstretch in which the arm is stretched out in the X direction.

@see: robot, puma560akb, stanford, twolink

@notes:
   - the value of m1 is given as 0 here.  Armstrong found no value for it
and it does not appear in the equation for tau1 after the substituion
is made to inertia about link frame rather than COG frame.
updated:

Python implementation by: Luis Fernando Lara Tobar and Peter Corke.
Based on original Robotics Toolbox for Matlab code by Peter Corke.
Permission to use and copy is granted provided that acknowledgement of
the authors is made.

@author: Luis Fernando Lara Tobar and Peter Corke
'''

from numpy import *
from Link import *
from Robot import *


print "in puma560"
L = [];
L.append( Link(alpha=pi/2,  A=0,      D=0) )
L.append( Link(alpha=0,     A=0.4318, D=0) )
L.append( Link(alpha=-pi/2, A=0.0203, D=0.15005) )
L.append( Link(alpha=pi/2,  A=0,      D=0.4318) )
L.append( Link(alpha=-pi/2, A=0,      D=0) )
L.append( Link(alpha=0,     A=0,      D=0) )


L[0].m = 0
L[1].m = 17.4
L[2].m = 4.8
L[3].m = 0.82
L[4].m = 0.34
L[5].m = .09

L[0].r = mat([ 0,    0,	   0 ])
L[1].r = mat([ -.3638,  .006,    .2275])
L[2].r = mat([ -.0203,  -.0141,  .070])
L[3].r = mat([ 0,    .019,    0])
L[4].r = mat([ 0,    0,	   0])
L[5].r = mat([ 0,    0,	   .032])

L[0].I = mat([  0,	 0.35,	 0,	 0,	 0,	 0])
L[1].I = mat([  .13,	 .524,	 .539,	 0,	 0,	 0])
L[2].I = mat([   .066,  .086,	 .0125,   0,	 0,	 0])
L[3].I = mat([  1.8e-3,  1.3e-3,  1.8e-3,  0,	 0,	 0])
L[4].I = mat([  .3e-3,   .4e-3,   .3e-3,   0,	 0,	 0])
L[5].I = mat([  .15e-3,  .15e-3,  .04e-3,  0,	 0,	 0])

L[0].Jm =  200e-6
L[1].Jm =  200e-6
L[2].Jm =  200e-6
L[3].Jm =  33e-6
L[4].Jm =  33e-6
L[5].Jm =  33e-6

L[0].G =  -62.6111
L[1].G =  107.815
L[2].G =  -53.7063
L[3].G =  76.0364
L[4].G =  71.923
L[5].G =  76.686

# viscous friction (motor referenced)
L[0].B =   1.48e-3
L[1].B =   .817e-3
L[2].B =    1.38e-3
L[3].B =   71.2e-6
L[4].B =   82.6e-6
L[5].B =   36.7e-6

# Coulomb friction (motor referenced)
L[0].Tc = mat([ .395,	-.435])
L[1].Tc = mat([ .126,	-.071])
L[2].Tc = mat([ .132,	-.105])
L[3].Tc = mat([ 11.2e-3, -16.9e-3])
L[4].Tc = mat([ 9.26e-3, -14.5e-3])
L[5].Tc = mat([ 3.96e-3, -10.5e-3])


#
# some useful poses
#
qz = [0, 0, 0, 0, 0, 0] # zero angles, L shaped pose
qr = [0, pi/2, -pi/2, 0, 0, 0] # ready pose, arm up
qs = [0, 0, -pi/2, 0, 0, 0]
qn=[0, pi/4, pi, 0, pi/4,  0]


p560 = Robot(L, name='Puma 560', manuf='Unimation', comment='params of 8/95')
