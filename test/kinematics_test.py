# DESCRIPTION:
# A test for the kinematics and trajectories codes of the robotics toolbox

# import section
from numpy import *

# import robot models section
from puma560 import *
from puma560akb import *
from stanford import *

# import kinematics section
from fkine import *
from ikine import *
from jacob0 import *
from jacobn import *

# import trajectories section
from ctraj import *
from jtraj import *

# import manipulability section
from maniplty import *

print '\nThis is a test for the kinematics and trajectories codes of the Robotics Toolbox'
print 'Codes are tested with puma560, puma560akb and stanford models in order to proof'
print 'both standar and modified conventions for DH parammeters, prismatic and rotatinal joints\n'

print '\n\t\t\tRobot Models:\n'
print p560
print '\n\n'
print p560m
print '\n\n'
print stanf
print '\n\n'

print '\t\t\t***Kinematics test***\n'
print 'T0 = fkine(p560,[0,0,0,0,0,0])\n'
T0 = fkine(p560,[0,0,0,0,0,0])
print T0,'\n'
print 'q0 = ikine(p560,T0)\n'
q0 = ikine(p560,T0)
print q0,'\n'
print 'fkine(p560,q0)\n'
print fkine(p560,q0),'\n\n'
print 'T1 = fkine(p560m,[pi/2,-pi/4,3*pi/4,-pi/8,0,-pi/5])\n'
T1 = fkine(p560m,[pi/2,-pi/4,3*pi/4,-pi/8,0,-pi/5])
print T1,'\n'
print 'q1 = ikine(p560m,T1)\n'
q1 = ikine(p560m,T1)
print q1,'\n'
print 'fkine(p560m,q1)\n'
print fkine(p560m,q1),'\n\n'
print 'T3 = fkine(stanf,[1,2,0.25,3,2,1])\n'
T3 = fkine(stanf,[1,2,0.25,3,2,1])
print T3,'\n'
print 'q3 = ikine(stanf,T3)\n'
q3 = ikine(stanf,T3)
print q3,'\n'
print 'fkine(stanf,q3)\n'
print fkine(stanf,q3),'\n\n'
print 'q4 = ikine(p560m,fkine(p560m,[1,2,3,1,3,2]))\n'
q4 = ikine(p560m,fkine(p560m,[1,2,3,1,3,2]))
print q4,'\n\n'
print 'q5 = ikine(stanf, fkine(stanf,[1,2,1,0,2,3]) )\n'
q5 = ikine(stanf,fkine(stanf,[1,2,1,0,2,3]))
print q5,'\n\n'
print 'fkine(stanf,q5)\n'
print fkine(stanf,q5),'\n\n'
print 'fkine(stanf,[1,2,1,0,2,3])\n'
print fkine(stanf,[1,2,1,0,2,3]),'\n\n'
print '\t\t\t***Test for Jacobian***\n'
print 'J1 = jacobn(stanf,[1,2,1,0,2,3])\n'
J1 = jacobn(stanf,[1,2,1,0,2,3])
print J1,'\n\n'
print 'J2 = jacobn(p560m,[1,0,3,-3,0,1])\n'
J2 = jacobn(p560m,[1,0,3,-3,0,1])
print J2,'\n\n'
print 'J3 = jacobn(p560,[1,0,3,-3,0,1])\n'
J3 = jacobn(p560,[1,0,3,-3,0,1])
print J3,'\n\n'
print 'J01 = jacob0(stanf,[1,2,1,0,2,3])\n'
J01 = jacob0(stanf,[1,2,1,0,2,3])
print J01,'\n\n'
print 'J02 = jacob0(p560m,[1,0,3,-3,0,1])\n'
J02 = jacob0(p560m,[1,0,3,-3,0,1])
print J02,'\n\n'
print 'J03 = jacob0(p560,[1,0,3,-3,0,1])\n'
J03 = jacob0(p560,[1,0,3,-3,0,1])
print J03,'\n\n'
print '\t\t\t***Trajectory Test***\n'
print 'Joint trajectory'
print 'qj,qdj,qddj = jtraj([0,0,0,0,0,0],[pi/2, pi/4, -3*pi/5, 4*pi/6, 0, 1], 5)\n'
qj,qdj,qddj = jtraj([0,0,0,0,0,0],[pi/2, pi/4, -3*pi/5, 4*pi/6, 0, 1], 5)
print 'qj:\n',qj,'\n\nqdj:\n',qdj,'\n\nqddj:\n',qddj,'\n\n'
print 'Cartesian trajectory'
print 'tt = ctraj(fkine(p560m,[0,0,0,0,0,0]), fkine(p560m,[pi/2,pi/4,-3*pi/4,-pi/8,0,1]), 5)\n'
tt = ctraj(fkine(p560m,[0,0,0,0,0,0]), fkine(p560m,[pi/2,pi/4,-3*pi/4,-pi/8,0,1]), 5)
for i in tt:
    print i,'\n\n'
print '\t\t\t***Trajectory Case for Kinematics Test***\n'
print 'qj:\n',qj,'\n'
print 'Tt1 = fkine(p560m,qj)\n'
Tt1 = fkine(p560m,qj)
for i in Tt1:
    print i,'\n\n'
print 'Qt1 = ikine(p560m,Tt1)\n'
Qt1 = ikine(p560m,Tt1)
for i in Qt1:
    print i,'\n\n'
print 'Qt2 = ikine(stanf, fkine(stanf,[[0,0,0,0,0,0],[1,0,1,0,1,0],[1,0,0.8,0,1,0]]))\n'
Qt2 = ikine(stanf, fkine(stanf,[[0,0,0,0,0,0],[1,0,1,0,1,0],[1,0,0.8,0,1,0]]) )
for i in Qt2:
    print i,'\n\n'
print '\t\t\tManipulability test:'
print '\nmanip1 = maniplty(p560,[1,2,3,4,5,6],\'y\')\n'
manip1 = maniplty(p560,[1,2,3,4,5,6],'y')
print manip1,'\n\n'
print '\nmanip1 = maniplty(p560,[1,2,3,4,5,6],\'a\')\n'
manip1 = maniplty(p560,[1,2,3,4,5,6],'a')
print manip1,'\n\n'
print '\nmanip2 = maniplty(p560m,[1,2,3,4,5,6],\'y\')\n'
manip2 = maniplty(p560m,[1,2,3,4,5,6],'y')
print manip2,'\n\n'
print '\nmanip2 = maniplty(p560m,[1,2,3,4,5,6],\'a\')\n'
manip2 = maniplty(p560m,[1,2,3,4,5,6],'a')
print manip2,'\n\n'
print '\nmanip3 = maniplty(stanf,[pi/3,pi/6,0.3,-3*pi/4,-2*pi/5,pi/6],\'y\')\n'
manip3 = maniplty(stanf,[pi/3,pi/6,0.3,-3*pi/4,-2*pi/5,pi/6],'y')
print manip3,'\n\n'
print '\nmanip3 = maniplty(stanf,[pi/3,pi/6,0.3,-3*pi/4,-2*pi/5,pi/6],\'a\')\n'
manip3 = maniplty(stanf,[pi/3,pi/6,0.3,-3*pi/4,-2*pi/5,pi/6],'a')
print manip3,'\n\n'
print '\nTrajectory case test for manipulability\n'
print '\nmanip1 = maniplty(p560,[[1,2,3,4,5,6],[6,5,4,3,2,1],[0,1,0,1,0,1]],\'a\')\n'
manip1 = maniplty(p560,[[1,2,3,4,5,6],[0,-5,4,0,2,1],[pi/4,-1,0.5,1,-pi/6,1]],'a')
for i in manip1.T:
    print i,'\n\n'

