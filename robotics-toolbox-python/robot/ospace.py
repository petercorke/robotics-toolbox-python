#OSPACE Operational space coefficients
#
# [Lambda, mu, p] = ospace(robot, q, qd)
#

from inertia import *
from coriolis import *
from gravload import *
from jacob0 import *
from numpy import *
from numpy.linalg import inv

def ospace(robot, q, qd):
    q = mat(q)
    qd = mat(qd)
    M = inertia(robot, q)
    C = coriolis(robot, q, qd)
    g = gravload(robot, q)
    J = jacob0(robot, q)
    Ji = inv(J)
    print 'Ji\n',Ji,'\n\n'
    print 'J\n',J,'\n\n'
    print 'M\n',M,'\n\n'
    print 'C\n',C,'\n\n'
    print 'g\n',g,'\n\n'
    Lambda = Ji.T*M*Ji
    mu = J.T*C - Lamba*H*qd
    p = J.T*g
    return Lambda, mu, p

