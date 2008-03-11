'''Test quaternions and tranform primitives'''

#from robot import *
from robot.transform import *
from robot.Quaternion import *

tests = '''
quaternion(0.1)
quaternion( mat([1,2,3]), 0.1 )
quaternion( rotx(0.1) )
quaternion( trotx(0.1) )
quaternion( quaternion(0.1) )
quaternion( mat([1,2,3,4]) )
quaternion( array([1,2,3,4]) )
quaternion( [1,2,3,4] )
quaternion( 1, 2, 3, 4)
quaternion( 0.1, mat([1,2,3]) )

q1 = quaternion( rotx(0.1) );
q1.norm()
q1.unit()
q1.norm()
q1.double()
q1.r()
q1.tr()

q1 = quaternion( rotx(0.1) );
q2 = quaternion( roty(0.2) );
q1_t = q1.copy()
q2_t = q2.copy()

q1
q1 *= q2
q1
q1 *= 2
q1
q2

q1 = q1_t
q1
q2 = q2_t

q1*2
2*q1
q1+q2
q1-q2
q1*q2
q1**1
q1**2
q1*q1
q2.inv()
q2*q2.inv()
q2*q2**-1
q1/q2

v1 = mat([0, 1, 0]);
q1*v1
q1*v1.T

q1
q2
q1.interp(q2, 0)
q1.interp(q2, 1)
q1.interp(q2, 0.5)
q1.interp(q2, [0, .2, .5, 1])

q1-q1_t
q2-q2_t
''';

for line in tests.split('\n'):
    if line == '' or line[0] in '%#':
        continue;
    print '::', line;
    if '=' in line:
        exec line;
    else:
        print eval(line);
    print

