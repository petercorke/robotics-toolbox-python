'''Test dynamics'''

from robot import *
from robot.puma560akb import *
from robot.puma560 import *

set_printoptions(precision=4, suppress=True);

tests = '''
q1 = mat(ones((1,6)))

# test the RNE primitive for DH
p560
rne(p560, qz, qz, qz)
rne(p560, qz, qz, qz, gravity=[1,2,3])
rne(p560, qz, qz, qz, fext=[1,2,3,0,0,0])
rne(p560, qz, 0.2*q1, qz)
rne(p560, qz, 0.2*q1, 0.2*q1)

z = hstack((qr, qz, qz))
rne(p560, z, gravity=[1,2,3])
rne(p560, z, fext=[1,2,3,0,0,0])
rne(p560, vstack((z,z,z)))

# test the RNE primitive for MDH
p560m
rne(p560m, qz, qz, qz)
rne(p560m, qz, qz, qz, gravity=[1,2,3])
rne(p560m, qz, qz, qz, fext=[1,2,3,0,0,0])
rne(p560m, qz, 0.2*q1, qz)
rne(p560m, qz, 0.2*q1, 0.2*q1)

z = hstack((qr, qz, qz))
rne(p560m, z, gravity=[1,2,3])
rne(p560m, z, fext=[1,2,3,0,0,0])
rne(p560m, vstack((z,z,z)))

# at zero pose
gravload(p560, qz)
gravload(p560, qz, gravity=[9,0,0])
gravload(p560, vstack((qz,qz,qz)))

inertia(p560, qz)
inertia(p560, vstack((qz,qr)))

accel(p560, qz, 0.2*q1, 0.5*q1)

# need to pick a non-singular configuration for cinertia()
cinertia(p560, qn)

coriolis(p560, qn, 0.5*q1)

# along trajectory
(q,qd,qdd) = jtraj(qz, qr, 20)
rne(p560, q, qd, qdd)
''';

for line in tests.split('\n'):
    if line == '' or line[0] in '%#':
        continue;
    print '::', line;
    if ' = ' in line:
        exec line;
    else:
        print eval(line);
    print

