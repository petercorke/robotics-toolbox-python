'''Test quaternions and tranform primitives'''

from robot import *
from robot.puma560 import *

set_printoptions(precision=4, suppress=True);

tests = '''
(q,qd,qdd) = jtraj(qz, qr, 20)
q
qd
qdd

(q,qd,qdd) = jtraj(qz, qr, 20, 0.1*mat(ones((1,6))), -0.1*mat(ones((1,6))) )
q
qd
qdd

(q,qd,qdd) = jtraj(qz, qr, arange(0, 10, 0.2))
q

t1 = trotx(0.1) * transl(0.2, 0.3, 0.4)
t1
t2 = troty(-0.3) * transl(-0.2, -0.3, 0.6)
t2
ctraj(t1, t2, 5)
ctraj(t1, t2, arange(0, 1, 0.1))
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

