'''Test quaternions and tranform primitives'''

from robot import *
from robot.puma560 import *

set_printoptions(precision=4, suppress=True);

tests = '''
p560

jacob0(p560, qz)
jacob0(p560, qr)
jacob0(p560, qn)

jacobn(p560, qz)
jacobn(p560, qr)
jacobn(p560, qn)

t = fkine(p560, qn)
tr2jac(t)
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

