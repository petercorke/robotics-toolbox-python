'''Test manipulability'''

from robot import *
from robot.puma560 import *

set_printoptions(precision=4, suppress=True);

tests = """
p560

q = qn
manipulability(p560, q)
manipulability(p560, q, 'yoshi')
manipulability(p560, q, 'y')
manipulability(p560, q, 'asada')
a=manipulability(p560, q, 'a')
real(a)
imag(a)
manipulability(p560, q, 'z')

qq = vstack((q, q, q, q))
qq
manipulability(p560, qq, 'yoshi')
manipulability(p560, qq, 'asada')
""";

for line in tests.split('\n'):
    if line == '' or line[0] in '%#':
        continue;
    print '::', line;
    if '=' in line:
        exec line;
    else:
        print eval(line);
    print

