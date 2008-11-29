'''Test Link and Robot objects'''

from robot import *
from robot.puma560 import *

set_printoptions(precision=4, suppress=True);

tests = '''
l = Link(1, 2, 3, 4, 0, Link.LINK_DH)
l
l = Link(1, 2, 3, 4, 1, Link.LINK_DH)
l

l = Link(1, 2, 3, 4, 0, Link.LINK_MDH)
l
l = Link(1, 2, 3, 4, 1, Link.LINK_MDH)
l

l = Link(1, 2, 3, 4, 0, Link.LINK_DH)
l.display()
l.offset = 9
l.m = 10
l.G = 11
l.Jm = 12
l.B = 14

l.r = [1,2,3]
l.r
l.r = mat([1,2,3])
l.r
l.I = [1,2,3]
l.I
l.I = [1,2,3,4,5,6]
l.I
l.I = mat(diag([1,2,3]))
l.I

l.Tc = 1
l.Tc
l.Tc = [-1,2]
l.Tc

l.qlim = array([4,5])

l.display()
l2 = l.nofriction()
l.display()
l2.display()

l.friction(2)
l.friction(-2)

l.tr(0)
l.tr(0.2)

p560
p560.n
p560.mdh
p560.links
p560.base
p560.tool
p560.config()
p560.ismdh()

pbig=p560*p560
pbig
p560

p560.showlinks()

p560nf = p560.nofriction()
p560nf.showlinks()

p560.base = transl(1,2,3);
p560.base

p560.tool = transl(4,5,6);
p560.tool

p560.gravity = [1,2,3]
p560.gravity



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

