'''Test quaternions and tranform primitives'''

from robot import *

tests = '''
# utility
v1 = mat([0, 1, 0]);
v2 = mat([0, 0, 1]);
unit(v1+v2)
crossp(v1, v2)
crossp(v1.T, v2)
crossp(v1, v2.T)
crossp(v1.T, v2.T)

m = mat(zeros( (3,4) ));
numcols(m)
numrows(m)

# transform primitives
rotx(.1)
roty(.1)
rotz(.1)

r = rotz(0.1)
r2t(r)


trotx(.1)
troty(.1)
trotz(.1)

t = trotz(0.1)
t2r(t)

t1 = trotx(.1)
t2 = troty(.2)
trinterp(t1, t2, 0)
trinterp(t1, t2, 1)
trinterp(t1, t2, 0.5)

# predicates
isvec(v1)
isvec(r)
isvec(t)

isrot(v1)
isrot(r)
isrot(t)

ishomog(v1)
ishomog(r)
ishomog(t)
linalg.det(t)

# translational matrices
transl(0.1, 0.2, 0.3)
transl( [0.1, 0.2, 0.3] )
transl( mat([0.1, 0.2, 0.3]) )
t = transl(0.1, 0.2, 0.3)
transl(t)

# angle conversions
eul2r(.1, .2, .3)
eul2r( [.1, .2, .3] )
eul2r( mat([[.1, .2, .3], [.4, .5, .6]]) )
eul2r( mat([.1, .2, .3]) )
eul2tr(.1, .2, .3)
eul2tr( [.1, .2, .3] )
eul2tr( mat([.1, .2, .3]) )
te = eul2tr( mat([.1, .2, .3]) )
tr2eul(te)

rpy2r(.1, .2, .3)
rpy2r( [.1, .2, .3] )
rpy2r( mat([.1, .2, .3]) )
rpy2tr(.1, .2, .3)
rpy2tr( [.1, .2, .3] )
rpy2tr( mat([.1, .2, .3]) )
tr = rpy2tr( mat([.1, .2, .3]) )
tr2rpy(tr)

oa2r(v1, v2)
oa2tr(v1, v2)
t = oa2tr(v1, v2)
trnorm(t)

rotvec2r(0.1, mat([1,2,3]) )

# special matrices
skew(.1, .2, .3)
skew( mat([.1, .2 ,.3]) )
m = skew( mat([.1, .2 ,.3]) )
skew(m)
skew( mat([.1, .2 ,.3, .4, .5, .6]) )
m = skew( mat([.1, .2 ,.3, .4, .5, .6]) )
skew(m)
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

