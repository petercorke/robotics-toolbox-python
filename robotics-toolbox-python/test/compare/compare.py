"""
Compare the RNE implementations of Matlab and Python.

We take the Matlab results as the gold standard since they were cross-checked
against a Maple and other implementations a long time ago.

The process:

1. genpath.m creates random q, qd and qdd data and saves to path.dat.  Random values
 in the q, qd and qdd statespace are used with significant velocity and acceleration
 so that errors in velocity and acceleration specific parts of the RNE algorithms will
 be shown up.  There are 60 rows:
    rows 1-20, qd=qdd=0, gravity and friction torques only
    rows 21-40 qdd=0, gravity, friction and centripetal/Coriolis forces
    rows 41-60 all forces.
2. genpath.m creates tau for the Puma560 (DH) and saves to puma560.dat
3. genpath.m creates tau for the Puma560 (MDH) and saves to puma560m.dat
4. compare.py loads path.dat, computes the torques for DH and MDH cases and find the
   difference from the Matlab versions
"""

from robot import *;

print "Compare Python and Matlab RNE implementations"

# load the (q,qd,qdd) data
path = loadtxt('path.dat');

# load the Matlab computed torques
matlab_dh = loadtxt('puma560.dat');

from robot.puma560 import *

tau = rne(p560, path);

diff = matlab_dh - tau;
#print diff
print "RNE DH, error norm =", linalg.norm(diff, 'fro')

#############

# load the Matlab computed torques
matlab_mdh = loadtxt('puma560m.dat');

from robot.puma560akb import *

tau = rne(p560m, path);

diff = matlab_dh - tau;
print diff
print "RNE MDH, error norm =", linalg.norm(diff, 'fro')
