"""
Robot dynamics operations.

@author: Peter Corke
@copyright: Peter Corke
"""

from numpy import *
from utility import *
from transform import *
import jacobian as Jac
from numpy.linalg import inv

def accel(robot, *args):
    """
    Compute manipulator forward dynamics, the joint accelerations that result
    from applying the actuator torque to the manipulator robot in state q and qd.

       - qdd = accel(robot, q, qd, torque)
       - qdd = accel(robot, [q qd torque])

    Uses the method 1 of Walker and Orin to compute the forward dynamics.
    This form is useful for simulation of manipulator dynamics, in
    conjunction with a numerical integration function.
    
    @type robot: Robot object, n-axes
    @param robot: The robot
    @rtype: n-vector
    @return: Joint coordinate acceleration
    @bug: Should handle the case of q, qd are matrices
    @see: L{rne}, L{robot}
    """
    n = robot.n
    if len(args) == 1:
        q = mat(args[0])[0,0:n]
        qd = mat(args[0])[0,n:2*n]
        torque = mat(args[0])[0,2*n:3*n]
    else:
        q = mat(args[0])
        if numcols(q) == robot.n:
            q = q.flatten().T
            qd = mat(args[1]).flatten().T
        torque = mat(args[2])
    
    # Compute current manipulator inertia
    # Torques resulting from unit acceleration of each joint with
    # no gravity

    M = rne(robot, mat(ones((n,1)))*q.T, zeros((n,n)), eye(n), [[0],[0],[0]])

    # Compute gravity and coriolis torque
    # torques resulting from zero acceleration at given velocity &
    # with gravity acting

    tau = rne(robot, q.T, qd.T, zeros((1,n)))
    qdd = inv(M) * (torque.flatten().T - tau.T)
    return qdd.T

def coriolis(robot, q, qd):
    """
    Compute the manipulator Coriolis matrix

    c = coriolis(robot, q, qd)

    Returns the n-element Coriolis/centripetal torque vector at the specified 
    pose and velocity.

    If C{q} and C{qd} are row vectors, the result is a row vector 
    of joint torques.
    If C{q} and C{qd} are matrices, each row is interpretted as a joint state 
    vector, and the result is a matrix each row being the corresponding joint torques.

    @type q: M{m x n} matrix
    @type q: Joint coordinate
    @type qd: M{m x n} matrix
    @type qd: Joint coordinate velocity
    @type robot: Robot object, n-axes
    @param robot: The robot
    @rtype: M{m x n} matrix
    @return: Joint coordinate acceleration
    @see: L{robot}, L{rne}, L{itorque}, L{gravload}
    """
    return rne(robot, q, qd, zeros(shape(q)), [[0],[0],[0]])

def inertia(robot, q):
    """
    Compute the manipulator inertia matrix

    inertia(robot, q)

    Returns the M{n x n} symmetric inertia matrix which relates joint torque 
    to joint acceleration for the robot in pose C{q}.
    
    If C{q} is a matrix then return a list of inertia matrices, corresponding
    to the joint coordinates from rows of C{q}.

    @type q: M{m x n} matrix
    @type q: Joint coordinate
    @type robot: Robot object, n-axes
    @param robot: The robot
    @rtype: m-list of M{n x n} matrices
    @return: List of inertia matrices
    @see: L{cinertia}, L{rne}, L{robot}
    """
    n = robot.n
    q = mat(q)
    if numrows(q) > 1:
        ilist = [];
        for row in q:
            I = rne(robot, ones((n, 1))*row, zeros((n,n)), eye(n), [[0],[0],[0]])
            ilist.append( I )
        return ilist
    else:
        return rne(robot, ones((n, 1))*q, zeros((n,n)), eye(n), [[0],[0],[0]])

def cinertia(robot, q):
    """
    Compute the Cartesian (operational space) manipulator inertia matrix

    m = cinertia(robot, q)

    Return the M{6 x 6} inertia matrix which relates Cartesian force/torque to 
    Cartesian acceleration.

    @type q: n-vector
    @type q: Joint coordinate
    @type robot: Robot object, n-axes
    @param robot: The robot
    @rtype: M{6 x 6} matrix
    @return: Cartesian inertia matrix
    @bug: Should handle the case of q as a matrix
    @see: L{inertia}, L{rne}, L{robot}
    """
    
    J = Jac.jacob0(robot, q)
    Ji = inv(J)
    M = inertia(robot, q)
    return Ji.T*M*Ji
    
def gravload(robot, q, gravity=None):
    """
    Compute the gravity loading on manipulator joints

    taug = gravload(robot, q)
    taug = gravload(robot, q, grav)

    Compute the joint gravity loading for the manipulator C{robot} in the
    configuration C{q}.

    If C{q} is a row vector, the result is a row vector of joint torques.
    If C{q} is a matrix, each row is interpretted as a joint state vector, and
    the result is a matrix each row being the corresponding joint torques.

    Gravity vector can be given explicitly using the named gravity keyword, otherwise
    it defaults to the value of the C{robot} object.

    @type q: M{m x n} matrix
    @type q: Joint coordinate
    @type grav: 3-vector
    @type grav: Gravitation acceleration, overrides C{robot} (optional)
    @type robot: Robot object, n-axes
    @param robot: The robot
    @rtype: M{m x n} matrix
    @return: Gravity torque
    @see: L{rne}, L{robot}
    """
    q = mat(q)
    if numcols(q) != robot.n:
        raise 'Insuficient columns in q'
    if gravity == None:
        tg = rne(robot, q, zeros(shape(q)), zeros(shape(q)))
    else:
        tg = rne(robot, q, zeros(shape(q)), zeros(shape(q)), gravity=gravity)
    return tg


def rne(robot, *args, **options):
    """
    Compute inverse dynamics via recursive Newton-Euler formulation.

    tau = rne(robot, q, qd, qdd)
    tau = rne(robot, [q qd qdd])

    Returns the joint torque required to achieve the specified joint position,
    velocity and acceleration state.

    Options
    =======
    
        One or more options can be provided as named arguments.
        
        Gravity
        -------
        Gravity vector is an attribute of the robot object but this may be 
        overriden by providing a gravity acceleration vector [gx gy gz] using the
        named argument gravity

        tau = rne(robot, ..., gravity=[gx, gy, gz])


        External force/moment
        ---------------------
        An external force/moment acting on the end of the manipulator may also be
        specified by a 6-element vector [Fx Fy Fz Mx My Mz].

        tau = rne(robot, ..., fext=[fx,fy,fz, mx,my,mz])

        where Q, QD and QDD are row vectors of the manipulator state; pos, vel, 
        and accel.
        
        Debug
        -----

        tau = rne(robot, ..., debug=n)
        
        Use the debug named argument to enable
            - 0 no messages
            - 1 display results of forward and backward recursions
            - 2 display print R and p*

    @see: L{robot}, L{accel}, L{inertia}, L{coriolis}, L{gravload}
    @note: verified against Matlab toolbox
    """

    n = robot.n
    
    a1 = mat(args[0])
    if numcols(a1)== 3*n:
        # single state parameter: [q qd qdd]
        q = a1[:,0:n]
        qd = a1[:,n:2*n]
        qdd = a1[:,2*n:3*n]
    else:
        # three state parameters: q, qd, qdd
        np = numrows(a1)
        q = a1
        qd = mat(args[1])
        qdd = mat(args[2])
        if numcols(a1) != n or numcols(qd) != n or numcols(qdd) != n or numrows(qd) != np or numrows(qdd) != np:
            error('inconsistant sizes of q, qd, qdd')

    # process options: gravity,  fext, debug
    debug = 0;
    gravity = robot.gravity;
    fext = mat(zeros((6,1)))
    
    for k,v in options.items():
        if k == "gravity":
            gravity = arg2array(v);
            if not isvec(gravity, 3):
                error('gravity must be 3-vector')
            gravity = mat(gravity).T
        elif k == "fext":
            fext = arg2array(v);
            if not isvec(fext, 6):
                error('fext must be 6-vector')
            fext = mat(fext).T
        elif k == "debug":
            debug = v
        else:
            error('unknown option')
        
    if robot.ismdh():
        tau = _rne_mdh(robot, q, qd, qdd, gravity, fext, debug=debug)
    else:
        tau = _rne_dh(robot, q, qd, qdd, gravity, fext, debug=debug)
    return tau



def _rne_dh(robot, Q, Qd, Qdd, grav, fext, debug=0):
  
    z0 = mat([0,0,1]).T
    n = robot.n
    np = numrows(Q)
    tau = mat(zeros((np,n)))
   
    for p in range(0,np):
        q = Q[p,:].T
        qd = Qd[p,:].T
        qdd = Qdd[p,:].T

        Fm = []
        Nm = []
        pstarm = []
        Rm = []
        w = mat(zeros((3,1)))
        wd = mat(zeros((3,1)))
        v = mat(zeros((3,1)))
        vd = grav

        #
        # init some variables, compute the link rotation matrices
        #
        for j in range(0,n):
            link = robot.links[j]
            Tj = link.tr(q[j,0])
            Rm.append(t2r(Tj))
            if link.sigma == 0:
                D = link.D
            else:
                D = q[j,0]
            alpha = link.alpha
            pstarm.append(mat([[link.A],[D*sin(alpha)],[D*cos(alpha)]]))
            if debug > 1:
                print 'Rm:'
                print Rm[j]
                print 'pstarm:'
                print pstarm[j].T
        
        #
        # the forward recursion
        #
        for j in range(0,n):
            link = robot.links[j]
            Rt = Rm[j].T   # transpose!!
            pstar = pstarm[j]
            r = link.r

            #
            # statement orden is important here
            #
            
            if link.sigma == 0:
                # revolute axis
                wd = Rt*(wd + z0*qdd[j,0]+crossp(w,z0*qd[j,0]))
                w = Rt*(w + z0*qd[j,0])
                #v = crossp(w,pstar) + Rt*v
                vd = crossp(wd,pstar) + crossp(w,crossp(w,pstar)) + Rt*vd
            
            else:
                # prismatic axis
                w = Rt*w
                wd = Rt*wd
                vd = Rt*(z0*qdd[j,0]+vd) + crossp(wd,pstar) + 2*crossp(w,Rt*z0*qd[j,0]) + crossp(w,crossp(w,pstar))

            vhat = crossp(wd,r) + crossp(w,crossp(w,r)) + vd
            F = link.m*vhat
            N = link.I*wd + crossp(w,link.I*w)
            Fm.append(F)
            Nm.append(N)

            if debug:
                print
                print "w:\t%f\t%f\t%f"%(w[0,0], w[1,0], w[2,0])
                print "wd:\t%f\t%f\t%f"%(wd[0,0], wd[1,0], wd[2,0])
                print "vd:\t%f\t%f\t%f"%(vd[0,0], vd[1,0], vd[2,0])
                print "vdbar:\t%f\t%f\t%f"%(vhat[0,0], vhat[1,0], vhat[2,0])
                print

        #
        # the backward recursion
        #

        fext = fext.flatten().T
        f = fext[0:3,0]           # force/moments on end of arm
        nn = fext[3:6,0]
        
        for j in range(n-1,-1,-1):
            link = robot.links[j]
            pstar = pstarm[j]

            #
            # order of these statements is important, since both
            # mm and f are functions of previous f.
            #

            if j == n-1:
                R = mat(eye(3,3))
            else:
                R = Rm[j+1]
            r = link.r.T

            nn = R*(nn + crossp(R.T*pstar,f)) + crossp(pstar+r,Fm[j]) + Nm[j]
            f = R*f + Fm[j]
            if debug:
                print
                print "f:\t%f\t%f\t%f"%(f[0,0],f[1,0],f[2,0])
                print "nn:\t%f\t%f\t%f"%(nn[0,0],nn[1,0],nn[2,0])
                print

            R = Rm[j]
            if link.sigma == 0:
                # revolute
                tau[p,j] = nn.T*(R.T*z0) + link.G**2*link.Jm*qdd[j,0] + link.G*link.friction(qd[j,0])
            else:
                # prismatic
                tau[p,j] = f.T*(R.T*z0) + link.G**2*link.Jm*qdd[j,0] + link.G*link.friction(qd[j,0])
    return tau


def _rne_mdh(robot, Q, Qd, Qdd, grav, fext, debug=0):

    z0 = mat([0,0,1]).T
    n = robot.n
    np = numrows(Q)
    tau = mat(zeros((np,n)))
        
    for p in range(0,np):
        q = Q[p,:].T
        qd = Qd[p,:].T
        qdd = Qdd[p,:].T

        Fm = []
        Nm = []
        Pm = []
        Rm = []
        w = mat(zeros((3,1)))
        wd = mat(zeros((3,1)))
        v = mat(zeros((3,1)))
        vd = grav.flatten().T

        #
        # init some variables, compute the link rotation matrices
        #
        for j in range(0,n):
            link = robot.links[j]
            Tj = link.tr(q[j,0])
            Rm.append(t2r(Tj))
            if link.sigma == 0:
                D = link.D
            else:
                D = q[j,0]
            alpha = link.alpha
            Pm.append(mat([[link.A],[-D*sin(alpha)],[D*cos(alpha)]])) # (i-1) P i
            if debug > 1:
                print 'Rm:'
                print Rm[j]
                print 'Pm:'
                print Pm[j].T

        #
        # the forward recursion
        #
        for j in range(0,n):
            link = robot.links[j]
            R = Rm[j].T   # transpose!!
            P = Pm[j]
            Pc = link.r

            #
            # trailing underscore means new value
            #
            
            if link.sigma == 0:
                # revolute axis
                w_ = R*w + z0*qd[j,0]
                wd_ = R*wd + crossp(R*w,z0*qd[j,0]) + z0*qdd[j,0]
                #v = crossp(w,P) + R*v
                vd_ = R*(crossp(wd,P) + crossp(w,crossp(w,P)) + vd)
            
            else:
                # prismatic axis
                w_ = R*w
                wd_ = R*wd
                #v = R*(z0*qd[j,0] + v) + crossp(w,P)
                vd_ = R*(crossp(wd,P) + crossp(w,crossp(w,P)) + vd) + 2*crossp(R*w,z0*qd[j,0]) + z0*qdd[j,0]

            # update variables
            w = w_
            wd = wd_
            vd = vd_

            vdC = crossp(wd,Pc) + crossp(w,crossp(w,Pc)) + vd
            F = link.m*vdC
            N = link.I*wd + crossp(w,link.I*w)
            Fm.append(F)
            Nm.append(N)

            if debug:
                print
                print "w:\t%f\t%f\t%f"%(w[0,0], w[1,0], w[2,0])
                print "wd:\t%f\t%f\t%f"%(wd[0,0], wd[1,0], wd[2,0])
                print "vd:\t%f\t%f\t%f"%(vd[0,0], vd[1,0], vd[2,0])
                print "vdbar:\t%f\t%f\t%f"%(vdC[0,0], vdC[1,0], vdC[2,0])
                print

        #
        # The backward recursion
        #

        fext = fext.flatten().T
        f = fext[0:3,0]           # force/moments on end of arm
        nn = fext[3:6,0]

        for j in range(n-1,-1,-1):
            link = robot.links[j]

            #
            # order of these statements is important, since both
            # mm and f are functions of previous f.
            #

            if j == n-1:
                R = mat(eye(3,3))
                P = mat([[0],[0],[0]])
            else:
                R = Rm[j+1]
                P = Pm[j+1]   # i/P/(i+1)
            Pc = link.r
            f_ = R*f + Fm[j]
            nn_ = Nm[j] + R*nn + crossp(Pc,Fm[j]) + crossp(P,R*f)
            f = f_
            nn = nn_
            if debug:
                print
                print "f:\t%f\t%f\t%f"%(f[0,0],f[1,0],f[2,0])
                print "nn:\t%f\t%f\t%f"%(nn[0,0],nn[1,0],nn[2,0])
                print
            
            if link.sigma == 0:
                # revolute
                tau[p,j] = nn.T*z0 + link.G**2*link.Jm*qdd[j,0] + link.G*link.friction(qd[j,0])
            else:
                # prismatic
                tau[p,j] = f.T*z0 + link.G**2*link.Jm*qdd[j,0] + link.G*link.friction(qd[j,0])
    return tau

