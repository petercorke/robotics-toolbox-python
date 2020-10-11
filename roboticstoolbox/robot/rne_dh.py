import numpy as np
from math import sin, cos
from spatialmath import SE3
from spatialmath.base import symbolic as symbolic
from spatialmath.base import t2r, r2t, removesmall, getvector, getmatrix

def rne_dh(self, Q, QD=None, QDD=None, grav=None, fext=None, debug=False, basewrench=False):
    """
    Compute inverse dynamics via recursive Newton-Euler formulation

    :param Q: Joint coordinates
    :param QD: Joint velocity
    :param QDD: Joint acceleration
    :param grav: [description], defaults to None
    :type grav: [type], optional
    :param fext: end-effector wrench, defaults to None
    :type fext: 6-element array-like, optional
    :param debug: print debug information to console, defaults to False
    :type debug: bool, optional
    :param basewrench: compute the base wrench, defaults to False
    :type basewrench: bool, optional
    :raises ValueError: for misshaped inputs
    :return: Joint force/torques
    :rtype: NumPy array

    Recursive Newton-Euler for standard Denavit-Hartenberg notation.

    - ``rne_dh(q, qd, qdd)`` where the arguments have shape (n,) where n is the
      number of robot joints.  The result has shape (n,).
    - ``rne_dh(q, qd, qdd)`` where the arguments have shape (m,n) where n is the
      number of robot joints and where m is the number of steps in the joint
      trajectory.  The result has shape (m,n).
    - ``rne_dh(p)`` where the input is a 1D array ``p`` = [q, qd, qdd] with 
      shape (3n,), and the result has shape (n,).
    - ``rne_dh(p)`` where the input is a 2D array ``p`` = [q, qd, qdd] with
      shape (m,3n) and the result has shape (m,n).

    .. notes::
       - this is a pure Python implementation and slower than the .rne() which
         is written in C.
       - this version supports symbolic model parameters
       - verified against MATLAB code
    """

    n = self.n

    z0 = np.r_[0, 0, 1]

    if grav is None:
        grav = self.gravity  # default gravity from the object
    else:
        grav = getvector(grav, 3)

    if fext is None:
        fext = np.zeros((6,))
    else:
        fext = getvector(fext, 6)

    if debug:
        print('grav', grav)
        print('fext', fext)

    # unpack the joint coordinates and derivatives
    if Q is not None and QD is None and QDD is None:
        # single argument case
        Q = getmatrix(Q, (None, self.n * 3))
        q = Q[:, 0:n]
        qd = Q[:, n:2 * n]
        qdd = Q[:, 2 * n:]

    else:
        # 3 argument case
        q = getmatrix(Q, (None, self.n))
        qd = getmatrix(QD, (None, self.n))
        qdd = getmatrix(QDD, (None, self.n))

    nk = q.shape[0]

    if self.symbolic:
        dtype = 'O'
    else:
        dtype = None

    tau = np.zeros((nk, n), dtype=dtype)
    if basewrench:
        wbase = np.zeros((nk,n), dtype=dtype)

    for k in range(nk):
        # take the k'th row of data
        q_k = q[k, :]
        qd_k = qd[k, :]
        qdd_k = qdd[k, :]

        if debug:
            print('q_k', q_k)
            print('qd_k', qd_k)
            print('qdd_k', qdd_k)
            print()

        # joint vector quantities stored columwise in matrix
        #  m suffix for matrix
        Fm = np.zeros((3,n), dtype=dtype)
        Nm = np.zeros((3,n), dtype=dtype)
        # if robot.issym
        #     pstarm = sym([]);
        # else
        #     pstarm = [];
        pstarm = np.zeros((3,n), dtype=dtype)
        Rm = []
        
        # rotate base velocity and acceleration into L1 frame
        Rb = t2r(self.base.A).T
        w = Rb @ np.zeros((3,))   # base has zero angular velocity
        wd = Rb @ np.zeros((3,))  # base has zero angular acceleration
        vd = Rb @ grav

        # ----------------  initialize some variables -------------------- #

        for j in range(n):
            link = self.links[j]

            # compute the link rotation matrix
            Tj = link.A(q_k[j]).A
            if link.sigma == 0:
                # revolute axis
                d = link.d
            else:
                # prismatic
                d = q_k[j]
            Rm.append(t2r(Tj))

            # compute pstar: 
            #   O_{j-1} to O_j in {j}, negative inverse of link xform
            alpha = link.alpha
            pstarm[:,j] = np.r_[link.a, d * sin(alpha), d * cos(alpha)]

        # -----------------  the forward recursion ----------------------- #

        for j, link in enumerate(self.links):

            Rt = Rm[j].T    # transpose!!
            pstar = pstarm[:,j]
            r = link.r

            # statement order is important here

            if link.isrevolute() == 0:
                # revolute axis
                wd = Rt @ (wd + z0 * qdd_k[j] \
                     + np.cross(w, z0 * qd_k[j]))
                w = Rt @ (w + z0 * qd_k[j])
                vd = np.cross(wd, pstar) \
                     + np.cross(w, np.cross(w, pstar)) \
                     + Rt @ vd
            else:
                # prismatic axis
                w = Rt @ w
                wd = Rt @ wd
                vd = Rt @  (z0 * qdd_k[j] + vd) \
                     + np.cross(wd, pstar) \
                     + 2 * np.cross(w, Rt @ z0 * qd_k[j]) \
                     + np.cross(w, np.cross(w, pstar))

            vhat = np.cross(wd, r) \
                   + np.cross(w, np.cross(w, r)) \
                   + vd
            Fm[:,j] = link.m * vhat
            Nm[:,j] = link.I @ wd + np.cross(w, link.I @ w)

            if debug:
                print('w:     ',  removesmall(w))
                print('wd:    ', removesmall(wd))
                print('vd:    ', removesmall(vd))
                print('vdbar: ', removesmall(vhat))
                print()

        if debug:
            print('Fm\n', Fm)
            print('Nm\n', Nm)

        # -----------------  the forward recursion ----------------------- #

        f = fext[:3]      # force/moments on end of arm
        nn = fext[3:]

        for j in range(n - 1, -1, -1):
            link = self.links[j]
            pstar = pstarm[:,j]
            
            #
            # order of these statements is important, since both
            # nn and f are functions of previous f.
            #
            if j == (n - 1):
                R = np.eye(3)
            else:
                # R = Rm[j+1]   # TODO lose the +1??
                R = Rm[j + 1]

            r = link.r
            nn = R @ (nn + np.cross(R.T @ pstar, f)) \
                    + np.cross(pstar + r, Fm[:,j]) \
                    + Nm[:,j]
            f = R @ f + Fm[:,j]

            if debug:
                print('f: ', removesmall(f))
                print('n: ', removesmall(nn))

            R = Rm[j]
            if link.isrevolute():
                # revolute axis
                t = nn @ (R.T @ z0)
            else:
                # prismatic
                t = f @ (R.T @ z0)
            
            # add joint inertia and friction
            #  no Coulomb friction if model is symbolic
            tau[k, j] = t \
                        + link.G ** 2 * link.Jm * qdd_k[j] \
                        - link.friction(qd_k[j], coloumb=not self.symbolic)
            tau[k,j] = t
            if debug:
                print(f'j={j:}, G={link.G:}, Jm={link.Jm:}, friction={link.friction(qd_k[j]):}')
                print()

        # compute the base wrench and save it
        if basewrench:
            R = Rm[0]
            nn = R @ nn
            f = R @ f
            wbase[k,:] = np.r_[f, nn]

    if self.symbolic:
        # simplify symbolic expressions
        tau = sym.simplify(tau)

    if tau.shape[0] == 1:
        return tau.flatten()
    else:
        return tau

if __name__ == "__main__":

    import roboticstoolbox as rtb

    puma = rtb.models.DH.Puma560()
    for j, link in enumerate(puma):
        print(f'joint {j:}::')
        print(link.dyn(indent=4))
        print()

    tau = rne_dh(puma, puma.qz, puma.qz, puma.qz)
    print(tau)
    tau = rne_dh(puma, np.r_[puma.qz, puma.qz, puma.qz])
    print(tau)
    tau = rne_dh(puma, [0,0,0,0,0,0],  [0,0,0,0,0,0],  [0,0,0,0,0,0])
    print(tau)
    tau = rne_dh(puma, [0,0,0,0,0,0,  0,0,0,0,0,0,  0,0,0,0,0,0])
    print(tau)
