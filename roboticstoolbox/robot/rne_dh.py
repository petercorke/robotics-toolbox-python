
%SERIALLINK.RNE_DH Compute inverse dynamics via recursive Newton-Euler formulation
%
% Recursive Newton-Euler for standard Denavit-Hartenberg notation.  Is invoked by
% R.RNE().
%
% See also SERIALLINK.RNE.

%
% verified against MAPLE code, which is verified by examples
%


def rne_dh(self, q_, qd_=None, qdd_=None, grav=None, fext=None):

    n = robot.n

    z0 = np.r_[0, 0, 1]
    if grav is None:
        grav = self.gravity  # default gravity from the object
    if fext is None:
        fext = np.zeros((6,))

    # check that robot object has dynamic parameters for each link
    if all([link.m == 0 for link in self.links]):
        raise ValueError('dynamic parameters likely not initialized')

    # Set debug to:
    #   0 no messages
    #   1 display results of forward and backward recursions
    #   2 display print R and p*
    debug = 1

    # check the arguments
    if len(q_.shape) == 1:
        if q.shape[0] == 3 * n:
            qdd = q_[2n:].reshape((n,))
            qd = q_[n:2n].reshape((n,))
            q = q_[0:n].reshape((n,))
        else:
            if q_.shape[0] != n:
                raise ValueError('q must have n elements')

    elif len(q.shape == 2:
        if q.shape[1] == 3 * n:
            _qdd = q[:, 2 * n:]
            _qd = q[:, n:2 * n]
            _q = q[:, 0:n]
        else:
            if q.shape[1] != n:
                raise ValueError('q must have n columns')
            _q = 

    else:
        raise ValueError('q must have 1 or 2 dimensions')

    if q.shape != qd.shape:
        raise ValueError('qd must be same shape as q')
    if q.shape != qdd.shape:
        raise ValueError('qdd must be same shape as q')

    
    # if robot.issym || any([isa(Q,'sym'), isa(Qd,'sym'), isa(Qdd,'sym')])
    #     tau = zeros(np,n, 'sym');
    # else
    #     tau = zeros(np,n);
    # end

    np = q.shape[0]

    for p in range(np):
        q = _q[p, :]
        qd = _qd[p, :]
        qdd = _qdd[p, :]

        Fm = []
        Nm = []
        # if robot.issym
        #     pstarm = sym([]);
        # else
        #     pstarm = [];
        pstarm = []
        Rm = []
        
        # rotate base velocity and acceleration into L1 frame
        Rb = t2r(self.base).T
        w = Rb @ zeros(3,1);
        wd = Rb @ np.zeros(3,1);
        vd = Rb @ grav

        #
        # init some variables, compute the link rotation matrices
        #
        for j=1:n
            link = robot.links[j]
            Tj = link.A(q[j])
            if link.sigma == 0:
                # revolute axis
                d = link.d
            else:
                # prismatic
                d = q[j]

            alpha = link.alpha
            # O_{j-1} to O_j in {j}, negative inverse of link xform
            pstar = np.r_[link.a, d * sin(alpha), d * cos(alpha)]

            pstarm.append(pstar)
            Rm.append(t2r(Tj))
            if debug>1
                print(Rm[j])
                Pstarm(:,j).'

    #
    #  the forward recursion
    #
    for j, link in enumerate(self.links):

        Rt = Rm{j}.';    % transpose!!
        pstar = pstarm(:,j);
        r = link.r;

        #
        # statement order is important here
        #
        if link.sigma == 0:
            # revolute axis
            wd = Rt @ (wd + z0 * qdd[j] + \
                cross(w, z0 * qd[j]))
            w = Rt * (w + z0 * qd[j])
            # v = cross(w,pstar) + Rt*v;
            vd = np.cross(wd, pstar) + \
                np.cross(w, np.cross(w, pstar)) + Rt * vd
    
        else:
            # prismatic axis
            w = Rt @ w
            wd = Rt @ wd
            vd = Rt @  (z0 * qdd[j] + vd) + \
                cross(wd,pstar) + \
                2 * cross(w, Rt @ z0 * qd[j]) + \
                cross(w, cross(w, pstar))

        %whos
        vhat = cross(wd, r) + \
            cross(w, cross(w, r)) + vd
        F = link.m * vhat
        N = link.I * wd + cross(w, link.I * w)
        Fm = [Fm F]
        Nm = [Nm N]

        if debug
            fprintf('w: '); disp( w)
            fprintf('\nwd: '); disp( wd)
            fprintf('\nvd: '); disp( vd)
            fprintf('\nvdbar: '); disp( vhat)
            fprintf('\n');


    #
    #  the backward recursion
    #

        fext = fext
        f = fext[0:3]      % force/moments on end of arm
        nn = fext[3:]

        for j in range(n-1, -1, -1):
            link = robot.links[j]
            pstar = pstarm(:,j)
            
            #
            # order of these statements is important, since both
            # nn and f are functions of previous f.
            #
            if j == n:
                R = np.eye(3)
            else:
                R = Rm[j+1]
            r = link.r
            nn = R * (nn + cross(R.T @ pstar, f)) + \
                 cross(pstar + r, Fm[:,j]) + \
                 Nm(:,j)
            f = R @ f + Fm(:,j)
            if debug
                fprintf('f: '); disp( f)
                fprintf('\nn: '); disp( nn)
                fprintf('\n');

            R = Rm[j]
            if link.sigma == 0:
                # revolute axis
                t = nn.' * (R.T @ z0) + \
                    link.G ** 2 * link.Jm * qdd[j] - \
                    link.friction(qd[j])
                tau(p,j) = t
             else:
                # prismatic
                t = f.'*(R.'*z0) + \
                    link.G ** 2 * link.Jm * qdd[j] - \
                    link.friction(qd[j])
                tau(p,j) = t

        % this last bit needs work/testing
        R = Rm{1};
        nn = R*(nn);
        f = R*f;
        wbase = [f; nn];

    
    if isa(tau, 'sym')
        tau = simplify(tau);
    end
