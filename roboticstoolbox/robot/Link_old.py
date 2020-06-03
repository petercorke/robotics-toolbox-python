"""
Link object.
Python implementation by: Luis Fernando Lara Tobar and Peter Corke.
Based on original Robotics Toolbox for Matlab code by Peter Corke.
Permission to use and copy is granted provided that acknowledgement of
the authors is made.
@author: Luis Fernando Lara Tobar and Peter Corke
"""

from numpy import *
from abc import ABC
import copy


class Link(ABC):

    """
    Link abstract class
    """

    def __init__(self, theta, d, a, alpha, offset, jointtype, mdh):
        """
        initialises the Link object
        :param theta:
        :param d:
        :param a:
        :param alpha:
        :param offset:
        :param jointtype: 'R' or 'P' as input. 'R' for Revolute. 'P' for Prismatic.
        :param mdh:
        """
        self.theta = theta
        self.d = d
        self.a = a
        self.alpha = alpha
        self.offset = offset
        self.mdh = mdh

        # we know nothing about the dynamics
        self.m = None
        self.r = None
        self.v = None
        self.I = None
        self.Jm = None
        self.G = None
        self.B = None
        self.Tc = None
        self.qlim = None

        return None

    def __repr__(self):

        if self.convention == Link.LINK_DH:
            conv = 'std'
        else:
            conv = 'mod'

        if self.sigma == 0:
            jtype = 'R'
        else:
            jtype = 'P'

        if self.D == None:
            return "alpha=%f, A=%f, theta=%f jtype: (%c) conv: (%s)" % (self.alpha,
                 self.A, self.theta, jtype, conv)
        elif self.theta == None:
            return "alpha=%f, A=%f, D=%f jtype: (%c) conv: (%s)" % (self.alpha,
                 self.A, self.D, jtype, conv)
        else:
            return "alpha=%f, A=%f, theta=%f, D=%f jtype: (%c) conv: (%s)" % (self.alpha,
                 self.A, self.theta, self.D, jtype, conv)

    # invoked at print
    def __str__(self):
        if self.convention == Link.LINK_DH:
            conv = 'std'
        else:
            conv = 'mod'

        if self.sigma == 0:
            jtype = 'R'
        else:
            jtype = 'P'

        if self.D == None:
            return "alpha = %f\tA = %f\ttheta = %f\t--\tjtype: %c\tconv: (%s)" % (
                self.alpha, self.A, self.theta, jtype, conv)
        elif self.theta == None:
            return "alpha = %f\tA = %f\t--\tD = %f\tjtype: %c\tconv: (%s)" % (
                self.alpha, self.A, self.D, jtype, conv)
        else:
            return "alpha = %f\tA = %f\ttheta = %f\tD=%f\tjtype: %c\tconv: (%s)" % (
                self.alpha, self.A, self.theta, self.D, jtype, conv)


    def display(self):

        print(self)
        print

        if self.m != None:
            print("m:", self.m)
        if self.r != None:
            print("r:", self.r)
        if self.I != None:
            print("I:\n", self.I)
        if self.Jm != None:
            print("Jm:", self.Jm)
        if self.B != None:
            print("B:", self.B)
        if self.Tc != None:
            print("Tc:", self.Tc)
        if self.G != None:
            print("G:", self.G)
        if self.qlim != None:
            print("qlim:\n", self.qlim)

    def copy(self):
        """
        Return copy of this Link
        """
        return copy.copy(self);

    def friction(self, qd):
        """
        Compute friction torque for joint rate C{qd}.
        Depending on fields in the Link object viscous and/or Coulomb friction
        are computed.
        
        @type qd: number
        @param qd: joint rate
        @rtype: number
        @return: joint friction torque
        """
        tau = 0.0
        if isinstance(qd, (ndarray, matrix)):
                qd = qd.flatten().T
        if self.B == None:
            self.B = 0
        tau = self.B * qd
        if self.Tc == None:
            self.Tc = mat([0,0])
        tau = tau + (qd > 0) * self.Tc[0,0] + (qd < 0) * self.Tc[0,1]
        return tau
        
    def nofriction(self, all=False):
        """
        Return a copy of the Link object with friction parameters set to zero.
        
        @type all: boolean
        @param all: if True then also zero viscous friction
        @rtype: Link
        @return: Copy of original Link object with zero friction
        @see: L{robot.nofriction}
        """
        
        l2 = self.copy()

        l2.Tc = array([0, 0])
        if all:
            l2.B = 0
        return l2;


# methods to set kinematic or dynamic parameters

    fields = ["alpha", "A", "theta", "D", "sigma", "offset", "m", "Jm", "G", "B", "convention"];
    
    def __setattr__(self, name, value):
        """
        Set attributes of the Link object
        
            - alpha; scalar
            - A; scalar
            - theta; scalar
            - D; scalar
            - sigma; scalar
            - offset; scalar
            - m; scalar
            - Jm; scalar
            - G; scalar
            - B; scalar
            - r; 3-vector
            - I; 3x3 matrix, 3-vector or 6-vector
            - Tc; scalar or 2-vector
            - qlim; 2-vector
        
        Inertia, I, can be specified as:
            - 3x3 inertia tensor
            - 3-vector, the diagonal of the inertia tensor
            - 6-vector, the unique elements of the inertia tensor [Ixx Iyy Izz Ixy Iyz Ixz]
            
        Coloumb friction, Tc, can be specifed as:
            - scalar, for the symmetric case when Tc- = Tc+
            - 2-vector, the assymetric case [Tc- Tc+]
            
        Joint angle limits, qlim, is a 2-vector giving the lower and upper limits
        of motion.
        """
    
        if value == None:
            self.__dict__[name] = value;
            return;
            
        if name in self.fields:
            # scalar parameter
            if isinstance(value, (ndarray,matrix)) and value.shape != (1,1):
                raise(ValueError, "Scalar required")
            if not isinstance(value, (int,float,int32,float64)):
                raise(ValueError)
            self.__dict__[name] = value

        elif name == "r":
            r = arg2array(value);
            if len(r) != 3:
                raise (ValueError, "matrix required")

            self.__dict__[name] = mat(r)
            
        elif name == "I":
            if isinstance(value, matrix) and value.shape == (3,3):
                self.__dict__[name] = value;
            else:
                v = arg2array(value);
                if len(v) == 3:
                    self.__dict__[name] = mat(diag(v))
                elif len(v) == 6:
                    self.__dict__[name] = mat([
                        [v[0],v[3],v[5]],
                        [v[3],v[1],v[4]],
                        [v[5],v[4],v[2]]])
                else:
                    raise(ValueError, "matrix required")

        elif name == "Tc":
            v = arg2array(value)
            
            if len(v) == 1:
                self.__dict__[name] =  mat([-v[0], v[0]])
            elif len(v) == 2:
                self.__dict__[name] = mat(v)
            else:
                raise ValueError;

        elif name == "qlim":
            v = arg2array(value);
            if len(v) == 2:
                self.__dict__[name] = mat(v);
            else:
                raise ValueError
        else:
            raise(NameError, "Unknown attribute <%s> of link" % name)


#   LINK.islimit(q) return if limit is exceeded: -1, 0, +1
    def islimit(self,q):
        """
        Check if joint limits exceeded.  Returns
            - -1 if C{q} is less than the lower limit
            - 0 if C{q} is within the limits
            - +1 if C{q} is greater than the high limit
        
        @type q: number
        @param q: Joint coordinate
        @rtype: -1, 0, +1
        @return: joint limit status
        """
        if not self.qlim:
            return 0

        return (q > self.qlim[1,0]) - (q < self.qlim[0,0])

    def tr(self, q):
        """
        Compute the transformation matrix for this link.  This is a function
        of kinematic parameters, the kinematic model (DH or MDH) and the joint
        coordinate C{q}.
        
        @type q: number
        @param q: joint coordinate
        @rtype: homogeneous transformation
        @return: Link transform M{A(q)}
        """
        
        an = self.A
        dn = self.D
        theta = self.theta

        if self.sigma == 0:
            theta = q   # revolute
        else:
            dn = q      # prismatic

        sa = sin(self.alpha); ca = cos(self.alpha);
        st = sin(theta); ct = cos(theta);

        if self.convention == Link.LINK_DH:
            # standard
            t =   mat([[ ct,    -st*ca, st*sa,  an*ct],
                    [st,    ct*ca,  -ct*sa, an*st],
                    [0, sa, ca, dn],
                    [0, 0,  0,  1]]);

        else:
            # modified
            t =   mat([[ ct,    -st,    0,  an],
                [st*ca, ct*ca,  -sa,    -sa*dn],
                [st*sa, ct*sa,  ca, ca*dn],
                [0, 0,  0,  1]]);

        return t;