"""
Link object.

@author: Peter Corke
@copyright: Peter Corke
"""

from numpy import *
from utility import *
from transform import *
import copy


class Link:
    """
    LINK create a new LINK object

    A LINK object holds all information related to a robot link such as
    kinematics of the joint
        - alpha; the link twist angle
        - an; the link length
        - theta; the link rotation angle
        - dn; the link offset
        - sigma; 0 for a revolute joint, non-zero for prismatic
        
    rigid-body inertial parameters
        - I; 3x3 inertia matrix about link COG
        - m; link mass
        - r; link COG wrt link coordinate frame 3x1

    motor and transmission parameters
        - B; link viscous friction (motor referred)
        - Tc; link Coulomb friction 1 element if symmetric, else 2
        - G; gear ratio
        - Jm; inertia (motor referred)

    and miscellaneous
        - qlim; joint limit matrix [lower upper] 2 x 1
        - offset; joint coordinate offset
    Handling the different kinematic conventions is now hidden within the LINK
    object.

    Conceivably all sorts of stuff could live in the LINK object such as
    graphical models of links and so on.

    @see: L{Robot}
    """
    
    LINK_DH = 1
    LINK_MDH = 2

    def __init__(self, alpha=0, A=0, theta=0, D=0, sigma=0, convention=LINK_DH):
        """
        L = LINK([alpha A theta D])
        L =LINK([alpha A theta D sigma])
        L =LINK([alpha A theta D sigma offset])
        L =LINK([alpha A theta D], CONVENTION)
        L =LINK([alpha A theta D sigma], CONVENTION)
        L =LINK([alpha A theta D sigma offset], CONVENTION)

        If sigma or offset are not provided they default to zero.  Offset is a
        constant amount added to the joint angle variable before forward kinematics
        and is useful if you want the robot to adopt a 'sensible' pose for zero
        joint angle configuration.

        The optional CONVENTION argument is 'standard' for standard D&H parameters 
        or 'modified' for modified D&H parameters.  If not specified the default
        'standard'.
        """
        self.alpha = alpha
        self.A = A
        self.theta = theta
        self.D = D
        self.sigma = sigma
        self.convention = convention

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

        print self;
        print

        if self.m != None:
            print "m:", self.m
        if self.r != None:
            print "r:", self.r
        if self.I != None:
            print "I:\n", self.I
        if self.Jm != None:
            print "Jm:", self.Jm
        if self.B != None:
            print "B:", self.B
        if self.Tc != None:
            print "Tc:", self.Tc
        if self.G != None:
            print "G:", self.G
        if self.qlim != None:
            print "qlim:\n", self.qlim

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
                raise ValueError, "Scalar required"
            if not isinstance(value, (int,float,int32,float64)):
                raise ValueError;
            self.__dict__[name] = value

        elif name == "r":
            r = arg2array(value);
            if len(r) != 3:
                raise ValueError, "matrix required"

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
                    raise ValueError, "matrix required";

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
            raise NameError, "Unknown attribute <%s> of link" % name


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
