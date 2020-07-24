"""
Link object.
Python implementation by Samuel Drew
"""

from numpy import *
from spatialmath.pose3d import *
import argparse


class Link(list):
    """

    """

    def __init__(self, units='rad', **kwargs):
        """
        Link Create robot link object

        % This the class constructor which has several call signatures.
        %
        % L = Link() is a Link object with default parameters.
        %
        % L = Link(LNK) is a Link object that is a deep copy of the link
        % object LNK and has type Link, even if LNK is a subclass.
        %
        % L = Link(OPTIONS) is a link object with the kinematic and dynamic
        % parameters specified by the key/value pairs.
        %
        % Options::
        % 'theta',TH    joint angle, if not specified joint is revolute
        % 'd',D         joint extension, if not specified joint is prismatic
        % 'a',A         joint offset (default 0)
        % 'alpha',A     joint twist (default 0)
        % 'standard'    defined using standard D&H parameters (default).
        % 'modified'    defined using modified D&H parameters.
        % 'offset',O    joint variable offset (default 0)
        % 'qlim',L      joint limit (default [])
        % 'I',I         link inertia matrix (3x1, 6x1 or 3x3)
        % 'r',R         link centre of gravity (3x1)
        % 'm',M         link mass (1x1)
        % 'G',G         motor gear ratio (default 1)
        % 'B',B         joint friction, motor referenced (default 0)
        % 'Jm',J        motor inertia, motor referenced (default 0)
        % 'Tc',T        Coulomb friction, motor referenced (1x1 or 2x1), (default [0 0])
        % 'revolute'    for a revolute joint (default)
        % 'prismatic'   for a prismatic joint 'p'
        % 'standard'    for standard D&H parameters (default).
        % 'modified'    for modified D&H parameters.
        % 'sym'         consider all parameter values as symbolic not numeric
        %
        % Notes::
        % - It is an error to specify both 'theta' and 'd'
        % - The joint variable, either theta or d, is provided as an argument to
        %   the A() method.
        % - The link inertia matrix (3x3) is symmetric and can be specified by giving
        %   a 3x3 matrix, the diagonal elements [Ixx Iyy Izz], or the moments and products
        %   of inertia [Ixx Iyy Izz Ixy Iyz Ixz].
        % - All friction quantities are referenced to the motor not the load.
        % - Gear ratio is used only to convert motor referenced quantities such as
        %   friction and interia to the link frame.
        %
        % Old syntax::
        % L = Link(DH, OPTIONS) is a link object using the specified kinematic
        % convention  and with parameters:
        %  - DH = [THETA D A ALPHA SIGMA OFFSET] where SIGMA=0 for a revolute and 1
        %    for a prismatic joint; and OFFSET is a constant displacement between the
        %    user joint variable and the value used by the kinematic model.
        %  - DH = [THETA D A ALPHA SIGMA] where OFFSET is zero.
        %  - DH = [THETA D A ALPHA], joint is assumed revolute and OFFSET is zero.
        %
        % Options::
        %
        % 'standard'    for standard D&H parameters (default).
        % 'modified'    for modified D&H parameters.
        % 'revolute'    for a revolute joint, can be abbreviated to 'r' (default)
        % 'prismatic'   for a prismatic joint, can be abbreviated to 'p'
        %
        % Notes::
        % - The parameter D is unused in a revolute joint, it is simply a placeholder
        %   in the vector and the value given is ignored.
        % - The parameter THETA is unused in a prismatic joint, it is simply a placeholder
        %   in the vector and the value given is ignored.
        %
        % Examples::
        % A standard Denavit-Hartenberg link
        %        L3 = Link('d', 0.15005, 'a', 0.0203, 'alpha', -pi/2);
        % since 'theta' is not specified the joint is assumed to be revolute, and
        % since the kinematic convention is not specified it is assumed 'standard'.
        %
        % Using the old syntax
        %        L3 = Link([ 0, 0.15005, 0.0203, -pi/2], 'standard');
        % the flag 'standard' is not strictly necessary but adds clarity.  Only 4 parameters
        % are specified so sigma is assumed to be zero, ie. the joint is revolute.
        %
        %        L3 = Link([ 0, 0.15005, 0.0203, -pi/2, 0], 'standard');
        % the flag 'standard' is not strictly necessary but adds clarity.  5 parameters
        % are specified and sigma is set to zero, ie. the joint is revolute.
        %
        %        L3 = Link([ 0, 0.15005, 0.0203, -pi/2, 1], 'standard');
        % the flag 'standard' is not strictly necessary but adds clarity.  5 parameters
        % are specified and sigma is set to one, ie. the joint is prismatic.
        %
        % For a modified Denavit-Hartenberg revolute joint
        %        L3 = Link([ 0, 0.15005, 0.0203, -pi/2, 0], 'modified');
        %
        % Notes::
        % - Link object is a reference object, a subclass of Handle object.
        % - Link objects can be used in vectors and arrays.
        % - The joint offset is a constant added to the joint angle variable before
        %   forward kinematics and subtracted after inverse kinematics.  It is useful
        %   if you  want the robot to adopt a 'sensible' pose for zero joint angle
        %   configuration.
        % - The link dynamic (inertial and motor) parameters are all set to
        %   zero.  These must be set by explicitly assigning the object
        %   properties: m, r, I, Jm, B, Tc.
        % - The gear ratio is set to 1 by default, meaning that motor friction and
        %   inertia will be considered if they are non-zero.
        %
        % See also Revolute, Prismatic, RevoluteMDH, PrismaticMDH.
        """

        # kinematic parameters

        self.alpha = 0
        self.a = 0
        self.theta = 0
        self.d = 0
        self.jointtype = 'R'
        self.mdh = 0
        self.offset = 0
        self.flip = False
        self.qlim = [-pi, pi]
        self.mdh = False

        """
        Dynamic parameters
        These parameters must be set by the user if dynamics is used
        """
        self.m = 0
        self.r = [0, 0, 0]
        self._I = zeros([3, 3])

        # Dynamic params with default(zero friction)
        self.Jm = 0
        self.G = 1
        self.B = 0
        self.Tc = [0, 0]

        # for every passed argument, check if its a valid attribute and then set it
        for name, value in kwargs.items():

            # convert angular parameters to radians if required
            if name in ['alpha', 'theta'] and units == 'deg':
                value *= pi / 180

            if name in self.__dict__:
                setattr(self, name, value)
            if '_' + name in self.__dict__:
                setattr(self, name, value)

        # convert qlim to radians if required, can only be done after jointtype is known
        if self.jointtype == 'R' and units == 'deg':
            self.qlim = [v * pi / 180 for v in self.qlim]

    def __str__(self):

        if not self.mdh:
            conv = 'std'
        else:
            conv = 'mod'

        if self.jointtype == 'R':
            return "Revolute("+conv+") joint with attributes: " + \
                f"d={self.d:.3g}, a={self.a:.3g}, alpha={self.alpha:.3g}, qlim=({self.qlim[0]:.3g}, {self.qlim[1]:.3g})"
                   #str(self.d)+", a = "+str(self.a)+", alpha = "+str(self.alpha)+", qlim = "+str(self.qlim)
        elif self.jointtype == 'P':
            return "Prismatic("+conv+") joint with attributes: theta = "+\
                   str(self.theta)+", a = "+str(self.a)+", alpha = "+str(self.alpha)+", qlim = "+str(self.qlim)
        else:
            return "jointtype unspecified"

    ## TODO getter/setter for all values that need checking
    @property
    def I(self):
        return self._I
    
    @I.setter
    def I(self, value):
        print("setting I to ", value)
        # do stuff here to handle 3-vector, 6-vector, or 3x3 matrix (check it is symmetric)
        self._I = value

    def type(self):
        """
        Link.type Joint type

        c = L.type() is a character'R' or 'P' depending on whether
        joint is revolute or prismatic respectively.
        TODO If L is a list vector of Link objects return an array of characters in joint order.
        """
        return self.jointtype

    def isrevolute(self):
        """
        Link.isrevolute() Test if joint is revolute
        returns True id joint is revolute
        """
        return self.jointtype == 'R'

    def isprismatic(self):
        """
        Link.isprismatic() Test if joint is prismatic
        returns True id joint is prismatic
        """
        return self.jointtype == 'P'

    def A(self, q):
        """
        Link.A Link transform matrix

        T = L.A(q) is an SE3 object representing the transformation between link
        frames when the link variable q which is either the Denavit-Hartenberg
        parameter theta (revolute) or d (prismatic).  For:
         - standard DH parameters, this is from the previous frame to the current.
         - modified DH parameters, this is from the current frame to the next.

        Notes::
        - For a revolute joint the THETA parameter of the link is ignored, and Q used instead.
        - For a prismatic joint the D parameter of the link is ignored, and Q used instead.
        - The link offset parameter is added to Q before computation of the transformation matrix.
        """
        sa = sin(self.alpha)
        ca = cos(self.alpha)
        if self.flip:
            q = -q + self.offset
        else:
            q = q + self.offset
        if self.isrevolute():
            # revolute
            st = sin(q)
            ct = cos(q)
            d = self.d
        else:
            # prismatic
            st = sin(self.theta)
            ct = cos(self.theta)
            d = q

        if not self.mdh:
            # standard DH

            T = array([[ct, -st*ca, st*sa, self.a*ct],
                       [st, ct*ca, -ct*sa, self.a*st],
                       [0, sa, ca, d],
                       [0, 0, 0, 1]])
        else:
            # modified DH

            T = array([[ct, -st, 0, self.a],
                       [st*ca, ct*ca, -sa, -sa*d],
                       [st*sa, ct*sa, ca, ca*d],
                       [0, 0, 0, 1]])

        return SE3(T)

class RevoluteDH(Link):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class PrismaticDH(Link):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
