"""
Link object.
Python implementation by: Luis Fernando Lara Tobar and Peter Corke.
Based on original Robotics Toolbox for Matlab code by Peter Corke.
Permission to use and copy is granted provided that acknowledgement of
the authors is made.
@author: Luis Fernando Lara Tobar and Peter Corke
"""

from numpy import *
from spatialmath.pose3d import *
import argparse


class Link(list):
    """

    """

    def __init__(self, *argv):
        """
        %Link Create robot link object
        %
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

        print("Link constructor called with ", len(argv), " arguments:", argv)
        if len(argv) == 0:
            print("Creating Link class object with default parameters")
            """
            Create an 'empty' Link object
            This call signature is needed to support arrays of Links
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
            self.qlim = []

            """
            Dynamic parameters
            These parameters must be set by the user if dynamics is used
            """
            self.m = 0
            self.r = [0, 0, 0]
            self.I = zeros([3, 3])

            # Dynamic params with default(zero friction)
            self.Jm = 0
            self.G = 1
            self.B = 0
            self.Tc = [0, 0]

        elif len(argv) == 1 and isinstance(argv, Link):
            # Clone the passed Link object
            self = argv

        else:
            # format input into argparse
            argstr = ""
            known = ['theta', 'a', 'd', 'alpha', 'G', 'B', 'Tc', 'Jm', 'I', 'm', 'r',
                     'offset', 'qlim', 'type', 'convention', 'sym', 'flip']
            for arg in argv:
                if arg in known:
                    argstr += "--" + arg + " "
                else:
                    argstr += str(arg) + " "

            # Create a new Link based on parameters
            # parse all possible options
            parser = argparse.ArgumentParser()
            parser.add_argument('--theta', help="joint angle, if not specified joint is revolute",
                                type=float, default=0)
            parser.add_argument("--a", help="joint offset (default 0)",
                                type=float, default=0)
            parser.add_argument("--d", help="joint extension, if not specified joint is prismatic",
                                type=float, default=0)
            parser.add_argument("--alpha", help="joint twist (default 0)",
                                type=float, default=0)
            parser.add_argument("--G", help="motor gear ratio (default 1)",
                                type=float, default=0)
            parser.add_argument("--B", help="joint friction, motor referenced (default 0)",
                                type=float, default=0)
            parser.add_argument("--Tc", help="Coulomb friction, motor referenced (1x1 or 2x1), (default [0, 0])",
                                type=list, default=[0, 0])
            parser.add_argument("--Jm", help="motor inertia, motor referenced (default 0))",
                                type=float, default=0)
            parser.add_argument("--I", help="link inertia matrix (3x1, 6x1 or 3x3)",
                                type=ndarray, default=zeros([3, 3]))
            parser.add_argument("--m", help="link mass (1x1)",
                                type=float, default=0)
            parser.add_argument("--r", help="link centre of gravity (3x1)",
                                type=list, default=[0, 0, 0])
            parser.add_argument("--offset", help="joint variable offset (default 0)",
                                type=float, default=0)
            parser.add_argument("--qlim", help="joint limit",
                                type=list, default=[-pi/2, pi/2])
            parser.add_argument("--type", help="joint type, 'revolute', 'prismatic' or 'fixed'",
                                choices=['', 'revolute', 'prismatic', 'fixed'], default='')
            parser.add_argument("--convention", help="D&h parameters, 'standard' or 'modified'",
                                choices=['standard', 'modified'], default='standard')
            parser.add_argument("--sym", help="consider all parameter values as symbolic not numeric'",
                                action="store_true")
            parser.add_argument("--flip", help="TODO add help for 'flip'",
                                action="store_true")
            (opt, args) = parser.parse_known_args(argstr.split())

            if not args:

                assert opt.d == 0 or opt.theta == 0, "Bad argument, cannot specify both d and theta"

                if opt.type == 'revolute':
                    print('Revolute joint')
                    self.jointtype = 'R'
                    assert opt.theta == 0, "Bad argument, cannot specify 'theta' for revolute joint"
                elif opt.type == 'prismatic':
                    print('Prismatic joint')
                    self.jointtype = 'P'
                    assert opt.d == 0, "Bad argument, cannot specify 'd' for prismatic joint"

                if opt.theta != 0:
                    # constant value of theta means it must be prismatic
                    self.theta = opt.theta
                    self.jointtype = 'P'
                    print('Prismatic joint, theta =', opt.theta)
                if opt.d != 0:
                    # constant value of d means it must be revolute
                    self.d = opt.d
                    self.jointtype = 'R'
                    print('Revolute joint, d =', opt.d)

                self.a = opt.a
                self.alpha = opt.alpha

                self.offset = opt.offset
                self.flip = opt.flip
                self.qlim = argcheck.getvector(opt.qlim)

                self.m = opt.m
                self.r = opt.r
                self.I = opt.I
                self.Jm = opt.Jm
                self.G = opt.G
                self.B = opt.B
                self.Tc = opt.Tc
                self.mdh = ['standard', 'modified'].index(opt.convention)

            else:
                """
                This is the old call format, where all parameters are given by
                a vector containing kinematic-only, or kinematic plus dynamic
                parameters
                
                eg. L3 = Link([0, 0.15005, 0.0203, -pi/2, 0], 'standard')
                """
                print("old format")
                dh = argv[0]
                assert len(dh) >= 4, "Bad argument, must provide params (theta d a alpha)"

                # set the kinematic parameters
                self.theta = dh[0]
                self.d = dh[1]
                self.a = dh[2]
                self.alpha = dh[3]

                self.jointtype = 'R'
                self.offset = 0
                self.flip = False
                self.mdh = 0

                # optionally set jointtype and offset
                if len(dh) >= 5:
                    if dh[4] == 1:
                        self.jointtype = 'P'
                if len(dh) == 6:
                    self.offset = dh[5]

                else:
                    # we know nothing about the dynamics
                    self.m = 0
                    self.r = [0, 0, 0]
                    self.I = zeros([3, 3])
                    self.Jm = 0
                    self.G = 1
                    self.B = 0
                    self.Tc = [0, 0]
                    self.qlim = 0
                if argv[1] == 'modified':
                    self.mdh = 1
                else:
                    self.mdh = 0

    def __repr__(self):

        if not self.mdh:
            conv = 'std'
        else:
            conv = 'mod'

        if self.jointtype == 'R':
            return "Revolute("+conv+") joint with attributes: d = "+\
                   str(self.d)+", a = "+str(self.a)+", alpha = "+str(self.alpha)
        elif self.jointtype == 'P':
            return "Prismatic("+conv+") joint with attributes: theta = "+\
                   str(self.theta)+", a = "+str(self.a)+", alpha = "+str(self.alpha)
        else:
            return "jointtype unspecified"

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

