"""
Robot object.

@author: Peter Corke
@copyright: Peter Corke
"""

from numpy import *
from utility import *
from transform import *
import copy
from Link import *

class Robot(object):
    """Robot object.
    Instances of this class represent a robot manipulator
    within the toolbox.
    """
        
    def __init__(self, arg=None, gravity=None, base=None, tool=None, name='', comment='', manuf=''):
        """
        Robot object constructor.  Create a robot from a sequence of Link objects.
        
        Several basic forms exist:
            - Robot()        create a null robot
            - Robot(robot)   create a clone of the robot object
            - Robot(links)   create a robot based on the passed links
            
        Various options can be set using named arguments:
        
            - gravity; gravitational acceleration (default=[0,0,9.81])
            - base; base transform (default 0)
            - tool; tool transform (default 0)
            - name
            - comment
            - manuf
        """

        if isinstance(arg, Robot):
            for k,v in arg.__dict__.items():
                if k == "links":
                    self.__dict__[k] = copy.copy(v);           
                else:
                    self.__dict__[k] = v;           
        elif len(arg) > 1 and isinstance(arg[0], Link):
            self.links = arg;
        else:
            raise AttributeError;

        # fill in default base and gravity direction
        if gravity != None:
            self.gravity = gravity;
        else:
            self.gravity = [0, 0, 9.81];
        
        if base != None:
            self.base = base;
        else:
            self.base = mat(eye(4,4));
        
        if tool != None:
            self.tool = tool;
        else:
            self.tool = mat(eye(4,4));

        if manuf:
            self.manuf = manuf
        if comment:
            self.comment = comment
        if name:
            self.name = name

        #self.handle = [];
        #self.q = [];
        #self.plotopt = {};
        #self.lineopt = {'Color', 'black', 'Linewidth', 4};
        #self.shadowopt = {'Color', 'black', 'Linewidth', 1};

        return None;

    def __str__(self):
        s = 'ROBOT(%s, %s)' % (self.name, self.config());
        return s;
        
    def __repr__(self):
        s = '';
        if self.name:
            s += 'name: %s\n' % (self.name)
        if self.manuf:
            s += 'manufacturer: %s\n' % (self.manuf)
        if self.comment:
            s += 'commment: %s\n' % (self.comment)
        
        for link in self.links:
            s += str(link) + '\n';
        return s;   

    def __mul__(self, r2):
        r = Robot(self);        # clone the robot
        print r
        r.links += r2.links;
        return r;

    def copy(self):
        """
        Return a copy of the Robot object
        """
        return copy.copy(self);
               
    def ismdh(self):
        return self.mdh;
        
    def config(self):
        """
        Return a configuration string, one character per joint, which is
        either R for a revolute joint or P for a prismatic joint.
        For the Puma560 the string is 'RRRRRR', for the Stanford arm it is 'RRPRRR'.
        """
        s = '';
        
        for link in self.links:
            if link.sigma == 0:
                s += 'R';
            else:
                s += 'P';
        return s;

    def nofriction(self, all=False):
        """
        Return a Robot object where all friction parameters are zero.
        Useful to speed up the performance of forward dynamics calculations.
        
        @type all: boolean
        @param all: if True then also zero viscous friction
        @see: L{Link.nofriction}
        """
        r = Robot(self);
        r.name += "-nf";
        newlinks = [];
        for oldlink in self.links:
            newlinks.append( oldlink.nofriction(all) );
        r.links = newlinks;
        return r;
        
    def showlinks(self):
        """
        Shows details of all link parameters for this robot object, including
        inertial parameters.
        """

        count = 1;
        if self.name:
            print 'name: %s'%(self.name)
        if self.manuf:
            print 'manufacturer: %s'%(self.manuf)
        if self.comment:
            print 'commment: %s'%(self.comment)
        for l in self.links:
            print 'Link %d------------------------' % count;
            l.display()
            count += 1;

    def __setattr__(self, name, value):
        """
        Set attributes of the robot object
        
            - robot.name = string (name of this robot)
            - robot.comment = string (user comment)
            - robot.manuf = string (who built it)
            - robot.tool = 4x4 homogeneous tranform
            - robot.base = 4x4 homogeneous tranform
            - robot.gravity = 3-vector  (gx,gy,gz)
        """
        
        if name in ["manuf", "name", "comment"]:
            if not isinstance(value, str):
                raise ValueError, 'must be a string'
            self.__dict__[name] = value;
            
        elif name == "links":
            if not isinstance(value[0], Link):
                raise ValueError, 'not a Link object';
            self.__dict__[name] = value;
            self.__dict__['n'] = len(value);
            # set the robot object mdh status flag
            for link in self.links:
                if link.convention != self.links[0].convention:
                    raise 'robot has mixed D&H link conventions'
            self.__dict__['mdh'] = self.links[0].convention == Link.LINK_MDH;
            
        elif name == "tool":
            if not ishomog(value):
                raise ValueError, 'tool must be a homogeneous transform';
            self.__dict__[name] = value;

        elif name == "gravity":
            v = arg2array(value);
            if len(v) != 3:
                raise ValueError, 'gravity must be a 3-vector';
            self.__dict__[name] = mat(v).T
            
        elif name == "base":
            if not ishomog(value):
                raise ValueError, 'base must be a homogeneous transform';
            self.__dict__[name] = value;
            
        else:
            raise AttributeError;

