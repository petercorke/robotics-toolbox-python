#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 15:48:52 2020

@author: corkep
"""

import transforms
import numpy as np
from collections import UserList
import argcheck
import math
    
import super_pose as sp

class SO2(sp.SuperPose):
    
    # SO2()  identity matrix
    # SO2(angle, unit)
    # SO2( obj )   # deep copy
    # SO2( np )  # make numpy object
    # SO2( nplist )  # make from list of numpy objects
    
    # constructor needs to take ndarray -> SO2, or list of ndarray -> SO2
    def __init__(self, arg = None, *, unit='rad'):
        super().__init__()  # activate the UserList semantics
        
        if arg is None:
            # empty constructor
            self.data = [np.eye(2)]
        
        elif argcheck.isvector(arg):
            # SO2(value)
            # SO2(list of values)
            self.data = [transforms.rot2(x, unit) for x in argcheck.getvector(arg)]
            
        else:
            super().arghandler(arg)

    @classmethod
    def rand(cls, *, range=[0, 2*math.pi], unit='rad', N=1):
        rand = np.random.uniform(low=range[0], high=range[1], size=N)  # random values in the range
        return cls([transforms.rot2(x) for x in argcheck.getunit(rand, unit)])
    
    @classmethod
    def isvalid(self, x):
        return transforms.isrot2(x, check=True)

    @property
    def T(self):
        return SO2(self.A.T)
    
    def inv(self):
        return SO2(self.A.T)
    
    # for symmetry with other 
    @classmethod
    def R(cls, theta, unit='rad'):
        return SO2([transforms.rot1(x, unit) for x in argcheck.getvector(theta)])
    
    @property
    def angle(self):
        """Returns angle of SO2 object matrices in unit radians"""
        angles = []
        for each_matrix in self:
            angles.append(math.atan2(each_matrix[1, 0], each_matrix[0, 0]))
        # TODO !! Return list be default ?
        if len(angles) == 1:
            return angles[0]
        elif len(angles) > 1:
            return angles

class SE2(SO2):
    # constructor needs to take ndarray -> SO2, or list of ndarray -> SO2
    def __init__(self, x = None, y = None, theta = None, *, unit='rad'):
        super().__init__()  # activate the UserList semantics
        
        if x is None:
            # empty constructor
            self.data = [np.eye(3)]
        
        elif all(map(lambda x: isinstance(x, (int,float)), [x, y, theta])):
            # SE2(x, y, theta)
            self.data = [transforms.trot2(theta, t=[x,y], unit=unit)]
            
        elif argcheck.isvector(x) and argcheck.isvector(y) and argcheck.isvector(theta):
            # SE2(xvec, yvec, tvec)
            xvec = argcheck.getvector(x)
            yvec = argcheck.getvector(y, dim=len(xvec))
            tvec = argcheck.getvector(theta, dim=len(xvec))
            self.data = [transforms.trot2(_t, t=[_x, _y]) for (_x, _y, _t) in zip(xvec, yvec, argcheck.getunit(tvec, unit))]
            
        elif isinstance(x, np.ndarray) and y is None and theta is None:
            assert x.shape[1] == 3, 'array argument must be Nx3'
            self.data = [transforms.trot2(_t, t=[_x, _y], unit=unit) for (_x, _y, _t) in x]
            
        else:
            super().arghandler(x)

    @property
    def T(self):
        raise NotImplemented('transpose is not meaningful for SE3 object')

    @classmethod
    def rand(cls, *, xrange=[-1, 1], yrange=[-1, 1], trange=[0, 2*math.pi], unit='rad', N=1):
        x = np.random.uniform(low=xrange[0], high=xrange[1], size=N)  # random values in the range
        y = np.random.uniform(low=yrange[0], high=yrange[1], size=N)  # random values in the range
        theta = np.random.uniform(low=trange[0], high=trange[1], size=N)  # random values in the range
        return cls([transforms.trot2(t, t=[x,y]) for (t,x,y) in zip(x, y, argcheck.getunit(theta, unit))])
    
    @classmethod
    def isvalid(self, x):
        return transforms.ishom2(x, check=True)

    @property
    def t(self):
        return self.A[:2,2]
    
    @property
    def R(self):
        return SO2(self.A[:2,:2])
    
    def inv(self):
        return SO2(self.A.T)
    ArithmeticError()
    
class SO3(sp.SuperPose):
    
    # SO2()  identity matrix
    # SO2(angle, unit)
    # SO2( obj )   # deep copy
    # SO2( np )  # make numpy object
    # SO2( nplist )  # make from list of numpy objects
    
    # constructor needs to take ndarray -> SO2, or list of ndarray -> SO2
    def __init__(self, arg = None, *, unit='rad'):
        super().__init__()  # activate the UserList semantics
        
        if arg is None:
            # empty constructor
            self.data = [np.eye(3)]
        
            
        else:
            super().arghandler(arg)

    @classmethod
    def rand(cls, *, range=[0, 2*math.pi], unit='rad', N=1):
        rand = np.random.uniform(low=range[0], high=range[1], size=N)  # random values in the range
        return cls([transforms.rot2(x) for x in argcheck.getunit(rand, unit)])
    
    @classmethod
    def isvalid(self, x):
        return transforms.isrot(x, check=True)

    @property
    def T(self):
        return SO3(self.A.T)
    
    def inv(self):
        return SO3(self.A.T)
    
    @property
    def n(self):
        return self.A[:,0]
       
    @property
    def o(self):
        return self.A[:,1]
        
    @property
    def a(self):
        return self.A[:,2]
    
    @classmethod
    def Rx(cls, theta, unit='rad'):
        return cls([transforms.rotx(x, unit) for x in argcheck.getvector(theta)])

    @classmethod
    def Ry(cls, theta, unit='rad'):
        return cls([transforms.roty(x, unit) for x in argcheck.getvector(theta)])

    @classmethod
    def Rz(cls, theta, unit='rad'):
        return cls([transforms.rotz(x, unit) for x in argcheck.getvector(theta)])

    @classmethod
    def rand(cls):
        ran = randint(1, 3)
        if ran == 1:
            rot = transforms.rotx(uniform(0, 360), unit='deg')
            return cls(null=True).__fill([transforms.rotx(uniform(0, 360), unit='deg')])
        elif ran == 2:
            return cls(null=True).__fill([transforms.roty(uniform(0, 360), unit='deg')])
        elif ran == 3:
            return cls(null=True).__fill([transforms.rotz(uniform(0, 360), unit='deg')])
        

    # 
    

    @classmethod
    def eul(cls, angles, unit='rad'):
        return cls(transforms.eul2r(angles, unit=unit))

    @classmethod
    def rpy(cls, angles, order='zyx', unit='rad'):
        return cls(transforms.rpy2r(angles, order=order, unit=unit))

    @classmethod
    def oa(cls, o, a):
        return cls(transforms.oa2r(o, a))

    @classmethod
    def angvec(cls, theta, v, *, unit='rad'):
        return cls(transforms.angvec2r(theta, v, unit=unit))

class SE3(SO3):

    def __init__(self, arg = None, *, unit='rad'):
        super().__init__()  # activate the UserList semantics
        
        if arg is None:
            # empty constructor
            self.data = [np.eye(3)]
        
            
        else:
            super().arghandler(arg)

    @property
    def T(self):
        raise NotImplemented('transpose is not meaningful for SE3 object')
    
    @classmethod
    def rand(cls, *, range=[0, 2*math.pi], unit='rad', N=1):
        rand = np.random.uniform(low=range[0], high=range[1], size=N)  # random values in the range
        return cls([transforms.rot2(x) for x in argcheck.getunit(rand, unit)])
    
    @classmethod
    def isvalid(self, x):
        return transforms.ishom(x, check=True)
    
    @classmethod
    def Rx(cls, theta, unit='rad'):
        return cls([transforms.trotx(x, unit) for x in argcheck.getvector(theta)])

    @classmethod
    def Ry(cls, theta, unit='rad'):
        return cls([transforms.troty(x, unit) for x in argcheck.getvector(theta)])

    @classmethod
    def Rz(cls, theta, unit='rad'):
        return cls([transforms.rtotz(x, unit) for x in argcheck.getvector(theta)])
    
    @classmethod
    def Tx(cls, x):
        return cls(transforms.transl(x, 0, 0))

    @classmethod
    def Ty(cls, y):
        return cls(transforms.transl(0, y, 0))

    @classmethod
    def Tz(cls, z):
        return cls(transforms.transl(0, 0, z))
    
    @classmethod
    def trans(cls, x = None, y = None, z = None):
        return cls(transforms.transl(x, y, z))

    @classmethod
    def rand(cls, *, xrange=[-1, 1], yrange=[-1, 1], zrange=[-1, 1], trange=[0, 2*math.pi], unit='rad', N=1):
        x = np.random.uniform(low=xrange[0], high=xrange[1], size=N)  # random values in the range
        y = np.random.uniform(low=yrange[0], high=yrange[1], size=N)  # random values in the range
        z = np.random.uniform(low=yrange[0], high=zrange[1], size=N)  # random values in the range
        theta = np.random.uniform(low=trange[0], high=trange[1], size=N)  # random values in the range
        return cls([transforms.transl(t, t=[x,y]) for (t,x,y,z) in zip(x, y, z, argcheck.getunit(theta, unit))])

    @classmethod
    def eul(cls, angles, unit='rad'):
        return cls(transforms.eul2tr(angles, unit=unit))

    @classmethod
    def rpy(cls, angles, order='zyx', unit='rad'):
        return cls(transforms.rpy2tr(angles, order=order, unit=unit))

    @classmethod
    def oa(cls, o, a):
        return cls(transforms.oa2tr(o, a))

    @classmethod
    def angvec(cls, theta, v, *, unit='rad'):
        return cls(transforms.angvec2tr(theta, v, unit=unit))


a = SO2(0.2)
b = SO2(a)
print(a+a)
print(a*a)

b = SO2(0.1)
b.append(a)
b.append(a)
b.append(a)
b.append(a)
print(len(a))
print(len(b))
print(b)

c = SO2(0.3)
c.extend(a)
c.extend(b)
print(len(c))

d = SO2(0.4)
d.append(b)
print(len(d))
print(d)


# if __name__ == '__main__':

#     import numpy.testing as nt
        
#     class Test_check(unittest.TestCase):
        
#         def test_unit(self):
            

#print(a)
#print(a*a)
#c = SO2(0)
#
#b = a
#print(len(b))
#b.append(c)
#b.append(c)
#print(len(b))
#print(b)
#print(b)
#print(b[0])
#print(type(a))
#print(type(b))

#arr = [SO2(0), SO2(0.1), SO2(0.2)]
#print(arr)
#b = np.array(arr)
#print(b)
#print('--')
#print(arr[0])
#print(b[1])
#print(b*a)
        