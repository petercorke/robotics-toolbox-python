# Author: Aditya Dua
# 28 January, 2018

from collections import UserList
import math
import numpy as np

import spatialmath.base.transforms as tr
import spatialmath.base.quaternion as quat
import spatialmath.base.argcheck as argcheck

class Quaternion(UserList):
    
    def __init__(self, s=None, v=None, check=True, norm=True):
        """
        A quaternion is a compact method of representing a 3D rotation that has
        computational advantages including speed and numerical robustness.
        A quaternion has 2 parts, a scalar s, and a vector v and is typically written::
        q = s <vx vy vz>
        A unit quaternion is one for which M{s^2+vx^2+vy^2+vz^2 = 1}.
        A quaternion can be considered as a rotation about a vector in space where
        q = cos (theta/2) sin(theta/2) <vx vy vz>
        where <vx vy vz> is a unit vector.
        :param s: scalar
        :param v: vector
        """
        if s is None and v is None:
            self.data = [ quat.qone() ]
            
        elif argcheck.isscalar(s) and argcheck.isvector(v,3):
            self.data = [ np.r_[s, argcheck.getvector(v)] ]
            
        elif argcheck.isvector(s,4):
            self.data = [ argcheck.getvector(s) ]
            
        elif type(s) is list:
            if check:
                assert argcheck.isvectorlist(s,4), 'list must comprise 4-vectors'
            self.data = s
        
        elif isinstance(s, np.ndarray) and s.shape[1] == 4:
            self.data = [x for x in s]
            
        else:
            raise ValueError('bad argument to Quaternion constructor')
            
    def append(self, x):
        print('in append method')
        if not type(self) == type(x):
            raise ValueError("cant append different type of pose object")
        if len(x) > 1:
            raise ValueError("cant append a pose sequence - use extend")
        super().append(x.A)
        
    @property
    def A(self):
        # get the underlying numpy array
        if len(self.data) == 1:
            return self.data[0]
        else:
            return self.data

    def __getitem__(self, i):
        print('getitem', i)
        #return self.__class__(self.data[i])
        return self.__class__(self.data[i])


    @property
    def s(self):
        if len(self) == 1:
            return self.A[0]
        else:
            return np.array([q.s for q in self])

    @property
    def v(self):
        if len(self) == 1:
            return self.A[1:4]
        else:
            return np.array([q.v for q in self])
    
    @property
    def vec(self):
        """Return the quaternion as 4-element vector.
        Code retrieved from: https://github.com/petercorke/robotics-toolbox-python/blob/master/robot/Quaternion.py
        Original authors: Luis Fernando Lara Tobar and Peter Corke
        @rtype: 4-vector
        @return: the quaternion elements
        """
        if len(self) == 1:
            return self.A
        else:
            return np.array([q.A for q in self])
    

    @classmethod
    def pure(cls, v):
        return cls(s=0, v=argcheck.getvector(v,3), norm=True)
    
    def conj(self):
        if instance(v, np.ndarray) and len(shape) > 1 and v.shape[1] == 3:
            return self.__class__( [quat.conj(q.A) for q in self] )
        else:
            return self.__class__(quat.qconj(self.A))

    def conj(self):
        return self.__class__( [quat.qconj(q.A) for q in self] )

    def norm(self):
        """Return the norm of this quaternion.
        Code retrieved from: https://github.com/petercorke/robotics-toolbox-python/blob/master/robot/Quaternion.py
        Original authors: Luis Fernando Lara Tobar and Peter Corke
        @rtype: number
        @return: the norm
        """
        if len(self) == 1:
            return np.linalg.norm(self.double())
        else:
            return np.array([quat.qnorm(q.A) for q in self])

    def unit(self):
        """Return an equivalent unit quaternion
        Code retrieved from: https://github.com/petercorke/robotics-toolbox-python/blob/master/robot/Quaternion.py
        Original authors: Luis Fernando Lara Tobar and Peter Corke
        @rtype: quaternion
        @return: equivalent unit quaternion
        """
        return UnitQuaternion( [quat.unit(q.A) for q in self], norm=False)


    def matrix(self):
        return qmatrix(self.A)
    
    #-------------------------------------------- arithmetic
    
    def __mul__(self, other):
        if type(self) == type(other):
            # quaternion * quaternion case (same class)
            return self.__class__( self._op2(other, lambda x, y: quat.qqmul(x, y) ) )
        elif isinstance(other, Quaternion):
            # quaternion * quaternion case (different class)
            return Quaternion( self._op2(other, lambda x, y: quat.qqmul(x, y) ) )
        elif argcheck.isscalar(other):
            # quaternion * scalar case
            print('scalar * quat')
            return Quaternion([other*q.A for q in self])
        elif isinstance(self, UnitQuaternion) and argcheck.isvector(other,3):
            # scalar * vector case
            return quat.qvmul(self.A, argcheck.getvector(other,3))
        else:
            raise ValueError('operands to * are of different types')
            
        return self._op2(other, lambda x, y: x @ y )

    def __rmul__(self, other):
        # scalar * quaternion case
        return Quaternion([other*q.A for q in self])
        
    def __imul__(self, other):
        return self.__mul__(other)
    
    def __pow__(self, n):
        return self.__class__([quat.qpow(q.A, n) for q in self])
    
    def __ipow__(self, n):
        return self.__pow__(n)
                    

    def __truediv__(self, other):
        raise NotImplemented('Quaternion division not supported')
    

    def __add__(self, other):
        # results is not in the group, return an array, not a class
        assert type(self) == type(other), 'operands to + are of different types'
        return Quaternion( self._op2(other, lambda x, y: x + y ) )

    def __sub__(self, other):
        # results is not in the group, return an array, not a class
        # TODO allow class +/- a conformant array
        assert type(self) == type(other), 'operands to - are of different types'
        return Quaternion( self._op2(other, lambda x, y: x - y ) )
    

    def __eq__(self, other):
        assert type(self) == type(other), 'operands to == are of different types'
        return self._op2(other, lambda x, y: quat.qequal(x, y), list1=False )
    
    def __ne__(self, other):
        return not self.__eq__(other)
    
    def _op2(self, other, op, list1=True):
        
        if len(self) == 1:
            if len(other) == 1:
                if list1:
                    return [op(self.A, other.A)]
                else:
                    return op(self.A, other.A)
            else:
                print('== 1xN')
                return [op(self.A, x.A) for x in other]
        else:
            if len(other) == 1:
                print('== Nx1')
                return [op(x.A, other.A) for x in self]
            elif len(self) == len(other):
                print('== NxN')
                return [op(x.A, y.A) for (x,y) in zip(self, other)]
            else:
                raise ValueError('length of lists to == must be same length')
                
                

    # def __truediv__(self, other):
    #     assert isinstance(other, Quaternion) or isinstance(other, int) or isinstance(other,
    #                                                                                  float), "Can be divided by a " \
    #                                                                                          "Quaternion, " \
    #                                                                                          "int or a float "
    #     qr = Quaternion()
    #     if type(other) is Quaternion:
    #         qr = self * other.inv()
    #     elif type(other) is int or type(other) is float:
    #         qr.s = self.s / other
    #         qr.v = self.v / other
    #     return qr

    # def __eq__(self, other):
    #     # assert type(other) is Quaternion
    #     try:
    #         np.testing.assert_almost_equal(self.s, other.s)
    #     except AssertionError:
    #         return False
    #     if not matrices_equal(self.v, other.v, decimal=7):
    #         return False
    #     return True

    # def __ne__(self, other):
    #     if self == other:
    #         return False
    #     else:
    #         return True

    def __repr__(self):
        s = ''
        for q in self:
            s += quat.qprint(q.A, file=None) + '\n'
        s.rstrip('\n')
        return s

    def __str__(self):
        return self.__repr__()


    
class UnitQuaternion(Quaternion):
    
    def __init__(self, s=None, v=None, norm=True, check=True):
        if s is None and v is None:
            self.data = [ quat.qone() ]
            
        elif argcheck.isscalar(s) and argcheck.isvector(v,3):
            q = np.r_[ s, argcheck.getvector(v) ]
            if norm:
                q = quat.qunit(q)
            self.data = [q]
            
        elif argcheck.isvector(s,4):
            print('uq constructor 4vec')
            q = argcheck.getvector(s)
            # if norm:
            #     q = quat.qunit(q)
            print(q)
            self.data = [q]
            
        elif type(s) is list:
            if check:
                assert argcheck.isvectorlist(s,4), 'list must comprise 4-vectors'
            if norm:
                s = [quat.qunit(q) for q in s]
            self.data = s
        
        elif isinstance(s, np.ndarray) and s.shape[1] == 4:
            if norm:
                self.data = [quat.qnorm(x) for x in s]
            else:
                self.data = [x for x in s]
            
        elif tr.isrot(s, check=check):
            self.data = [ quat.r2q(s) ]
            
        else:
            raise ValueError('bad argument to UnitQuaternion constructor')

    # def __getitem__(self, i):
    #     print('uq getitem', i)
    #     #return self.__class__(self.data[i])
    #     return self.__class__(self.data[i])
    
    @property
    def R(self):
        return quat.q2r(self.A)

    
    #-------------------------------------------- constructor variants
    @classmethod
    def Rx(cls, angle, unit='rad'):
        return cls(tr.rotx(angle, unit=unit), check=False)

    @classmethod
    def Ry(cls, angle, unit='rad'):
        return cls(tr.roty(angle, unit=unit), check=False)

    @classmethod
    def Rz(cls, angle, unit='rad'):
        return cls(tr.rotz(angle, unit=unit), check=False)

    @classmethod
    def vec3(cls, arg_in):
        assert isvec(arg_in, 3)
        s = 1 - np.linalg.norm(arg_in)
        return cls(s=s, v=arg_in)
    
    @classmethod
    def eul(cls, arg_in, unit='rad'):
        assert isvec(arg_in, 3)
        return cls.rot(eul2r(phi=arg_in, unit=unit))

    @classmethod
    def rpy(cls, arg_in, unit='rad'):
        return cls.rot(rpy2r(thetas=arg_in, unit=unit))

    @classmethod
    def angvec(cls, theta, v, unit='rad'):
        v = argcheck.getvector(v, 3)
        argcheck.isscalar(theta)
        theta = argcheck.getunit(theta, unit)
        return UnitQuaternion(s=math.cos(theta/2), v=math.sin(theta/2) * tr.unit(v), norm=False)

    def __truediv__(self, other):
        assert type(self) == type(other), 'operands to * are of different types'
        return self._op2(other, lambda x, y: quat.qqmul(x, quat.qconj(y)) )
    
    def inv(self):
        return self.__class__([quat.qconj(q.A) for q in self])
    
    @classmethod
    def omega(cls, w):
        assert isvec(w, 3)
        theta = np.linalg.norm(w)
        s = math.cos(theta / 2)
        v = math.sin(theta / 2) * unitize(w)
        return cls(s=s, v=v)


    def dot(self, omega):
        E = self.s * np.asmatrix(np.eye(3, 3)) - skew(self.v)
        qd = -self.v * omega
        return 0.5 * np.r_[qd, E*omega]

    def dotb(self, omega):
        E = self.s * np.asmatrix(np.eye(3, 3)) + skew(self.v)
        qd = -self.v * omega
        return 0.5 * np.r_[qd, E*omega]

    def plot(self):
        from .pose import SO3
        SO3.np(self.r()).plot()

    def animate(self, qr=None, duration=5, gif=None):
        self.pipeline = VtkPipeline(total_time_steps=duration*60, gif_file=gif)
        axis = vtk.vtkAxesActor()
        axis.SetAxisLabels(0)
        self.pipeline.add_actor(axis)
        if qr is None:
            q1 = UnitQuaternion()
            q2 = self
        else:
            assert type(qr) is UnitQuaternion
            q1 = self
            q2 = qr

        cube_axes = axesCube(self.pipeline.ren)
        self.pipeline.add_actor(cube_axes)

        def execute(obj, event):
            # print(self.timer_count)
            nonlocal axis
            self.pipeline.timer_tick()

            axis.SetUserMatrix(np2vtk(q1.interp(q2, r=1/self.pipeline.total_time_steps * self.pipeline.timer_count).q2tr()))
            self.pipeline.iren.GetRenderWindow().Render()

        self.pipeline.iren.AddObserver('TimerEvent', execute)
        self.pipeline.animate()


    def interp(self, qr, r=0.5, shortest=False):
        """
        Algorithm source: https://en.wikipedia.org/wiki/Slerp
        :param qr: UnitQuaternion
        :param shortest: Take the shortest path along the great circle
        :param r: interpolation point
        :return: interpolated UnitQuaternion
        """
        assert type(qr) is UnitQuaternion
        if self == qr:
            return self

        q1 = self.double()
        q2 = qr.double()
        dot = q1*np.transpose(q2)

        # If the dot product is negative, the quaternions
        # have opposite handed-ness and slerp won't take
        # the shorter path. Fix by reversing one quaternion.
        if shortest:
            if dot < 0:
                q1 = - q1
                dot = -dot

        dot = np.clip(dot, -1, 1)  # Clip within domain of acos()
        theta_0 = math.acos(dot)  # theta_0 = angle between input vectors
        theta = theta_0 * r  # theta = angle between v0 and result
        s1 = float(math.cos(theta) - dot * math.sin(theta) / math.sin(theta_0))
        s2 = math.sin(theta) / math.sin(theta_0)
        out = (q1 * s1) + (q2 * s2)
        return UnitQuaternion(s=float(out[0, 0]), v=out[0, 1:])



    def to_angvec(self, unit='rad'):
        vec, theta = 0, 0
        if np.linalg.norm(self.v) < 10 * np.spacing([1])[0]:
            vec = np.matrix([[0, 0, 0]])
            theta = 0
        else:
            vec = unitize(vec)
            theta = 2 * math.atan2(np.linalg.norm(self.v), self.s)

        if unit == 'deg':
            theta = theta * 180 / math.pi
        return theta, vec

    def to_so3(self):
        from .pose import SO3
        return SO3.np(self.r())

    def to_se3(self):
        from .pose import SE3
        from .pose import SO3
        return SE3(so3=SO3.np(self.r()))


    def q2r(self):
        return self.to_rot()

    def q2tr(self):
        return r2t(self.to_rot())


    def __matmul__(self, other):
        assert type(other) is UnitQuaternion
        return (self * other).unit()

    def __floordiv__(self, other):
        assert type(other) is UnitQuaternion
        return (self / other).unit()
    
    def __repr__(self):
        s = ''
        for q in self:
            s += quat.qprint(q.A, delim=('<<', '>>'), file=None) + '\n'
        s.rstrip('\n')
        return s
    
    def __str__(self):
        return self.__repr__()

    @classmethod
    def rot(cls, arg_in):
        qr = cls()
        return qr.tr2q(arg_in)

    @classmethod
    def qt(cls, arg_in):
        if type(arg_in) is Quaternion:
            arg_in = arg_in.unit()
        else:
            assert type(arg_in) is UnitQuaternion
        return cls(arg_in.s, arg_in.v)

    @classmethod
    def eul(cls, arg_in, unit='rad'):
        assert isvec(arg_in, 3)
        return cls.rot(eul2r(phi=arg_in, unit=unit))

    @classmethod
    def rpy(cls, arg_in, unit='rad'):
        return cls.rot(rpy2r(thetas=arg_in, unit=unit))
    



    @classmethod
    def omega(cls, w):
        assert isvec(w, 3)
        theta = np.linalg.norm(w)
        s = math.cos(theta / 2)
        v = math.sin(theta / 2) * unitize(w)
        return cls(s=s, v=v)



    def dot(self, omega):
        E = self.s * np.asmatrix(np.eye(3, 3)) - skew(self.v)
        qd = -self.v * omega
        return 0.5 * np.r_[qd, E*omega]

    def dotb(self, omega):
        E = self.s * np.asmatrix(np.eye(3, 3)) + skew(self.v)
        qd = -self.v * omega
        return 0.5 * np.r_[qd, E*omega]

    def plot(self):
        from .pose import SO3
        SO3.np(self.r()).plot()

    def animate(self, qr=None, duration=5, gif=None):
        self.pipeline = VtkPipeline(total_time_steps=duration*60, gif_file=gif)
        axis = vtk.vtkAxesActor()
        axis.SetAxisLabels(0)
        self.pipeline.add_actor(axis)
        if qr is None:
            q1 = UnitQuaternion()
            q2 = self
        else:
            assert type(qr) is UnitQuaternion
            q1 = self
            q2 = qr

        cube_axes = axesCube(self.pipeline.ren)
        self.pipeline.add_actor(cube_axes)

        def execute(obj, event):
            # print(self.timer_count)
            nonlocal axis
            self.pipeline.timer_tick()

            axis.SetUserMatrix(np2vtk(q1.interp(q2, r=1/self.pipeline.total_time_steps * self.pipeline.timer_count).q2tr()))
            self.pipeline.iren.GetRenderWindow().Render()

        self.pipeline.iren.AddObserver('TimerEvent', execute)
        self.pipeline.animate()

    def matrix(self):
        pass



    def to_vec(self):
        if self.s < 0:
            return -self.v
        else:
            return self.v

    def to_rpy(self):
        return tr2rpy(self.r())

    def to_angvec(self, unit='rad'):
        vec, theta = 0, 0
        if np.linalg.norm(self.v) < 10 * np.spacing([1])[0]:
            vec = np.matrix([[0, 0, 0]])
            theta = 0
        else:
            vec = unitize(vec)
            theta = 2 * math.atan2(np.linalg.norm(self.v), self.s)

        if unit == 'deg':
            theta = theta * 180 / math.pi
        return theta, vec

    def to_so3(self):
        from .pose import SO3
        return SO3.np(self.r())

    def to_se3(self):
        from .pose import SE3
        from .pose import SO3
        return SE3(so3=SO3.np(self.r()))

    def to_rot(self):
        q = self.double()
        s = q[0, 0]
        x = q[0, 1]
        y = q[0, 2]
        z = q[0, 3]
        return np.matrix([[1 - 2 * (y ** 2 + z ** 2), 2 * (x * y - s * z), 2 * (x * z + s * y)],
                          [2 * (x * y + s * z), 1 - 2 * (x ** 2 + z ** 2), 2 * (y * z - s * x)],
                          [2 * (x * z - s * y), 2 * (y * z + s * x), 1 - 2 * (x ** 2 + y ** 2)]])

    def q2r(self):
        return self.to_rot()

    def q2tr(self):
        return r2t(self.to_rot())

    def __matmul__(self, other):
        assert type(other) is UnitQuaternion
        return (self * other).unit()

    def __floordiv__(self, other):
        assert type(other) is UnitQuaternion
        return (self / other).unit()

if __name__ == '__main__':
    q = Quaternion([1,2,3,4])
    print(q)
    q.append(q)
    print(len(q))
    print(q)

    a = np.random.uniform(size=(6,4))
    q = Quaternion(a)
    q2 = Quaternion()

    u = UnitQuaternion()
    #print(u)
    len(u)
    a = u[0]


    import numpy.testing as nt
    import unittest
    
    from transforms import *
            
    class TestUnitQuaternion(unittest.TestCase):
        
        def test_constructor(self):
            nt.assert_array_almost_equal(UnitQuaternion().vec, np.r_[1,0,0,0])
            
            nt.assert_array_almost_equal(UnitQuaternion.Rx(90,'deg').vec, np.r_[1,1,0,0]/math.sqrt(2))
            nt.assert_array_almost_equal(UnitQuaternion.Rx(-90,'deg').vec, np.r_[1,-1,0,0]/math.sqrt(2))
            nt.assert_array_almost_equal(UnitQuaternion.Ry(90,'deg').vec, np.r_[1,0,1,0]/math.sqrt(2))
            nt.assert_array_almost_equal(UnitQuaternion.Ry(-90,'deg').vec, np.r_[1,0,-1,0]/math.sqrt(2))
            nt.assert_array_almost_equal(UnitQuaternion.Rz(90,'deg').vec, np.r_[1,0,0,1]/math.sqrt(2))
            nt.assert_array_almost_equal(UnitQuaternion.Rz(-90,'deg').vec, np.r_[1,0,0,-1]/math.sqrt(2))
            
    unittest.main()
