# Author: Aditya Dua
# 28 January, 2018

import numpy as np
import math
import transforms as tr
import quat_np as quat
from collections import UserList
import argcheck


class Quaternion(UserList):
    def __init__(self, a1=None, a2=None, check=True):
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
        if a1 is None and a2 is None:
            self.data = [ quat.qone() ]
            
        elif argcheck.isscalar(a1) and argcheck.isvector(a2,3):
            self.data = [ a1, argcheck.getvector(a2) ]
            
        elif argcheck.isvector(a1,4):
            self.data = [ argcheck.getvector(a1) ]
            
        elif type(a1) is list:
            if check:
                assert argcheck.isvectorlist(a1,4), 'list must comprise 4-vectors'
            self.data = [ a1 ]
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
        return self.A[0]

    @property
    def v(self):
        return self.A[1:4]
    
    @property
    def vec(self):
        return self.A
    

    @classmethod
    def pure(cls, vec):
        assert isvec(vec, 3)
        return cls(s=0, v=vec)

    def conj(self):
        return Quaternion(quat.qconj(self.A))



    def tr(self):
        return t2r(self.r())

    def norm(self):
        """Return the norm of this quaternion.
        Code retrieved from: https://github.com/petercorke/robotics-toolbox-python/blob/master/robot/Quaternion.py
        Original authors: Luis Fernando Lara Tobar and Peter Corke
        @rtype: number
        @return: the norm
        """
        return np.linalg.norm(self.double())

    def double(self):
        """Return the quaternion as 4-element vector.
        Code retrieved from: https://github.com/petercorke/robotics-toolbox-python/blob/master/robot/Quaternion.py
        Original authors: Luis Fernando Lara Tobar and Peter Corke
        @rtype: 4-vector
        @return: the quaternion elements
        """
        return np.concatenate((np.matrix(self.s), self.v), 1)

    def unit(self):
        """Return an equivalent unit quaternion
        Code retrieved from: https://github.com/petercorke/robotics-toolbox-python/blob/master/robot/Quaternion.py
        Original authors: Luis Fernando Lara Tobar and Peter Corke
        @rtype: quaternion
        @return: equivalent unit quaternion
        """
        qr = UnitQuaternion()
        nm = self.norm()
        qr.s = float(self.s / nm)
        qr.v = self.v / nm
        return qr

    def r(self):
        """Return an equivalent rotation matrix.
        Code retrieved from: https://github.com/petercorke/robotics-toolbox-python/blob/master/robot/Quaternion.py
        Original authors: Luis Fernando Lara Tobar and Peter Corke
        @rtype: 3x3 orthonormal rotation matrix
        @return: equivalent rotation matrix
        """
        s = self.s
        x = self.v[0, 0]
        y = self.v[0, 1]
        z = self.v[0, 2]

        return self.__class__(q2r(self.A))

    def matrix(self):
        return qmatrix(self.A)

    def __mul__(self, other):
        assert isinstance(other, Quaternion) \
               or isinstance(other, int) \
               or isinstance(other, float), "Can be multiplied with Quaternion, int or a float. "
        if type(other) is Quaternion:
            qr = Quaternion()
        else:
            qr = UnitQuaternion()
        if isinstance(other, Quaternion):
            qr.s = self.s * other.s - self.v * np.transpose(other.v)
            qr.v = self.s * other.v + other.s * self.v + np.cross(self.v, other.v)
        elif type(other) is int or type(other) is float:
            qr.s = self.s * other
            qr.v = self.v * other
        return qr

    def __pow__(self, power):
        """
        Code retrieved from: https://github.com/petercorke/robotics-toolbox-python/blob/master/robot/Quaternion.py
        Original authors: Luis Fernando Lara Tobar and Peter Corke
        :param power:
        :param modulo:
        :return:
        """
        return self.__class__([quat.qpow(q.A) for q in self])

    def __imul__(self, other):
        """
        Code retrieved from: https://github.com/petercorke/robotics-toolbox-python/blob/master/robot/Quaternion.py
        Original authors: Luis Fernando Lara Tobar and Peter Corke
        :param other:
        :return: self
        """
        if isinstance(other, Quaternion):
            s1 = self.s
            v1 = self.v
            s2 = other.s
            v2 = other.v

            # form the product
            self.s = s1 * s2 - v1 * v2.T
            self.v = s1 * v2 + s2 * v1 + np.cross(v1, v2)

        elif type(other) is int or type(other) is float:
            self.s *= other
            self.v *= other

        return self

    def __add__(self, other):
        assert type(self) == type(other), "Both objects should be of type: Quaternion"
        return Quaternion(s=self.s + other.s, v=self.v + other.v)

    def __sub__(self, other):
        assert type(self) == type(other), "Both objects should be of type: Quaternion"
        return Quaternion(s=self.s - other.s, v=self.v - other.v)

    def __truediv__(self, other):
        assert isinstance(other, Quaternion) or isinstance(other, int) or isinstance(other,
                                                                                     float), "Can be divided by a " \
                                                                                             "Quaternion, " \
                                                                                             "int or a float "
        qr = Quaternion()
        if type(other) is Quaternion:
            qr = self * other.inv()
        elif type(other) is int or type(other) is float:
            qr.s = self.s / other
            qr.v = self.v / other
        return qr

    def __eq__(self, other):
        # assert type(other) is Quaternion
        try:
            np.testing.assert_almost_equal(self.s, other.s)
        except AssertionError:
            return False
        if not matrices_equal(self.v, other.v, decimal=7):
            return False
        return True

    def __ne__(self, other):
        if self == other:
            return False
        else:
            return True

    def __repr__(self):
        return quat.qprint(self.A)

    def __str__(self):
        return self.__repr__()


class UnitQuaternion(Quaternion):
    
    def __init__(self, s=None, v=None):
        self.pipeline = None
        if s is None:
            s = 1
        if v is None:
            v = np.matrix([[0, 0, 0]])
        super().__init__(s, v)

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
    
    def inv(self):
        return Quaternion(s=self.s, v=-self.v)

    @classmethod
    def angvec(cls, theta, v, unit='rad'):
        assert isvec(v, 3)
        assert type(theta) is float or type(theta) is int
        uq = UnitQuaternion()
        if unit == 'deg':
            theta = theta * math.pi / 180
        uq.s = math.cos(theta/2)
        uq.v = math.sin(theta/2) * unitize(v)
        return uq

    @classmethod
    def omega(cls, w):
        assert isvec(w, 3)
        theta = np.linalg.norm(w)
        s = math.cos(theta / 2)
        v = math.sin(theta / 2) * unitize(w)
        return cls(s=s, v=v)

    @classmethod
    def Rx(cls, angle, unit='rad'):
        return cls.rot(rotx(angle, unit=unit))

    @classmethod
    def Ry(cls, angle, unit='rad'):
        return cls.rot(roty(angle, unit=unit))

    @classmethod
    def Rz(cls, angle, unit='rad'):
        return cls.rot(rotz(angle, unit=unit))

    @classmethod
    def vec(cls, arg_in):
        assert isvec(arg_in, 3)
        s = 1 - np.linalg.norm(arg_in)
        return cls(s=s, v=arg_in)

    def new(self):
        return UnitQuaternion(s=self.s, v=self.v)

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

    @staticmethod
    def tr2q(t):
        """
        Converts a homogeneous rotation matrix to a Quaternion object
        Code retrieved from: https://github.com/petercorke/robotics-toolbox-python/blob/master/robot/Quaternion.py
        Original authors: Luis Fernando Lara Tobar and Peter Corke
        :param t: homogeneous matrix
        :return: quaternion object
        """
        assert ishomog(t, (3, 3)), "Argument must be 3x3 homogeneous numpy matrix"
        qs = sqrt(trace(t) + 1) / 2.0
        kx = t[2, 1] - t[1, 2]  # Oz - Ay
        ky = t[0, 2] - t[2, 0]  # Ax - Nz
        kz = t[1, 0] - t[0, 1]  # Ny - Ox

        if (t[0, 0] >= t[1, 1]) and (t[0, 0] >= t[2, 2]):
            kx1 = t[0, 0] - t[1, 1] - t[2, 2] + 1  # Nx - Oy - Az + 1
            ky1 = t[1, 0] + t[0, 1]  # Ny + Ox
            kz1 = t[2, 0] + t[0, 2]  # Nz + Ax
            add = (kx >= 0)
        elif t[1, 1] >= t[2, 2]:
            kx1 = t[1, 0] + t[0, 1]  # Ny + Ox
            ky1 = t[1, 1] - t[0, 0] - t[2, 2] + 1  # Oy - Nx - Az + 1
            kz1 = t[2, 1] + t[1, 2]  # Oz + Ay
            add = (ky >= 0)
        else:
            kx1 = t[2, 0] + t[0, 2]  # Nz + Ax
            ky1 = t[2, 1] + t[1, 2]  # Oz + Ay
            kz1 = t[2, 2] - t[0, 0] - t[1, 1] + 1  # Az - Nx - Oy + 1
            add = (kz >= 0)

        if add:
            kx = kx + kx1
            ky = ky + ky1
            kz = kz + kz1
        else:
            kx = kx - kx1
            ky = ky - ky1
            kz = kz - kz1

        kv = np.matrix([[kx, ky, kz]])
        nm = np.linalg.norm(kv)
        qr = UnitQuaternion()
        if nm == 0:
            qr.s = 1.0
            qr.v = np.matrix([[0.0, 0.0, 0.0]])

        else:
            qr.s = qs
            qr.v = (sqrt(1 - qs ** 2) / nm) * kv

        return qr

    def __matmul__(self, other):
        assert type(other) is UnitQuaternion
        return (self * other).unit()

    def __floordiv__(self, other):
        assert type(other) is UnitQuaternion
        return (self / other).unit()

q = Quaternion([1,2,3,4])
#print(q)