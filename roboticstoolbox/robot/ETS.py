#!/usr/bin/env python3
"""
@author: Jesse Haviland
@author: Peter Corke
"""
from collections import UserList, namedtuple
from abc import ABC
import numpy as np
from spatialmath import SE3, SE2
from spatialmath.base import getvector, getunit, trotx, troty, trotz, \
    issymbol, tr2jac, transl2, trot2, removesmall, trinv


class SuperETS(UserList, ABC):

    # T is a NumPy array (4,4) or None
    ets_tuple = namedtuple('ETS3', 'eta axis_func axis joint T jindex flip')

    def __init__(
            self, axis_func=None, axis=None, eta=None,
            unit='rad', j=None, flip=False):

        super().__init__()  # init UserList superclass

        if axis_func is None and axis is None and eta is None:
            # ET()
            # create instance with no values
            self.data = []
            return

        elif isinstance(axis_func, ETS):
            # copy constructor
            e = axis_func
            axis_func = e.axis_func
            axis = e.axis
            # et = e.eta
            j = e.jindex
            flip = e.isflip
            joint = e.isjoint
            T = e.T

        elif callable(axis_func):
            if eta is None:
                # no value, it's a variable joint
                if unit != 'rad':
                    raise ValueError(
                        'can only use radians for a variable transform')
                joint = True
                T = None

            else:
                # constant value specified
                joint = False
                eta = getunit(eta, unit)
                T = axis_func(eta)
                if j is not None:
                    raise ValueError(
                        'cannot specify joint index for a constant ET')
                if flip:
                    raise ValueError(
                        'cannot specify flip for a constant ET')

        elif axis == 'C':
            # it's a constant element  Ci
            if isinstance(self, ETS):
                # ETS
                if not isinstance(eta, np.ndarray):
                    T = eta.A
                else:
                    T = eta
                if T.shape != (4, 4):
                    raise ValueError('argument must be ndarray(4,4) or SE3')
            else:
                # ETS2
                if not isinstance(eta, np.ndarray):
                    T = eta.A
                else:
                    T = eta
                if T.shape != (3, 3):
                    raise ValueError('argument must be ndarray(3,3) or SE2')
            axis = "C"
            joint = False
            axis_func = None
        else:
            raise ValueError('axis_func must be callable or ndarray')

        # Save all the params in a named tuple
        e = self.ets_tuple(eta, axis_func, axis, joint, T, j, flip)

        # And make it the only value of this instance
        self.data = [e]

    @property
    def eta(self):
        """
        The transform constant

        :return: The constant η if set
        :rtype: float or None

        Example:

        .. runblock:: pycon

            >>> from roboticstoolbox import ETS
            >>> e = ETS.tx(1)
            >>> e.eta
            >>> e = ETS.rx(90, 'deg')
            >>> e.eta
            >>> e = ETS.ty()
            >>> e.eta

        .. note:: If the value was given in degrees it will be converted and
            stored internally in radians
        """
        return self.data[0].eta

    @property
    def axis_func(self):
        return self.data[0].axis_func

    @property
    def axis(self):
        """
        The transform type and axis

        :return: The transform type and axis
        :rtype: str

        Example:

        .. runblock:: pycon

            >>> from roboticstoolbox import ETS
            >>> e = ETS.tx(1)
            >>> e.axis
            >>> e = ETS.rx(90, 'deg')
            >>> e.axis

        """
        return self.data[0].axis

    @property
    def n(self):
        """
        Number of joints

        :return: the number of joints in the ETS
        :rtype: int

        Counts the number of joints in the ETS.

        Example:

        .. runblock:: pycon

            >>> from roboticstoolbox import ETS
            >>> e = ETS.rx() * ETS.tx(1) * ETS.tz()
            >>> e.n

        :seealso: :func:`joints`
        """
        n = 0
        for et in self:
            if et.isjoint:
                n += 1

        return n

    @property
    def isjoint(self):
        """
        Test if ET is a joint

        :return: True if a joint
        :rtype: bool

        Example:

        .. runblock:: pycon

            >>> from roboticstoolbox import ETS
            >>> e = ETS.tx(1)
            >>> e.isjoint
            >>> e = ETS.tx()
            >>> e.isjoint
        """
        return self.data[0].joint

    @property
    def isflip(self):
        """
        Test if ET joint is flipped

        :return: True if joint is flipped
        :rtype: bool

        A flipped joint uses the negative of the joint variable, ie. it rotates
        or moves in the opposite direction.

        Example:

        .. runblock:: pycon

            >>> from roboticstoolbox import ETS
            >>> e = ETS.tx()
            >>> e.T(1)
            >>> eflip = ETS.tx(flip=True)
            >>> eflip.T(1)
        """
        return self.data[0].flip

    @property
    def jindex(self):
        """
        Get ET joint index

        :return: The assigmed joint index
        :rtype: int or None

        Allows an ET to be associated with a numbered joint in a robot.

        Example:

        .. runblock:: pycon

            >>> from roboticstoolbox import ETS
            >>> e = ETS.tx()
            >>> print(e)
            >>> e = ETS.tx(j=3)
            >>> print(e)
            >>> print(e.jindex)
        """
        return self.data[0].jindex

    @property
    def isrevolute(self):
        """
        Test if ET is a rotation

        :return: True if a rotation
        :rtype: bool

        Example:

        .. runblock:: pycon

            >>> from roboticstoolbox import ETS
            >>> e = ETS.tx(1)
            >>> e.isrevolute
            >>> e = ETS.rx()
            >>> e.isrevolute
        """
        return self.axis[0] == 'R'

    @property
    def isprismatic(self):
        """
        Test if ET is a translation

        :return: True if a translation
        :rtype: bool

        Example:

        .. runblock:: pycon

            >>> from roboticstoolbox import ETS
            >>> e = ETS.tx(1)
            >>> e.isprismatic
            >>> e = ETS.rx()
            >>> e.isprismatic
        """
        return self.axis[0] == 't'

    @property
    def isconstant(self):
        """
        Test if ET is a compilation constant

        :return: True if a compilation constant
        :rtype: bool

        During compilation, consectutive non-joint ETs are compounded/folded
        into a constant transform.  In a ``str`` representation these ETs are
        denoted by ``Ci`` where ``i`` are integers starting at zero.

        .. note:: This is ET is not actually "elementary", it can be a complex
            mix of rotations and translations.

        :seealso: :func:`compile`
        """
        return self.axis[0] == 'C'

    @property
    def config(self):
        """
        Joint configuration string

        :return: A string indicating the joint types
        :rtype: str

        A string comprising the characters 'R' or 'P' which indicate the types
        of joints in order from left to right.

        Example:

        .. runblock:: pycon

            >>> from roboticstoolbox import ETS
            >>> e = ETS.tz() * ETS.tx(1) * ETS.rz() * ETS.tx(1)
            >>> e.config

        """
        return ''.join(
            ['R' if self.isrevolute else 'P' for i in self.joints()])

    def joints(self):
        """
        Get index of joint transforms

        :return: indices of transforms that are joints
        :rtype: list of int

        Example:

        .. runblock:: pycon

            >>> from roboticstoolbox import ETS
            >>> e = ETS.rz() * ETS.tx(1) * ETS.rz() * ETS.tx(1)
            >>> e.joints()

        """
        return np.where([e.isjoint for e in self])[0]

    def jointset(self):
        """
        Get set of joint indices

        :return: set of unique joint indices
        :rtype: set

        Example:

        .. runblock:: pycon

            >>> from roboticstoolbox import ETS
            >>> e = ETS.rz(j=1) * ETS.tx(j=2) * ETS.rz(j=1) * ETS.tx(1)
            >>> e.jointset()
        """
        return set([self[j].jindex for j in self.joints()])

    def T(self, q=None):
        """
        Evaluate an elementary transformation

        :param q: Is used if this ET is variable (a joint)
        :type q: float (radians), required for variable ET's
        :return: The SE(3) or SE(2) matrix value of the ET
        :rtype:  ndarray(4,4) or ndarray(3,3)

        Example:

        .. runblock:: pycon

            >>> from roboticstoolbox import ETS
            >>> e = ETS.tx(1)
            >>> e.T()
            >>> e = ETS.tx()
            >>> e.T(2)

        """
        if self.isjoint:
            return self.axis_func(q)
        else:
            return self.data[0].T

    def eval(self, q=None, unit='rad'):
        """
        Evaluate an ETS with joint coordinate substitution

        :param q: joint coordinates
        :type q: array-like
        :param unit: angular unit, "rad" [default] or "deg"
        :type unit: str
        :return: The SE(3) or SE(2) matrix value of the ET sequence
        :rtype:  ndarray(4,4) or ndarray(3,3)

        Effectively the forward-kinematics of the ET sequence.  Compounds the
        transforms left to right, and substitutes in joint coordinates as
        needed from consecutive elements of ``q``.

        .. note:: if ETs have an explicit joint index, this is used to index
            into the vector ``q``.

        .. warning:: do not mix ETs with and without explicit joint index.

        Example:

        .. runblock:: pycon

            >>> from roboticstoolbox import ETS
            >>> e = ETS.rz() * ETS.tx(1) * ETS.rz() * ETS.tx(1)
            >>> print(e)
            >>> len(e)
            >>> e[1:3]
            >>> e.eval([0, 0])
            >>> e.eval([90, -90], 'deg')
            >>> e = ETS.tx(j=1) * ETS.ty(j=0)
            >>> e.eval([3, 4])
        """
        if q is not None:
            q = getvector(q, out='list')
        first = True
        for et in self:
            if et.isjoint:
                if et.jindex is None:
                    qj = q.pop(0)
                else:
                    qj = q[et.jindex]
                if et.isrevolute and unit == 'deg':
                    qj *= np.pi / 180.0
                if et.isflip:
                    qj = -qj
                Tk = et.T(qj)
            else:
                # for constants
                Tk = et.T()
            if first:
                T = Tk
                first = False
            else:
                T = T @ Tk

        if isinstance(self, ETS):
            T = SE3(T, check=False)
        elif isinstance(self, ETS2):
            T = SE2(T, check=False)
        T.simplify()
        return T

    def compile(self):
        """
        Compile an ETS

        :return: optimised ETS
        :rtype: ETS

        Perform constant folding for faster evaluation.  Consecutive constant
        ETs are compounded, leading to a constant ET which is denoted by
        ``Ci`` when displayed.

        Example:

        .. runblock:: pycon

            >>> from roboticstoolbox import ETS
            >>> robot = rtb.models.ETS.Panda()
            >>> ets = robot.ets()
            >>> ets
            >>> ets.compile()

        :seealso: :func:`isconstant`
        """
        const = None
        ets = ETS()
        for et in self:

            if et.isjoint:
                # a joint
                if const is not None:
                    # flush the constant
                    ets *= ETS._CONST(const)
                    const = None
                ets *= et  # emit the joint ET
            else:
                # not a joint
                if const is None:
                    const = et.T()
                else:
                    const = const @ et.T()

        if const is not None:
            # flush the constant, tool transform
            ets *= ETS._CONST(const)
        return ets

    def __str__(self, q=None):
        """
        Pretty prints the ETS

        :param q: control how joint variables are displayed
        :type q: str
        :return: Pretty printed ETS
        :rtype: str

        ``q`` controls how the joint variables are displayed:

        - None, format depends on number of joint variables
            - one, display joint variable as q
            - more, display joint variables as q0, q1, ...
            - if a joint index was provided, use this value
        - "", display all joint variables as empty parentheses ``()``
        - "θ", display all joint variables as ``(θ)``
        - format string with passed joint variables ``(j, j+1)``, so "θ{0}"
          would display joint variables as θ0, θ1, ... while "θ{1}" would
          display joint variables as θ1, θ2, ...  ``j`` is either the joint
          index, if provided, otherwise a sequential value.

        Example:

        .. runblock:: pycon

            >>> from roboticstoolbox import ETS
            >>> e = ETS.rz() * ETS.tx(1) * ETS.rz()
            >>> print(e[:2])
            >>> print(e)
            >>> print(e.__str__(""))
            >>> print(e.__str__("θ{0}"))  # numbering from 0
            >>> print(e.__str__("θ{1}"))  # numbering from 1
            >>> # explicit joint indices
            >>> e = ETS.rz(j=3) * ETS.tx(1) * ETS.rz(j=4)
            >>> print(e)
            >>> print(e.__str__("θ{0}"))

        .. note:: Angular parameters are converted to degrees, except if they
            are symbolic.

        Example:

        .. runblock:: pycon

            >>> from roboticstoolbox import ETS
            >>> from spatialmath.base import symbol
            >>> theta, d = symbol('theta, d')
            >>> e = ETS.rx(theta) * ETS.tx(2) * ETS.rx(45, 'deg') * \
            >>>     ETS.ry(0.2) * ETS.ty(d)
            >>> str(e)

        :SymPy: supported
        """
        es = []
        j = 0
        c = 0

        if q is None:
            if len(self.joints()) > 1:
                q = "q{0}"
            else:
                q = "q"

        # For et in the object, display it, data comes from properties
        # which come from the named tuple
        for et in self:

            if et.isjoint:
                if q is not None:
                    if et.jindex is None:
                        _j = j
                    else:
                        _j = et.jindex
                    qvar = q.format(_j, _j+1)
                else:
                    qvar = ""
                if et.isflip:
                    s = f"{et.axis}(-{qvar})"
                else:
                    s = f"{et.axis}({qvar})"
                j += 1

            elif et.isrevolute:
                if issymbol(et.eta):
                    s = f"{et.axis}({et.eta})"
                else:
                    s = f"{et.axis}({et.eta * 180 / np.pi:.4g}°)"

            elif et.isprismatic:
                s = f"{et.axis}({et.eta})"

            elif et.isconstant:
                s = f"C{c}"
                c += 1

            es.append(s)

        return " * ".join(es)

    # redefine * operator to concatenate the internal lists
    def __mul__(self, rest):
        """
        Overloaded ``*`` operator

        :return: [description]
        :rtype: [type]

        Example:

        .. runblock:: pycon

            >>> from roboticstoolbox import ETS
            >>> e1 = ETS.rz()
            >>> len(e1)
            >>> e2= ETS.tx(2)
            >>> len(e2)
            >>> e = e1 * e2
            >>> len(e)

        .. note:: The ``*`` operator implies composition, but actually the
            result is a new ETS instance that contains the concatenation of
            the left and right operands in an internal list. In this example
            we see the length of the product is 2.
        """
        prod = ETS()
        prod.data = self.data + rest.data
        return prod

    def __imul__(self, rest):
        prod = ETS()
        prod.data = self.data + rest.data
        return prod

    # redefine so that indexing returns an ET type
    def __getitem__(self, i):
        """
        Index or slice an ETS

        :param i: the index or slince
        :type i: int or slice
        :return: Elementary transform (sequence)
        :rtype: ETS

        Example:

        .. runblock:: pycon

            >>> from roboticstoolbox import ETS
            >>> e = ETS.rz() * ETS.tx(1) * ETS.rz() * ETS.tx(1)
            >>> e[0]
            >>> e[1]
            >>> e[1:3]

        """
        item = ETS()
        data = self.data[i]  # can be [2] or slice, eg. [3:5]
        # ensure that data is always a list
        if isinstance(data, list):
            item.data = data
        else:
            item.data = [data]
        return item

    def pop(self, i=-1):
        """
        Pop value

        :param i: item in the list to pop, default is last
        :type i: int
        :return: the popped value
        :rtype: instance of same type
        :raises IndexError: if there are no values to pop

        Removes a value from the value list and returns it.  The original
        instance is modified.

        Example:

        .. runblock:: pycon

            >>> from roboticstoolbox import ETS
            >>> e = ETS.rz() * ETS.tx(1) * ETS.rz() * ETS.tx(1)
            >>> tail = e.pop()
            >>> tail
            >>> e
        """
        item = ETS()
        item.data = [super().pop(i)]
        return item

    def __repr__(self):
        return str(self)

    @classmethod
    def _CONST(cls, T):
        return cls(None, axis='C', eta=T)

    @classmethod
    def SE3(cls, t, rpy=None, tol=100):
        """
        Convert an SE3 to an ETS

        :param t: Translation vector, or an SE(3) matrix
        :type t: array_like(3) or SE3 instance
        :param rpy: roll-pitch-yaw angles in XYZ order
        :type rpy: array_like(3)
        :param tol: Elements small than this many eps are considered as
            being zero, defaults to 100
        :type tol: int, optional
        :return: ET sequence
        :rtype: ETS instance

        Create an ETS from the non-zero translational and rotational
        components.

        - ``SE3(t, rpy)`` convert translation ``t`` and rotation given by XYZ
          roll-pitch-yaw angles ``rpy`` into an ETS.
        - ``SE3(X)`` as above but convert from an SE3 instance ``X``.

        Example:

        .. runblock:: pycon

            >>> from roboticstoolbox import ETS
            >>> ETS.SE3(SE3(1,2,3))
            >>> ETS.SE3(SE3.Rx(90, 'deg'))

        .. warning:: ``SE3.rpy()`` is used to determine rotation about the x-,
            y- and z-axes.  For a y-axis rotation with magnitude greater than
            180° this will result in a non-minimal representation with non-zero
            amounts of x- and z-axis rotation.

        :seealso: :func:`~SE3.rpy`
        """
        if isinstance(t, SE3):
            T = t
            t = removesmall(T.t, tol)
            rpy = removesmall(T.rpy(order='zyx'))

        ets = ETS()
        if t[0] != 0:
            ets *= ETS.tx(t[0])
        if t[1] != 0:
            ets *= ETS.ty(t[1])
        if t[2] != 0:
            ets *= ETS.tz(t[2])

        if rpy is not None:
            if rpy[2] != 0:
                ets *= ETS.rz(rpy[2])
            if rpy[1] != 0:
                ets *= ETS.ry(rpy[1])
            if rpy[0] != 0:
                ets *= ETS.rx(rpy[0])

        return ets

    def inv(self):
        r"""
        Inverse of ETS

        :return: [description]
        :rtype: ETS instance

        The inverse of a given ETS.  It is computed as the inverse of the
        individual ETs in the reverse order.

        .. math::

            (\mathbf{E}_0, \mathbf{E}_1 \cdots \mathbf{E}_{n-1} )^{-1} = (\mathbf{E}_{n-1}^{-1}, \mathbf{E}_{n-2}^{-1} \cdots \mathbf{E}_0^{-1}{n-1} )

        Example:

        .. runblock:: pycon

            >>> from roboticstoolbox import ETS
            >>> e = ETS.rz(j=2) * ETS.tx(1) * ETS.rx(j=3,flip=True) * ETS.tx(1)
            >>> print(e)
            >>> print(e.inv())
            >>> q = [1,2,3,4]
            >>> print(e.eval(q) * e.inv().eval(q))

        .. note:: It is essential to use explicit joint indices to account for
            the reversed order of the transforms.
        """

        inv = ETS()
        for e in reversed(self.data):
            # get the named tuple from the list, and convert to a dict
            etdict = e._asdict()

            # update the dict to make this an inverse
            if etdict['joint']:
                etdict['flip'] ^= True   # toggle flip status
            elif etdict['axis'][0] == 'C':
                etdict['T'] = trinv(etdict['T'])
            elif etdict['eta'] is not None:
                etdict['T'] = trinv(etdict['T'])
                etdict['eta'] = -etdict['eta']
            et = ETS()  # create a new ETS instance
            et.data = [self.ets_tuple(**etdict)]  # set it's data from the dict
            inv *= et
        return inv


class ETS(SuperETS):
    """
    This class implements an elementary transform sequence (ETS) for 3D

    :param arg: Function to compute ET value
    :type arg: callable
    :param η: The coordinate of the ET. If not supplied the ET corresponds
        to a variable ET which is a joint
    :type η: float, optional
    :param unit: angular unit, "rad" [default] or "deg"
    :type unit: str
    :param j: Explicit joint number within the robot
    :type j: int, optional
    :param flip: Joint moves in opposite direction
    :type flip: bool

    An instance can contain an elementary transform (ET) or an elementary
    transform sequence (ETS). It has list-like properties by subclassing
    UserList, which means we can perform indexing, slicing pop, insert, as well
    as using it as an iterator over its values.

    - ``ETS()`` an empty ETS list
    - ``ETS.XY(η)`` is a constant elementary transform
    - ``ETS.XY(η, 'deg')`` as above but the angle is expressed in degrees
    - ``ETS.XY()`` is a joint variable, the value is left free until evaluation
      time
    - ``ETS.XY(j=J)`` as above but the joint index is explicitly given, this
      might correspond to the joint number of a multi-joint robot.
    - ``ETS.XY(flip=True)`` as above but the joint moves in the opposite sense

    where ``XY`` is one of ``rx``, ``ry``, ``rz``, ``tx``, ``ty``, ``tz``.

    Example:

        .. runblock:: pycon

            >>> from roboticstoolbox import ETS
            >>> e = ETS.rz(0.3) # a single ET, rotation about z
            >>> len(e)
            >>> e = ETS.rz(0.3) * ETS.tx(2) # an ETS
            >>> len(e)                      # of length 2
            >>> e[1]                        # an ET sliced from the ETS

    :references:
        - Kinematic Derivatives using the Elementary Transform Sequence,
          J. Haviland and P. Corke

    :seealso: :func:`rx`, :func:`ry`, :func:`rz`, :func:`tx`,
        :func:`ty`, :func:`tz`
    """
    @property
    def s(self):
        if self.axis[1] == 'x':
            if self.axis[0] == 'R':
                return np.r_[0, 0, 0, 1, 0, 0]
            else:
                return np.r_[1, 0, 0, 0, 0, 0]
        elif self.axis[1] == 'y':
            if self.axis[0] == 'R':
                return np.r_[0, 0, 0, 0, 1, 0]
            else:
                return np.r_[0, 1, 0, 0, 0, 0]
        else:
            if self.axis[0] == 'R':
                return np.r_[0, 0, 0, 0, 0, 1]
            else:
                return np.r_[0, 0, 1, 0, 0, 0]

    @classmethod
    def rx(cls, eta=None, unit='rad', **kwargs):
        """
        Pure rotation about the x-axis

        :param η: rotation about the x-axis
        :type η: float
        :param unit: angular unit, "rad" [default] or "deg"
        :type unit: str
        :param j: Explicit joint number within the robot
        :type j: int, optional
        :param flip: Joint moves in opposite direction
        :type flip: bool
        :return: An elementary transform
        :rtype: ETS instance

        - ``ETS.rx(η)`` is an elementary rotation about the x-axis by a
          constant angle η
        - ``ETS.rx()`` is an elementary rotation about the x-axis by a variable
          angle, i.e. a revolute robot joint. ``j`` or ``flip`` can be set in
          this case.

        :seealso: :func:`ETS`, :func:`isrevolute`
        :SymPy: supported
        """
        return cls(trotx, axis='Rx', eta=eta, unit=unit, **kwargs)

    @classmethod
    def ry(cls, eta=None, unit='rad', **kwargs):
        """
        Pure rotation about the y-axis

        :param η: rotation about the y-axis
        :type η: float
        :param unit: angular unit, "rad" [default] or "deg"
        :type unit: str
        :param j: Explicit joint number within the robot
        :type j: int, optional
        :param flip: Joint moves in opposite direction
        :type flip: bool
        :return: An elementary transform
        :rtype: ETS instance

        - ``ETS.ry(η)`` is an elementary rotation about the y-axis by a
          constant angle η
        - ``ETS.ry()`` is an elementary rotation about the y-axis by a variable
          angle, i.e. a revolute robot joint. ``j`` or ``flip`` can be set in
          this case.

        :seealso: :func:`ETS`, :func:`isrevolute`
        :SymPy: supported
        """
        return cls(troty, axis='Ry', eta=eta, unit=unit, **kwargs)

    @classmethod
    def rz(cls, eta=None, unit='rad', **kwargs):
        """
        Pure rotation about the z-axis

        :param η: rotation about the z-axis
        :type η: float
        :param unit: angular unit, "rad" [default] or "deg"
        :type unit: str
        :param j: Explicit joint number within the robot
        :type j: int, optional
        :param flip: Joint moves in opposite direction
        :type flip: bool
        :return: An elementary transform
        :rtype: ETS instance

        - ``ETS.rz(η)`` is an elementary rotation about the z-axis by a
          constant angle η
        - ``ETS.rz()`` is an elementary rotation about the z-axis by a variable
          angle, i.e. a revolute robot joint. ``j`` or ``flip`` can be set in
          this case.

        :seealso: :func:`ETS`, :func:`isrevolute`
        :SymPy: supported
        """
        return cls(trotz, axis='Rz', eta=eta, unit=unit, **kwargs)

    @classmethod
    def tx(cls, eta=None, **kwargs):
        """
        Pure translation along the x-axis

        :param η: translation distance along the z-axis
        :type η: float
        :param j: Explicit joint number within the robot
        :type j: int, optional
        :param flip: Joint moves in opposite direction
        :type flip: bool
        :return: An elementary transform
        :rtype: ETS instance

        - ``ETS.tx(η)`` is an elementary translation along the x-axis by a
          distance constant η
        - ``ETS.tx()`` is an elementary translation along the x-axis by a
          variable distance, i.e. a prismatic robot joint. ``j`` or ``flip``
          can be set in this case.

        :seealso: :func:`ETS`, :func:`isprismatic`
        :SymPy: supported
        """

        # this method is 3x faster than using lambda x: transl(x, 0, 0)
        def axis_func(eta):
            return np.array([
                [1, 0, 0, eta],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])

        return cls(axis_func, axis='tx', eta=eta, **kwargs)

    @classmethod
    def ty(cls, eta=None, **kwargs):
        """
        Pure translation along the y-axis

        :param η: translation distance along the y-axis
        :type η: float
        :param j: Explicit joint number within the robot
        :type j: int, optional
        :param flip: Joint moves in opposite direction
        :type flip: bool
        :return: An elementary transform
        :rtype: ETS instance

        - ``ETS.ty(η)`` is an elementary translation along the y-axis by a
          distance constant η
        - ``ETS.ty()`` is an elementary translation along the y-axis by a
          variable distance, i.e. a prismatic robot joint. ``j`` or ``flip``
          can be set in this case.

        :seealso: :func:`ETS`, :func:`isprismatic`
        :SymPy: supported
        """
        def axis_func(eta):
            return np.array([
                [1, 0, 0, 0],
                [0, 1, 0, eta],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])

        return cls(axis_func, axis='ty', eta=eta, **kwargs)

        # return cls(SE3.Ty, axis='ty', eta=eta)

    @classmethod
    def tz(cls, eta=None, **kwargs):
        """
        Pure translation along the z-axis

        :param η: translation distance along the z-axis
        :type η: float
        :param j: Explicit joint number within the robot
        :type j: int, optional
        :param flip: Joint moves in opposite direction
        :type flip: bool
        :return: An elementary transform
        :rtype: ETS instance

        - ``ETS.tz(η)`` is an elementary translation along the z-axis by a
          distance constant η
        - ``ETS.tz()`` is an elementary translation along the z-axis by a
          variable distance, i.e. a prismatic robot joint. ``j`` or ``flip``
          can be set in this case.

        :seealso: :func:`ETS`, :func:`isprismatic`
        :SymPy: supported
        """
        def axis_func(eta):
            return np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, eta],
                [0, 0, 0, 1]
            ])

        return cls(axis_func, axis='tz', eta=eta, **kwargs)

    def jacob0(self, q=None, T=None):
        r"""
        Jacobian in base frame

        :param q: joint coordinates
        :type q: array_like
        :param T: ETS value as an SE(3) matrix if known
        :type T: ndarray(4,4)
        :return: Jacobian matrix
        :rtype: ndarray(6,n)

        ``jacob0(q)`` is the ETS Jacobian matrix which maps joint
        velocity to spatial velocity in the {0} frame.

        End-effector spatial velocity :math:`\nu = (v_x, v_y, v_z, \omega_x,
        \omega_y, \omega_z)^T` is related to joint velocity by
        :math:`{}^{e}\nu = {}^{e}\mathbf{J}_0(q) \dot{q}`.

        If ``ets.eval(q)`` is already computed it can be passed in as ``T`` to
        reduce computation time.

        An ETS represents the relative pose from the {0} frame to the end frame
        {e}. This is the composition of many relative poses, some constant and
        some functions of the joint variables, which we can write as
        :math:`\mathbf{E}(q)`.

        .. math::
        
            {}^0 T_e = \mathbf{E}(q) \in \mbox{SE}(3)

        The temporal derivative of this is the spatial
        velocity :math:`\nu` which is a 6-vector is related to the rate of
        change of joint coordinates by the Jacobian matrix.

        .. math::

           {}^0 \nu = {}^0 \mathbf{J}(q) \dot{q} \in \mathbb{R}^6

        This velocity can be expressed relative to the {0} frame or the {e}
        frame.

        :references:

            - `Kinematic Derivatives using the Elementary Transform Sequence, J. Haviland and P. Corke <https://arxiv.org/abs/2010.08696>`_

        :seealso: :func:`jacobe`, :func:`hessian0`
        """

        # TODO what is offset
        # if offset is None:
        #     offset = SE3()

        n = self.n  # number of joints
        q = getvector(q, n)

        if T is None:
            T = self.eval(q)

        # we will work with NumPy arrays for maximum speed
        T = T.A
        U = np.eye(4)
        j = 0
        J = np.zeros((6, n))

        for et in self:

            if et.isjoint:
                # joint variable
                # U = U @ link.A(q[j], fast=True)
                U = U @ et.axis_func(q[j])

                # TODO???
                # if link == to_link:
                #     U = U @ offset.A

                Tu = np.linalg.inv(U) @ T
                n = U[:3, 0]
                o = U[:3, 1]
                a = U[:3, 2]
                x = Tu[0, 3]
                y = Tu[1, 3]
                z = Tu[2, 3]

                if et.axis == 'Rz':
                    J[:3, j] = (o * x) - (n * y)
                    J[3:, j] = a

                elif et.axis == 'Ry':
                    J[:3, j] = (n * z) - (a * x)
                    J[3:, j] = o

                elif et.axis == 'Rx':
                    J[:3, j] = (a * y) - (o * z)
                    J[3:, j] = n

                elif et.axis == 'tx':
                    J[:3, j] = n
                    J[3:, j] = np.array([0, 0, 0])

                elif et.axis == 'ty':
                    J[:3, j] = o
                    J[3:, j] = np.array([0, 0, 0])

                elif et.axis == 'tz':
                    J[:3, j] = a
                    J[3:, j] = np.array([0, 0, 0])

                j += 1
            else:
                # constant transform
                U = U @ et.T()

        return J

    def jacobe(self, q=None, T=None):
        r"""
        Jacobian in base frame

        :param q: joint coordinates
        :type q: array_like
        :param T: ETS value as an SE(3) matrix if known
        :type T: ndarray(4,4)
        :return: Jacobian matrix
        :rtype: ndarray(6,n)

        ``jacobe(q)`` is the manipulator Jacobian matrix which maps joint
        velocity to end-effector spatial velocity.

        End-effector spatial velocity :math:`\nu = (v_x, v_y, v_z, \omega_x, \omega_y, \omega_z)^T`
        is related to joint velocity by :math:`{}^{e}\nu = {}^{e}\mathbf{J}_0(q) \dot{q}`.

        If ``ets.eval(q)`` is already computed it can be passed in as ``T`` to
        reduce computation time.

       :seealso: :func:`jacob`, :func:`hessian0`
        """

        if T is None:
            T = self.eval(q)

        return tr2jac(T.A) @ self.jacob0(q, T)

    def hessian0(self, q=None, J0=None):
        r"""
        Hessian in base frame

        :param q: joint coordinates
        :type q: array_like
        :param J0: Jacobian in {0} frame
        :type J0: ndarray(6,n)
        :return: Hessian matrix
        :rtype: ndarray(6,n,n)

        This method calculcates the Hessisan of the ETS. One of ``J0`` or
        ``q`` is required. If ``J0`` is already calculated for the joint
        coordinates ``q`` it can be passed in to to save computation time

        An ETS represents the relative pose from the {0} frame to the end frame
        {e}. This is the composition of many relative poses, some constant and
        some functions of the joint variables, which we can write as
        :math:`\mathbf{E}(q)`.

        .. math::

            {}^0 T_e = \mathbf{E}(q) \in \mbox{SE}(3)

        The temporal derivative of this is the spatial
        velocity :math:`\nu` which is a 6-vector is related to the rate of
        change of joint coordinates by the Jacobian matrix.

        .. math::

            {}^0 \nu = {}^0 \mathbf{J}(q) \dot{q} \in \mathbb{R}^6

        This velocity can be expressed relative to the {0} frame or the {e}
        frame.

        The temporal derivative of spatial velocity is spatial acceleration,
        which again can be expressed with respect to the {0} or {e} frames

        .. math::

            {}^0 \dot{\nu} = \mathbf{J}(q) \ddot{q} + \dot{\mathbf{J}}(q) \dot{q} \in \mathbb{R}^6 \\
                      &= \mathbf{J}(q) \ddot{q} + \dot{q}^T \mathbf{H}(q) \dot{q}

        The manipulator Hessian tensor :math:`H` maps joint velocity to
        end-effector spatial acceleration, expressed in the {0} coordinate
        frame.

        :references:
            - `Kinematic Derivatives using the Elementary Transform Sequence, J. Haviland and P. Corke <https://arxiv.org/abs/2010.08696>`_

        :seealso: :func:`jacob0`
        """

        n = self.n

        if J0 is None:
            if q is None:
                q = np.copy(self.q)
            else:
                q = getvector(q, n)

            J0 = self.jacob0(q)
        else:
            verifymatrix(J0, (6, n))

        H = np.zeros((6, n, n))

        for j in range(n):
            for i in range(j, n):

                H[:3, i, j] = np.cross(J0[3:, j], J0[:3, i])
                H[3:, i, j] = np.cross(J0[3:, j], J0[3:, i])

                if i != j:
                    H[:3, j, i] = H[:3, i, j]

        return H


class ETS2(SuperETS):
    """
    This class implements an elementary transform sequence (ETS) for 2D

    :param arg: Function to compute ET value
    :type arg: callable
    :param η: The coordinate of the ET. If not supplied the ET corresponds
        to a variable ET which is a joint
    :type η: float, optional
    :param unit: angular unit, "rad" [default] or "deg"
    :type unit: str
    :param j: Explicit joint number within the robot
    :type j: int, optional
    :param flip: Joint moves in opposite direction
    :type flip: bool

    An instance can contain an elementary transform (ET) or an elementary
    transform sequence (ETS). It has list-like properties by subclassing
    UserList, which means we can perform indexing, slicing pop, insert, as well
    as using it as an iterator over its values.

    - ``ETS()`` an empty ETS list
    - ``ETS.XY(η)`` is a constant elementary transform
    - ``ETS.XY(η, 'deg')`` as above but the angle is expressed in degrees
    - ``ETS.XY()`` is a joint variable, the value is left free until evaluation
      time
    - ``ETS.XY(j=J)`` as above but the joint index is explicitly given, this
      might correspond to the joint number of a multi-joint robot.
    - ``ETS.XY(flip=True)`` as above but the joint moves in the opposite sense

    where ``XY`` is one of ``r``, ``tx``, ``ty``.

    Example:

        .. runblock:: pycon

            >>> from roboticstoolbox import ETS2 as ETS
            >>> e = ETS.r(0.3)  # a single ET, rotation about z
            >>> len(e)
            >>> e = ETS.r(0.3) * ETS.tx(2)  # an ETS
            >>> len(e)                      # of length 2
            >>> e[1]                        # an ET sliced from the ETS

    :references:
        - Kinematic Derivatives using the Elementary Transform Sequence,
          J. Haviland and P. Corke

    :seealso: :func:`r`, :func:`tx`, :func:`ty`
    """

    @classmethod
    def r(cls, eta=None, unit='rad', **kwargs):
        """
        Pure rotation

        :param η: rotation angle
        :type η: float
        :param unit: angular unit, "rad" [default] or "deg"
        :type unit: str
        :param j: Explicit joint number within the robot
        :type j: int, optional
        :param flip: Joint moves in opposite direction
        :type flip: bool
        :return: An elementary transform
        :rtype: ETS instance

        - ``ETS.r(η)`` is an elementary rotation by a constant angle η
        - ``ETS.r()`` is an elementary rotation by a variable angle, i.e. a
          revolute robot joint. ``j`` or ``flip`` can be set in
          this case.

        .. note:: In the 2D case this is rotation around the normal to the
            xy-plane.

        :seealso: :func:`ETS`, :func:`isrevolute`
        """
        return cls(
            lambda theta: trot2(theta), axis='R', eta=eta, unit=unit, **kwargs)

    @classmethod
    def tx(cls, eta=None, **kwargs):
        """
        Pure translation along the x-axis

        :param η: translation distance along the z-axis
        :type η: float
        :param j: Explicit joint number within the robot
        :type j: int, optional
        :param flip: Joint moves in opposite direction
        :type flip: bool
        :return: An elementary transform
        :rtype: ETS instance

        - ``ETS.tx(η)`` is an elementary translation along the x-axis by a
          distance constant η
        - ``ETS.tx()`` is an elementary translation along the x-axis by a
          variable distance, i.e. a prismatic robot joint. ``j`` or ``flip``
          can be set in this case.

        :seealso: :func:`ETS`, :func:`isprismatic`
        """
        return cls(lambda x: transl2(x, 0), axis='tx', eta=eta, **kwargs)

    @classmethod
    def ty(cls, eta=None, **kwargs):
        """
        Pure translation along the y-axis

        :param η: translation distance along the y-axis
        :type η: float
        :param j: Explicit joint number within the robot
        :type j: int, optional
        :param flip: Joint moves in opposite direction
        :type flip: bool
        :return: An elementary transform
        :rtype: ETS instance

        - ``ETS.tx(η)`` is an elementary translation along the y-axis by a
          distance constant η
        - ``ETS.tx()`` is an elementary translation along the y-axis by a
          variable distance, i.e. a prismatic robot joint. ``j`` or ``flip``
          can be set in this case.

        :seealso: :func:`ETS`
        """
        return cls(lambda y: transl2(0, y), axis='ty', eta=eta, **kwargs)


if __name__ == "__main__":

    print(ETS.rx(0.2))
    print(ETS.rx(45, 'deg'))
    print(ETS.tz(0.75))
    e = ETS.rx(45, 'deg') * ETS.tz(0.75)
    print(e)
    print(e.eval())

    from roboticstoolbox import ETS
    e = ETS.rz() * ETS.tx(1) * ETS.rz() * ETS.tx(1)
    print(e.eval([0, 0]))
    print(e.eval([90, -90], 'deg'))
    a = e.pop()
    print(a)

    from spatialmath.base import symbol

    theta, d = symbol('theta, d')

    e = ETS.rx(theta) * ETS.tx(2) * ETS.rx(45, 'deg') * ETS.ry(0.2) * ETS.ty(d)
    print(e)

    e = ETS()
    e *= ETS.rx()
    e *= ETS.tz()
    print(e)

    print(e.__str__("θ{0}"))
    print(e.__str__("θ{1}"))

    e = ETS.rx() * ETS._CONST(SE3()) * ETS.tx(0.3)
    print(e)

    l1 = 0.672
    l2 = -0.2337
    l3 = 0.4318
    l4 = 0.0203
    l5 = 0.0837
    l6 = 0.4318

    e = ETS.tz(l1) * ETS.rz() * ETS.ry() * ETS.ty(l2) * ETS.tz(l3) * ETS.ry() \
        * ETS.tx(l4) * ETS.ty(l5) * ETS.tz(l6) * ETS.rz() * ETS.ry() * ETS.rz()
    print(e.joints())
    print(e.config)
    print(e.eval(np.zeros(6)))
    ec = e.compile()
    print(ec)
    print(ec.eval(np.zeros(6)))

    print(ETS.SE3(SE3.Rz(200, 'deg')))

    a = ETS.rx()
    b = ETS(a)
    print(b)
    a = ETS.tz()
    print(b)
    e = ETS.rz(j=5) * ETS.tx(1) * ETS.rx(j=7, flip=True) * ETS.tx(1)
    print(e)

    print(e.inv())

    q = [1, 2, 3, 4, 5, 6, 7, 8]
    print(e.eval(q))
    print(e.inv().eval(q))
    print(e.eval(q) * e.inv().eval(q))
