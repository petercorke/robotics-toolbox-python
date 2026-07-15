#!/usr/bin/env python3

"""
@author: Jesse Haviland
@author: Peter Corke

Shared base for ETS (3D) and ETS2 (2D) - see roboticstoolbox.ets.ETS /
roboticstoolbox.ets.ETS2 for the concrete classes. This module exists
separately so ETS.py and ETS2.py can each import BaseETS without importing
each other (BaseETS is depended on by both, so it can't depend on either
without a cycle).
"""

from __future__ import annotations
from collections.abc import MutableSequence
from functools import wraps, cached_property
import numpy as np
from numpy.random import uniform
from numpy.linalg import inv, det, cond, svd
from spatialmath import SE3, SE2
from spatialmath.base import (
    getvector,
    issymbol,
    tr2jac,
    verifymatrix,
    tr2jac2,
    t2r,
    rotvelxform,
    simplify,
    getmatrix,
)
from roboticstoolbox.tools.params import rtb_get_param

from roboticstoolbox.ets.fknm import (
    ETS_init,
    ETS_fkine,
    ETS_jacob0,
    ETS_jacobe,
    ETS_hessian0,
    ETS_hessiane,
    IK_NR_c,
    IK_GN_c,
    IK_LM_c,
)
from copy import deepcopy
from roboticstoolbox.ets._ET import BaseET
from typing import overload, TypeVar
from typing import Literal as L
from roboticstoolbox.tools.types import ArrayLike, NDArray

T = TypeVar("T", bound="BaseETS")


def _dirties_fknm(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        result = func(self, *args, **kwargs)
        self._fknm_stale = True
        return result
    return wrapper


class BaseETS(MutableSequence):
    def __init__(self):
        self._data: list = []
        self._fknm_stale = True
        self._BaseETS__fknm = None

    # ------------------------------------------------------------------
    # MutableSequence abstract methods
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, i):
        return self._data[i]

    @_dirties_fknm
    def __setitem__(self, i, value):
        self._data[i] = value

    @_dirties_fknm
    def __delitem__(self, i):
        del self._data[i]

    @_dirties_fknm
    def insert(self, index: int, value) -> None:
        self._data.insert(index, value)

    def __repr__(self) -> str:
        return repr(self._data)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, BaseETS):
            return self._data == other._data
        return NotImplemented

    __hash__ = None  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # C handle: lazy build on first use after any mutation
    # ------------------------------------------------------------------

    @property
    def _fknm(self):
        if self._fknm_stale:
            self._copy_to_cpp()
        return self._BaseETS__fknm

    def _copy_to_cpp(self):
        self._BaseETS__fknm = ETS_init(
            [et.fknm for et in self._data],
            self.n,
            self.m,
        )
        self._fknm_stale = False

    def __str__(self, q: str | None = None):
        """
        Pretty prints the ETS

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

        :param q: control how joint variables are displayed
        :returns: Pretty printed ETS
        :rtype: str

        Examples
        --------

        .. runblock:: pycon

            >>> from roboticstoolbox import ET
            >>> e = ET.Rz() * ET.tx(1) * ET.Rz()
            >>> print(e[:2])
            >>> print(e)
            >>> print(e.__str__(""))
            >>> print(e.__str__("θ{0}"))  # numbering from 0
            >>> print(e.__str__("θ{1}"))  # numbering from 1
            >>> # explicit joint indices
            >>> e = ET.Rz(jindex=3) * ET.tx(1) * ET.Rz(jindex=4)
            >>> print(e)
            >>> print(e.__str__("θ{0}"))

        Angular parameters are converted to degrees, except if they
        are symbolic.

        .. runblock:: pycon
        >>> from roboticstoolbox import ET
        >>> from spatialmath.base import symbol
        >>> theta, d = symbol('theta, d')
        >>> e = ET.Rx(theta) * ET.tx(2) * ET.Rx(45, 'deg') * ET.Ry(0.2) * ET.ty(d)
        >>> str(e)

        """

        es = []
        j = 0
        c = 0
        s = None
        unicode = rtb_get_param("unicode")

        # An empty SE3
        if len(self._data) == 0:
            return "SE3()"

        if q is None:
            if len(self.joints()) > 1:
                q = "q{0}"
            else:
                q = "q"

        # For et in the object, display it, data comes from properties
        # which come from the named tuple
        for et in self._data:
            if et.isjoint:
                # A custom name from a string `param` descriptor (e.g.
                # "theta2") already encodes any leading sign itself, so it
                # takes over the whole "q0"/"-q0" formatting below rather
                # than combining with it.
                if et._joint_name is not None:
                    s = f"{et.kind}({et._joint_name})"
                    j += 1
                    es.append(s)
                    continue

                if q is not None:
                    if et.jindex is None:  # pragma: nocover  this is no longer possible
                        _j = j
                    else:
                        _j = et.jindex
                    qvar = q.format(
                        _j, _j + 1
                    )
                # else:
                #     qvar = ""

                if et.isflip:
                    s = f"{et.kind}(-{qvar})"
                else:
                    s = f"{et.kind}({qvar})"
                j += 1

            elif et.isrotation:
                if issymbol(et.param):
                    s = f"{et.kind}({et.param})"
                else:
                    s = f"{et.kind}({et.param * 180 / np.pi:.4g}°)"

            elif et.istranslation:
                try:
                    s = f"{et.kind}({et.param:.4g})"
                except TypeError:  # pragma: nocover
                    s = f"{et.kind}({et.param})"

            elif not et.iselementary:
                s = str(et)
                c += 1

            es.append(s)

        if unicode:
            return " \u2295 ".join(es)
        else:  # pragma: nocover
            return " * ".join(es)

    def _repr_pretty_(self, p, cycle):
        """
        Pretty string for IPython

        Print stringified version when variable is displayed in IPython, ie. on
        a line by itself.

        :param p: pretty printer handle (ignored)
        :param cycle: pretty printer flag (ignored)

        Examples
        --------

        In [1]: e
        Out [1]: R(q0) ⊕ tx(1) ⊕ R(q1) ⊕ tx(1)

        """

        print(self.__str__())  # pragma: nocover

    def joint_idx(self) -> list[int]:
        """
        Get index of joint transforms

        :returns: indices of transforms that are joints
        :rtype: ndarray

        Examples
        --------

        .. runblock:: pycon

            >>> from roboticstoolbox import ET
            >>> e = ET.Rz() * ET.tx(1) * ET.Rz() * ET.tx(1)
            >>> e.joint_idx()

        """

        return np.where([e.isjoint for e in self])[0]  # type: ignore

    def joints(self) -> list[ET]:
        """
        Get a list of the variable ETs with this ETS

        :returns: list of ETs that are joints
        :rtype: list[ET]

        Examples
        --------

        .. runblock:: pycon

            >>> from roboticstoolbox import ET
            >>> e = ET.Rz() * ET.tx(1) * ET.Rz() * ET.tx(1)
            >>> e.joints()

        """

        return [e for e in self if e.isjoint]

    def jindex_set(self) -> set[int]:  #
        """
        Get set of joint indices

        :returns: set of unique joint indices
        :rtype: set[int]

        Examples
        --------

        .. runblock:: pycon

            >>> from roboticstoolbox import ET
            >>> e = ET.Rz(jindex=1) * ET.tx(jindex=2) * ET.Rz(jindex=1) * ET.tx(1)
            >>> e.jointset()

        """

        return set([self[j].jindex for j in self.joint_idx()])  # type: ignore

    @cached_property
    def jindices(self) -> NDArray:
        """
        Get an array of joint indices

        :returns: array of unique joint indices
        :rtype: ndarray

        Examples
        --------

        .. runblock:: pycon

            >>> from roboticstoolbox import ET
            >>> e = ET.Rz(jindex=1) * ET.tx(jindex=2) * ET.Rz(jindex=1) * ET.tx(1)
            >>> e.jointset()

        """

        return np.array([j.jindex for j in self.joints()])  # type: ignore

    @property
    def qlim(self):
        r"""
        Get/Set Joint limits

        Limits are extracted from the link objects.  If joints limits are
        not set for:

        - a revolute joint [-𝜋. 𝜋] is returned
        - a prismatic joint an exception is raised

        :param new_qlim: new joint limits to set
        :type new_qlim: ndarray(2,n)
        :returns: array of joint limit values
        :rtype: ndarray(2,n)
        :raises ValueError: unset limits for a prismatic joint

        Examples
        --------

        .. runblock:: pycon

            >>> import roboticstoolbox as rtb
            >>> robot = rtb.models.DH.Puma560()
            >>> robot.qlim

        """

        limits = np.zeros((2, self.n))

        for i, et in enumerate(self.joints()):
            if et.isrotation:
                if et.qlim is None:
                    v = [-np.pi, np.pi]
                else:
                    v = et.qlim
            elif et.istranslation:
                if et.qlim is None:
                    raise ValueError("undefined prismatic joint limit")
                else:
                    v = et.qlim
            else:
                raise ValueError("Undefined Joint Type")  # pragma: nocover
            limits[:, i] = v

        return limits

    @qlim.setter
    def qlim(self, new_qlim: ArrayLike):
        new_qlim = np.array(new_qlim)

        if new_qlim.shape == (2,) and self.n == 1:
            new_qlim = new_qlim.reshape(2, 1)

        if new_qlim.shape != (2, self.n):
            raise ValueError("new_qlim must be of shape (2, n)")

        for j, i in enumerate(self.joint_idx()):
            et = self[i]
            et.qlim = new_qlim[:, j]
            self[i] = et

    @property
    def structure(self) -> str:
        """
        Joint structure string

        A string comprising the characters 'R' or 'P' which indicate the types
        of joints in order from left to right.

        :returns: a string indicating the joint types
        :rtype: str

        Examples
        --------

        .. runblock:: pycon

            >>> from roboticstoolbox import ET
            >>> e = ET.tz() * ET.tx(1) * ET.Rz() * ET.tx(1)
            >>> e.structure

        """

        return "".join(
            ["R" if self._data[i].isrotation else "P" for i in self.joint_idx()]
        )

    @property
    def n(self) -> int:
        """
        Number of joints

        :returns: the number of joints in the ETS
        :rtype: int

        Counts the number of joints in the ETS.

        Examples
        --------

        .. runblock:: pycon

            >>> from roboticstoolbox import ET
            >>> e = ET.Rx() * ET.tx(1) * ET.tz()
            >>> e.n

        See Also
        --------
        :func:`joints`

        """

        return sum(1 for et in self._data if et.isjoint)

    @property
    def m(self) -> int:
        """
        Number of transforms

        :returns: the number of transforms in the ETS
        :rtype: int

        Counts the number of transforms in the ETS.

        Examples
        --------

        .. runblock:: pycon

            >>> from roboticstoolbox import ET
            >>> e = ET.Rx() * ET.tx(1) * ET.tz()
            >>> e.m

        """

        return len(self._data)

    @overload
    def data(self: "ETS") -> list[ET]: ...  # pragma: nocover

    @overload
    def data(self: "ETS2") -> list[ET2]: ...  # pragma: nocover

    @property
    def data(self):
        return self._data

    @data.setter
    @overload
    def data(self: "ETS", new_data: list[ET]): ...  # pragma: nocover

    @data.setter
    @overload
    def data(self: "ETS", new_data: list[ET2]): ...  # pragma: nocover

    @data.setter
    def data(self, new_data):
        self._data = new_data
        self._fknm_stale = True

    def inv(self: T) -> T:
        r"""
        Inverse of ETS

        The inverse of a given ETS.  It is computed as the inverse of the
        individual ETs in the reverse order.

        .. math::

            (\mathbf{E}_0, \mathbf{E}_1 \cdots \mathbf{E}_{n-1} )^{-1} = (\mathbf{E}_{n-1}^{-1}, \mathbf{E}_{n-2}^{-1} \cdots \mathbf{E}_0^{-1}{n-1} )

        :returns: Inverse of the ETS

        Examples
        --------

        .. runblock:: pycon

            >>> from roboticstoolbox import ET
            >>> e = ET.Rz(jindex=2) * ET.tx(1) * ET.Rx(jindex=3,flip=True) * ET.tx(1)
            >>> print(e)
            >>> print(e.inv())

        .. rubric:: Notes

        - It is essential to use explicit joint indices to account for
            the reversed order of the transforms.

        """

        return self.__class__([et.inv() for et in reversed(self._data)])  # type: ignore[call-arg]

    @overload
    def __getitem__(self: "BaseETS", i: int) -> BaseET: ...

    @overload
    def __getitem__(self: "ETS", i: int) -> ET: ...

    @overload
    def __getitem__(self: "ETS", i: slice) -> list[ET]: ...

    @overload
    def __getitem__(self: "ETS2", i: int) -> ET2: ...

    @overload
    def __getitem__(self: "ETS2", i: slice) -> list[ET2]: ...

    def __getitem__(self, i):
        """
        Index or slice an ETS

        :param i: the index or slice
        :returns: elementary transform

        Examples
        --------

        .. runblock:: pycon

            >>> from roboticstoolbox import ET
            >>> e = ET.Rz() * ET.tx(1) * ET.Rz() * ET.tx(1)
            >>> e[0]
            >>> e[1]
            >>> e[1:3]

        """
        return self._data[i]  # can be [2] or slice, eg. [3:5]

    def __deepcopy__(self, memo):
        new_data = []

        for data in self:
            new_data.append(deepcopy(data))

        cls = self.__class__
        result = cls(new_data)
        memo[id(self)] = result
        return result

    def plot(self, *args, **kwargs):
        from roboticstoolbox.robot.Robot import Robot, Robot2
        # Deferred (like Robot/Robot2 above): BaseETS can't import the
        # concrete ETS without a cycle (ETS.py imports BaseETS from here).
        from roboticstoolbox.ets.ETS import ETS

        if isinstance(self, ETS):
            robot = Robot(self)
        else:
            robot = Robot2(self)

        robot.plot(*args, **kwargs)

    def teach(self, *args, **kwargs):
        from roboticstoolbox.robot.Robot import Robot, Robot2
        from roboticstoolbox.ets.ETS import ETS

        if isinstance(self, ETS):
            robot = Robot(self)
        else:
            robot = Robot2(self)

        robot.teach(*args, **kwargs)

    def random_q(self, i: int = 1) -> NDArray:
        """
        Generate a random valid joint configuration

        :param i: number of configurations to generate
        :returns: random joint configuration
        :rtype: ndarray(n,) or ndarray(i,n)

        Generates a random q vector within the joint limits defined by
        ``self.qlim``.

        Examples
        --------

        .. runblock:: pycon

            >>> import roboticstoolbox as rtb
            >>> robot = rtb.models.Panda()
            >>> ets = robot.ets()
            >>> q = ets.random_q()
            >>> q

        """

        if i == 1:
            q = np.zeros(self.n)

            for i in range(self.n):
                q[i] = uniform(self.qlim[0, i], self.qlim[1, i])

        else:
            q = np.zeros((i, self.n))

            for j in range(i):
                for i in range(self.n):
                    q[j, i] = uniform(self.qlim[0, i], self.qlim[1, i])

        return q

    def swap(self, i: int) -> None:
        """
        Swap two transforms in the ETS

        :param i: index of first transform

        Swaps the two transforms at indices ``i`` and ``i+1``.  This is useful for
        changing the order of commutative transforms in an ETS.

        Examples
        --------

        .. runblock:: pycon

            >>> from roboticstoolbox import ET
            >>> e = ET.Rz() * ET.tx(1) * ET.Rx() * ET.Rz(1)
            >>> print(e)
            >>> e.swap(1)
            >>> print(e)

        """
        if i < 0 or i >= len(self._data) - 1:
            raise IndexError("Index out of range")  # pragma: nocover
        
        e1 = self._data[i]
        e2 = self._data[i + 1]
        if e1.kind == e2.kind:
            self._data[i], self._data[i + 1] = self._data[i + 1], self._data[i]
            self._fknm_stale = True
        else:
            raise ValueError("Transforms are not commutative")  # pragma: nocover
        
    def merge(self, i: int) -> None:
        """
        Merge two transforms in the ETS

        :param i: index of first transform

        Merges the two transforms at indices ``i`` and ``i+1``.  This is useful for
        reducing the number of transforms in an ETS.

        Examples
        --------

        .. runblock:: pycon

            >>> from roboticstoolbox import ET
            >>> from math import pi
            >>> e = ET.Rz() * ET.tx(1) * ET.tx(2) * ET.Rz(1)
            >>> print(e)
            >>> e.merge(1)
            >>> print(e)
            >>> e = ET.tx(1) * ET.Rx() * ET.Rx(pi/2) * ET.tx(2)
            >>> print(e)
            >>> e.merge(1)
            >>> print(e)

        """
        if i < 0 or i >= len(self._data) - 1:
            raise IndexError("Index out of range")  # pragma: nocover
        e1 = self._data[i]
        e2 = self._data[i + 1]
        if e1.kind != e2.kind:
            raise ValueError("Transforms are not the same type")  # pragma: nocover

        elif  (e1.isjoint + e2.isjoint) == 2:
            raise ValueError("Transforms are both joints")  # pragma: nocover

        else:
            self._data[i].param = e1.param + e2.param
            del self._data[i + 1]
            self._fknm_stale = True

