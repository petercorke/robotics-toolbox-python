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

from collections.abc import Iterable, MutableSequence
from copy import deepcopy
from functools import cached_property, wraps
from typing import TYPE_CHECKING, Any, Callable, TypeVar, cast, overload

import numpy as np
from numpy.random import uniform
from spatialmath.base import (
    getmatrix,
    getvector,
    issymbol,
)
from typing_extensions import Self

from roboticstoolbox.ets._ET import BaseET, _is_nothing, _offset_suffix
from roboticstoolbox.ets.fknm import (
    ETS_init,
)
from roboticstoolbox.tools.params import rtb_get_param
from roboticstoolbox.tools.types import ArrayLike, NDArray

if TYPE_CHECKING:  # pragma: nocover
    pass

#: the element type of an ETS: ``ET`` for ``ETS``, ``ET2`` for ``ETS2``
ETT = TypeVar("ETT", bound=BaseET)
_F = TypeVar("_F", bound=Callable[..., Any])


def _dirties_fknm(func: _F) -> _F:
    @wraps(func)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = func(self, *args, **kwargs)
        self._fknm_stale = True
        return result

    return wrapper  # type: ignore[return-value]


class BaseETS(MutableSequence[ETT]):
    def __init__(self) -> None:
        self._data: list[ETT] = []
        self._fknm_stale = True
        self._BaseETS__fknm: Any = None

    # ------------------------------------------------------------------
    # MutableSequence abstract methods
    # ------------------------------------------------------------------

    def _new(self, data: Iterable[ETT]) -> Self:
        """A new ETS of the same concrete type as ``self`` from elements"""
        # the concrete subclasses, ETS and ETS2, take the list of elements
        return self.__class__(data)  # type: ignore[call-arg]

    def __len__(self) -> int:
        return len(self._data)

    @overload
    def __setitem__(self, i: int, value: ETT) -> None: ...

    @overload
    def __setitem__(self, i: slice, value: Iterable[ETT]) -> None: ...

    @_dirties_fknm
    def __setitem__(self, i: Any, value: Any) -> None:
        self._data[i] = value

    @_dirties_fknm
    def __delitem__(self, i: int | slice) -> None:
        del self._data[i]

    @_dirties_fknm
    def insert(self, index: int, value: ETT) -> None:
        self._data.insert(index, value)

    def __repr__(self) -> str:
        return repr(self._data)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, BaseETS):
            return self._data == other._data
        return NotImplemented

    __hash__ = None  # type: ignore[assignment]

    def __radd__(self, other: Any) -> Any:
        # lets sum() work without an explicit start value, since its
        # default start is the int 0, which has no idea how to compose
        # with an ETS/ETS2
        if other == 0:
            return self
        return NotImplemented

    # ------------------------------------------------------------------
    # C handle: lazy build on first use after any mutation
    # ------------------------------------------------------------------

    @property
    def _fknm(self) -> Any:
        if self._fknm_stale:
            self._copy_to_cpp()
        return self._BaseETS__fknm

    def _copy_to_cpp(self) -> None:
        self._BaseETS__fknm = ETS_init(
            [et.fknm for et in self._data],  # type: ignore[attr-defined]
            self.n,
            self.m,
        )
        self._fknm_stale = False

    def __str__(self, q: str | None = None) -> str:
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
        s = ""
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
                    suffix = _offset_suffix(et.offset, et.isrotation)
                    s = f"{et.kind}({et._joint_name}{suffix})"
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
                    qvar = f"-{qvar}"
                s = f"{et.kind}({qvar}{_offset_suffix(et.offset, et.isrotation)})"
                j += 1

            elif et.isrotation:
                if issymbol(et.param):
                    s = f"{et.kind}({et.param})"
                else:
                    s = f"{et.kind}({cast(float, et.param) * 180 / np.pi:.4g}°)"

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

    def _repr_pretty_(self, p: Any, cycle: bool) -> None:
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

    def joint_idx(self) -> NDArray:
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

        return np.where([e.isjoint for e in self])[0]

    def joints(self) -> list[ETT]:
        """
        Get a list of the variable ETs with this ETS

        :returns: list of ETs that are joints
        :rtype: list[BaseET]

        Examples
        --------

        .. runblock:: pycon

            >>> from roboticstoolbox import ET
            >>> e = ET.Rz() * ET.tx(1) * ET.Rz() * ET.tx(1)
            >>> e.joints()

        """

        return [e for e in self if e.isjoint]

    def split(self, method: str = "last") -> list[Self]:
        """
        Split ETS into link segments

        :param method: one of ``"first"`` or ``"last"`` (default).
        :returns: ``[base, *segments, gripper]`` -- a list of length
            ``n_joints + 2``. ``base``/``gripper`` are empty ETS when the
            method has no concept of one.

        Split an ETS into segments representing links, plus a base and gripper segment.
        Unpack with ``base, *segments, gripper = ets.split(method)``.

        The behaviour depends on the ``method`` argument:

        * ``"first"``: each link segment begins with a joint and continues upto, but not
          including the next joint. Any constant ET before the first joint are part of
          the base. There are no gripper ET, they are included in the last link segment.
        * ``"last"``: each link segment ends with a joint and include all ET after the
          previous joint. Any constant ET after the last joint are part of the gripper.
          There are no base ET, they are included in the first link segment.

        .. runblock:: pycon

            >>> from roboticstoolbox import ET
            >>> e = ET.tz(1) * ET.Rx("q1") * ET.tx(2) * ET.Ry("q2") * ET.ty(3) * ET.Rz("q3") * ET.tz(4)
            >>> base, *segments, gripper = e.split("first")
            >>> print("|".join(str(_) for _ in segments), f"+ base={str(base)}, gripper={str(gripper)}")
            >>> base, *segments, gripper = e.split("last")
            >>> print("|".join(str(_) for _ in segments), f"+ base={str(base)}, gripper={str(gripper)}")
        """

        segments: list[Self] = []
        joint_idx = self.joint_idx()
        head = self[0:0]
        tail = self[0:0]

        match method.lower():

            case "first":
                head = self[0:joint_idx[0]]
                start = len(head)
                for k in joint_idx[1:]:
                    ets_j = self[start: k]
                    start = k
                    segments.append(self._new(ets_j))
                segments.append(self._new(self[joint_idx[-1]:]))
            case "last":
                start = 0
                for k in joint_idx:
                    ets_j = self[start : k + 1]
                    start = k + 1
                    segments.append(self._new(ets_j))
                tail = self[start:]
            case _:
                raise ValueError(f"unknown split method '{method}'")

        return [head] + segments + [tail]

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
            >>> e.jindex_set()

        """

        return set([self[j].jindex for j in self.joint_idx()])

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
            >>> e.jindices

        """

        return np.array([j.jindex for j in self.joints()])

    def _resolve_q(self, q: ArrayLike, allow_trajectory: bool = False) -> NDArray:
        """
        Resolve q to a full jindex-addressed vector

        :param q: joint coordinates, either *compact* (length :attr:`n`,
            positionally ordered to match this ETS's own :meth:`joints`) or
            *global* (length ``max(jindices) + 1``, addressed by each
            joint's global ``jindex`` -- e.g. the whole robot's ``q`` on a
            branched robot, when this ETS is only one branch of it)
        :param allow_trajectory: if True, ``q`` is normalised with
            :func:`getmatrix` (preserving a genuine ``(m, n)`` trajectory's
            row count, matching :meth:`eval`'s own convention); if False,
            ``q`` is flattened with :func:`getvector` first, matching
            :meth:`jacob0`/:meth:`jacobe`/:meth:`hessian0`/:meth:`hessiane`,
            none of which accept a trajectory
        :raises TypeError: if ``q`` isn't numeric (from :func:`getvector`)
        :raises ValueError: if ``q``'s length matches neither interpretation
        :returns: a global jindex-addressed array; unchanged if ``q`` was
            already global, scattered via :attr:`jindices` if ``q`` was
            compact

        This is the single place that disambiguates the two ``q`` shapes
        accepted by :meth:`eval`, :meth:`jacob0`, :meth:`jacobe`,
        :meth:`hessian0` and :meth:`hessiane` -- everything downstream of
        this (the C++ extension and the pure-Python fallback) only ever
        sees a global, jindex-addressed vector, exactly as before this
        method existed.

        Checking the compact length first means the common case -- a
        single-chain robot, or any ETS whose own joints already happen to
        carry global jindex 0..n-1 in order -- is unaffected: scattering
        via :attr:`jindices` is then just the identity.
        """
        n = self.n
        jindices = self.jindices

        q = np.asarray(
            getmatrix(q, (None, None)) if allow_trajectory else getvector(q, None)
        )

        if n == 0 or jindices.dtype == object:
            # no joints, or at least one joint has no jindex assigned yet
            # (e.g. an ETS2 instance before its lazy auto-assignment runs)
            # -- nothing to safely disambiguate, leave q exactly as given
            return q

        global_len = int(jindices.max()) + 1
        length = q.shape[-1]

        if length == n and n < global_len:
            # a genuine sub-chain (this ETS's own joints don't span the
            # full global jindex range) -- scatter positionally
            full = np.zeros(q.shape[:-1] + (global_len,), dtype=q.dtype)
            full[..., jindices] = q
            return full

        if length >= global_len:
            # already global-addressed; longer-than-needed is accepted
            # (matches the pre-existing convention of indexing q by
            # jindex and ignoring unused trailing entries). This also
            # covers n == global_len, where compact and global lengths
            # coincide -- always treating that as global (not scattering)
            # matters when this ETS's own joints don't carry jindex in
            # increasing order (e.g. a reversed chain from .inv()), where
            # scattering would silently reorder q instead of leaving it
            # alone.
            return q

        raise ValueError(
            f"q has length {length}, expected {n} (compact, positionally "
            f"matching this ETS's own joint order) or at least {global_len} "
            f"(global, addressed by jindex -- e.g. the whole robot's q)"
        )

    @property
    def qlim(self) -> NDArray:
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

        v: ArrayLike
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
    def qlim(self, new_qlim: ArrayLike) -> None:
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

    @property
    def data(self) -> list[ETT]:
        return self._data

    @data.setter
    def data(self, new_data: list[ETT]) -> None:
        self._data = new_data
        self._fknm_stale = True

    def inv(self) -> Self:
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

        return self._new([et.inv() for et in reversed(self._data)])

    @overload
    def __getitem__(self, i: int) -> ETT: ...

    @overload
    def __getitem__(self, i: slice) -> Self: ...

    def __getitem__(self, i: int | slice) -> ETT | Self:
        """
        Index or slice an ETS

        :param i: the index or slice
        :returns: elementary transform if ``i`` is an int, otherwise an ETS
            of the same concrete type as ``self`` (a slice of an ETS is
            still an ETS)

        Examples
        --------

        .. runblock:: pycon

            >>> from roboticstoolbox import ET
            >>> e = ET.Rz() * ET.tx(1) * ET.Rz() * ET.tx(1)
            >>> e[0]
            >>> e[1]
            >>> e[1:3]

        """
        if isinstance(i, slice):
            return self._new(self._data[i])
        return self._data[i]

    def __deepcopy__(self, memo: dict[int, Any]) -> Self:
        new_data = []

        for data in self:
            new_data.append(deepcopy(data))

        result = self._new(new_data)
        memo[id(self)] = result
        return result

    def plot(self, *args: Any, **kwargs: Any) -> None:
        """
        Plot the sequence as a robot

        The sequence is wrapped in a :class:`~roboticstoolbox.robot.Robot.Robot`
        (:class:`~roboticstoolbox.robot.Robot.Robot2` for a planar sequence) and
        its ``plot`` method is called, which see for the arguments.
        """
        # Deferred (like Robot/Robot2 above): BaseETS can't import the
        # concrete ETS without a cycle (ETS.py imports BaseETS from here).
        from roboticstoolbox.ets.ETS import ETS
        from roboticstoolbox.robot.Robot import Robot, Robot2

        if isinstance(self, ETS):
            robot = Robot(self)
        else:
            robot = Robot2(self)

        robot.plot(*args, **kwargs)

    def teach(self, *args: Any, **kwargs: Any) -> None:
        """
        Interactively explore the sequence as a robot

        The sequence is wrapped in a :class:`~roboticstoolbox.robot.Robot.Robot`
        (:class:`~roboticstoolbox.robot.Robot.Robot2` for a planar sequence) and
        its ``teach`` method is called, which see for the arguments.
        """
        from roboticstoolbox.ets.ETS import ETS
        from roboticstoolbox.robot.Robot import Robot, Robot2

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
        r"""
        Swap two adjacent transforms, in place

        :param i: index of the first transform
        :raises IndexError: if ``i`` and ``i+1`` are not both in the sequence
        :raises ValueError: if the two transforms cannot be swapped

        Swaps the transforms at indices ``i`` and ``i+1`` without changing what the
        sequence means.  This **changes the sequence it is called on** and returns
        ``None``; use :meth:`swapped` to get a modified copy and leave the original
        alone.

        There are two ways the transforms can swap.

        **They commute**, so they trade places unchanged.  Two transforms commute
        if they are

        - of the same kind, for example two translations along x
        - both translations, along any axes
        - a rotation and a translation along the axis of that rotation, for
          example ``Rz`` and ``tz``

        **One is a constant quarter or half turn**, a rotation :math:`R` of
        :math:`\pm 90^\circ` or :math:`180^\circ`.  The other transform :math:`B`
        is carried across it by being turned with it, since
        :math:`R B = (R B R^{-1}) R`.  The rotation is unchanged, and the other
        transform moves to the other side of it and changes axis, and its sign if
        the axis comes out reversed.  For example ``Rx(90°) ⊕ ty(d)`` is
        ``tz(d) ⊕ Rx(90°)``, and ``Rx(90°) ⊕ Ry(q)`` is ``Rz(q) ⊕ Rx(90°)``.  The
        other transform may be a translation or a rotation, a joint or a constant.
        A joint keeps its ``jindex`` and ``qlim``; if its axis comes out reversed
        it is flipped, and its offset changes sign.

        A rotation is a quarter turn if it is a whole number of quarter turns of a
        symbolic angle, such as ``pi/2`` or ``3*pi/2`` with ``pi`` from SymPy, or a
        float that is within :math:`10^{-9}` of one, which a model loaded from a file
        will have.  The new transform takes its parameter from the old one by a
        change of sign only, so it is as exact as the old one.

        Anything else, such as a rotation that is not a quarter turn and a
        translation along a different axis, does not swap and raises
        ``ValueError``, as do two quarter turns about different axes, since either
        could be the one that is carried across.

        .. note:: Joints may be swapped.  Each joint keeps its ``jindex``, so the
            order of the elements of ``q`` is unchanged even though the joints now
            appear in the opposite order in the sequence.

        Examples
        --------

        .. runblock:: pycon

            >>> from roboticstoolbox import ET
            >>> e = ET.Rz() * ET.tx(1) * ET.ty(2) * ET.Rz(1)
            >>> print(e)
            >>> e.swap(1)
            >>> print(e)
            >>> e = ET.Rz() * ET.tz(1) * ET.Rx() * ET.Rz(1)
            >>> e.swap(0)
            >>> print(e)

        A quarter turn carries the other transform with it:

        .. runblock:: pycon

            >>> from math import pi
            >>> e = ET.Rx(pi / 2) * ET.ty(1) * ET.tx(2)
            >>> print(e)
            >>> e.swap(0)
            >>> print(e)
            >>> e = ET.Rx(pi / 2) * ET.Ry()
            >>> e.swap(0)
            >>> print(e)
            >>> e = ET.Rx(-pi / 2) * ET.Ry()
            >>> e.swap(0)
            >>> print(e)

        .. versionchanged:: 1.5.0
            A transform can be swapped across a constant quarter or half turn.

        .. seealso:: :meth:`swapped` :meth:`merge`
        """
        e1, e2 = self._adjacent(i)

        if self._commute(e1, e2):
            self._data[i], self._data[i + 1] = e2, e1
        else:
            k1, k2 = self._quarter_turns(e1), self._quarter_turns(e2)
            if k1 is not None and k2 is None and e2.iselementary:
                # R B = (R B R^-1) R
                self._data[i], self._data[i + 1] = self._conjugate(e2, cast(str, e1.ax), k1), e1
            elif k2 is not None and k1 is None and e1.iselementary:
                # B R = R (R^-1 B R)
                self._data[i], self._data[i + 1] = e2, self._conjugate(e1, cast(str, e2.ax), -k2)
            elif k1 is not None and k2 is not None:
                raise ValueError(
                    f"transforms {e1.kind} and {e2.kind} are both quarter turns about "
                    "different axes, so it is not clear which to carry across the other"
                )
            else:
                raise ValueError(
                    f"transforms {e1.kind} and {e2.kind} do not commute, so they "
                    "cannot be swapped"
                )
        self._changed()

    def swapped(self, i: int) -> Self:
        """
        Swap two adjacent transforms, returning a copy

        :param i: index of the first transform
        :return: a copy of the sequence with transforms ``i`` and ``i+1`` swapped
        :raises IndexError: if ``i`` and ``i+1`` are not both in the sequence
        :raises ValueError: if the two transforms do not commute

        The sequence it is called on is not changed.  See :meth:`swap` for which
        transforms can be swapped.

        Examples
        --------

        .. runblock:: pycon

            >>> from roboticstoolbox import ET
            >>> e = ET.Rz() * ET.tx(1) * ET.ty(2) * ET.Rz(1)
            >>> print(e.swapped(1))
            >>> print(e)

        .. versionadded:: 1.5.0

        .. seealso:: :meth:`swap` :meth:`merged`
        """
        new = deepcopy(self)
        new.swap(i)
        return new

    def _adjacent(self, i: int) -> tuple[ETT, ETT]:
        """The transforms at ``i`` and ``i+1``, with the index checked."""
        if i < 0 or i >= len(self._data) - 1:
            raise IndexError(
                f"index {i} out of range, the sequence has {len(self._data)} transforms"
            )
        return self._data[i], self._data[i + 1]

    def _changed(self) -> None:
        """Forget what depends on the order and kind of the transforms."""
        self._fknm_stale = True
        self.__dict__.pop("jindices", None)  # invalidate cached_property

    @staticmethod
    def _commute(e1: Any, e2: Any) -> bool:
        """Do the two elementary transforms commute, in either order?"""
        if not (e1.iselementary and e2.iselementary):
            return False
        if e1.kind == e2.kind:
            return True
        if e1.istranslation and e2.istranslation:
            return True
        # a rotation and a translation along the axis of the rotation, there is no
        # such axis for a rotation in the plane, which is about z, out of the plane
        return e1.ax is not None and e1.ax == e2.ax

    @staticmethod
    def _quarter_turns(e: Any) -> int | None:
        """
        The number of quarter turns, modulo four, of a constant rotation

        None if ``e`` is not a constant rotation about an axis, or is not a whole
        number of quarter turns.  A symbolic angle must be exactly so; a float
        is within 1e-9 of it.
        """
        if not (e.isrotation and e.iselementary and not e.isjoint) or e.ax is None:
            return None
        angle = e.param
        if issymbol(angle):
            import sympy

            turns = sympy.simplify(angle / (sympy.pi / 2))
            return int(turns) % 4 if turns.is_integer else None
        turns = float(angle) / (np.pi / 2)
        whole = round(turns)
        return whole % 4 if abs(turns - whole) * (np.pi / 2) < 1e-9 else None

    @staticmethod
    def _axis_image(axis: str, quarters: int, target: str) -> tuple[str, int]:
        """
        Where a rotation sends an axis

        :param axis: the axis of the rotation, ``"x"``, ``"y"`` or ``"z"``
        :param quarters: the angle of the rotation in quarter turns, may be negative
        :param target: the axis, ``"x"``, ``"y"`` or ``"z"``, that is rotated
        :returns: the axis that ``target`` is turned onto, and ``+1`` if it points
            along that axis or ``-1`` if the opposite way

        For the unit vectors this is exact, with no trigonometry.
        """
        letters = "xyz"
        k = quarters % 4
        c, s = (1, 0, -1, 0)[k], (0, 1, 0, -1)[k]  # cosine, sine, exactly
        i = letters.index(axis)
        a, b = (i + 1) % 3, (i + 2) % 3
        t = letters.index(target)
        if t == i:
            return target, 1
        if t == a:  # e_a -> c e_a + s e_b
            return (letters[a], c) if c else (letters[b], s)
        return (letters[b], c) if c else (letters[a], -s)  # e_b -> -s e_a + c e_b

    @staticmethod
    def _conjugate(e: Any, axis: str, quarters: int) -> Any:
        """
        The transform ``R e R^-1``, for a rotation R of quarter turns about an axis

        :param e: an elementary transform, a translation or rotation, a joint or a
            constant
        :param axis: the axis of R, ``"x"``, ``"y"`` or ``"z"``
        :param quarters: the angle of R in quarter turns, which may be negative
        :returns: a new transform, of the other kind if the axis of ``e`` is turned

        Turning a transform about an axis is turning its own axis, so the result
        is the same kind of transform on the new axis.  If the new axis is the
        opposite direction the parameter has its sign changed, and a joint is
        flipped and its offset negated.  Everything else about a joint, its
        ``jindex`` and ``qlim``, is kept.
        """
        from roboticstoolbox.ets.ET import ET

        new_axis, sign = BaseETS._axis_image(axis, quarters, cast(str, e.ax))

        make = getattr(ET, e.kind[0] + new_axis)
        if e.isjoint:
            new = make(
                flip=bool(e.isflip) != (sign < 0), qlim=e.qlim, jindex=e.jindex
            )
            if e.offset != 0:
                new.offset = sign * e.offset
            if sign > 0 and e._joint_name is not None:
                new._joint_name = e._joint_name
            return new
        return make(sign * e.param)

    def merge(self, i: int) -> None:
        """
        Merge two adjacent transforms into one, in place

        :param i: index of the first transform
        :raises IndexError: if ``i`` and ``i+1`` are not both in the sequence
        :raises ValueError: if the transforms are not of the same kind, or both
            are joints

        Replaces the transforms at indices ``i`` and ``i+1`` with a single
        transform.  This **changes the sequence it is called on** and returns
        ``None``; use :meth:`merged` to get a modified copy and leave the original
        alone.

        The two transforms must be of the same kind, for example two ``tx`` or two
        ``Rz``.  There are two cases:

        - Two constants become a constant whose parameter is the sum of the two.
        - A joint and a constant become the same joint, with the constant added to
          its :attr:`~roboticstoolbox.ets.ET.ET.offset`, so ``Rz(q) * Rz(a)`` is
          ``Rz(q + a)``.  The joint keeps its ``jindex``, ``flip`` and ``qlim``.

        Two joints cannot be merged, since they are two variables.

        The parameters and offsets may be numeric or symbolic, or one of each.  If
        two constants add up to nothing the transform is removed altogether, rather
        than being left as a transform that does nothing.  An offset that adds up
        to nothing is zero but the joint stays.  The sum is nothing if it is

        - exactly zero, for example ``tx(x) * tx(-x)`` with ``x`` a SymPy symbol, or
          ``Rx(5) * Rx(-5)``
        - for a rotation, a whole number of turns of a symbolic angle, for example
          ``Rz(pi/2) * Rz(3*pi/2)`` with ``pi`` from SymPy.  A floating point sum
          such as ``Rz(pi/2) * Rz(3*pi/2)`` using :obj:`math.pi` is not exactly
          zero, so the rotation is kept.

        Merging does not look at the neighbours of the result, so if removing a
        transform leaves two mergeable ones adjacent, call :meth:`merge` again.

        Examples
        --------

        .. runblock:: pycon

            >>> from roboticstoolbox import ET
            >>> from math import pi
            >>> e = ET.Rz() * ET.tx(1) * ET.tx(2) * ET.Rz(1)
            >>> print(e)
            >>> e.merge(1)
            >>> print(e)
            >>> e = ET.tx(1) * ET.Rx(0.1) * ET.Rx(pi/2) * ET.tx(2)
            >>> print(e)
            >>> e.merge(1)
            >>> print(e)
            >>> e = ET.tx(1) * ET.Rx(0.5) * ET.Rx(-0.5) * ET.tx(2)
            >>> e.merge(1)
            >>> print(e)

        A constant merged into a joint becomes its offset, which may be symbolic:

        .. runblock:: pycon

            >>> from roboticstoolbox import ET
            >>> from spatialmath.base import symbol
            >>> a = symbol("a")
            >>> e = ET.tx(1) * ET.Rz() * ET.Rz(a) * ET.tx(2)
            >>> e.merge(1)
            >>> print(e)
            >>> e = ET.Rz(-a) * ET.Rz() * ET.Rz(a)
            >>> e.merge(1)
            >>> print(e)
            >>> e.merge(0)
            >>> print(e)

        .. versionchanged:: 1.5.0
            A joint can be merged with a constant, and a transform that adds up to
            nothing is removed.

        .. seealso:: :meth:`merged` :meth:`swap`
        """
        e1, e2 = self._adjacent(i)
        if e1.kind != e2.kind:
            raise ValueError(
                f"transforms {e1.kind} and {e2.kind} are not the same kind, so they "
                "cannot be merged"
            )
        if e1.isjoint and e2.isjoint:
            raise ValueError("both transforms are joints, they cannot be merged")

        if e1.isjoint or e2.isjoint:
            joint, const = (e1, e2) if e1.isjoint else (e2, e1)
            if const.param is None:
                raise ValueError(
                    "transforms are not elementary, they cannot be merged"
                )
            merged = deepcopy(joint)
            merged.offset = joint.offset + const.param
            self._data[i : i + 2] = [merged]
        else:
            if e1.param is None or e2.param is None:
                raise ValueError(
                    "transforms are not elementary, they cannot be merged"
                )
            total = e1.param + e2.param
            if _is_nothing(total, e1.isrotation):
                del self._data[i : i + 2]
            else:
                # a new transform, not a change to e1, which other sequences may
                # share
                merged = deepcopy(e1)
                merged.param = total
                self._data[i : i + 2] = [merged]
        self._changed()

    def merged(self, i: int) -> Self:
        """
        Merge two adjacent transforms, returning a copy

        :param i: index of the first transform
        :return: a copy of the sequence with transforms ``i`` and ``i+1`` merged
        :raises IndexError: if ``i`` and ``i+1`` are not both in the sequence
        :raises ValueError: if the transforms are not of the same kind, or both
            are joints

        The sequence it is called on is not changed.  See :meth:`merge` for which
        transforms can be merged and when the result is removed.

        Examples
        --------

        .. runblock:: pycon

            >>> from roboticstoolbox import ET
            >>> e = ET.Rz() * ET.tx(1) * ET.tx(2) * ET.Rz(1)
            >>> print(e.merged(1))
            >>> print(e)

        .. versionadded:: 1.5.0

        .. seealso:: :meth:`merge` :meth:`swapped`
        """
        new = deepcopy(self)
        new.merge(i)
        return new
