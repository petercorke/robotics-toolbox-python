from __future__ import annotations

import numbers
import warnings
from typing import Any

import matplotlib.pyplot as plt
from roboticstoolbox.tools.plot import apply_block
import numpy as np
from spatialmath import base
from spatialmath.base.types import R2, R4, Points2

from roboticstoolbox.mobile.workspace import WorkspaceLike, resolve_workspace
from roboticstoolbox.tools.types import NDArray

#: default Matplotlib style of the landmarks, a user's options are merged over this
LANDMARK_STYLE: dict[str, Any] = {
    "linewidth": 0,
    "marker": "x",
    "color": "black",
    "linestyle": "None",
}


class LandmarkMap:
    """
    Map of planar point landmarks

    A LandmarkMap object represents a rectangular 2D environment with a number
    of point landmarks.

    The landmarks can be specified explicitly or be uniform randomly positioned
    inside a region defined by the workspace.  The workspace can be numeric:

    ==============  =======  =======
    ``workspace``   x-range  y-range
    ==============  =======  =======
    A (scalar)      -A:A     -A:A
    [A, B]           A:B      A:B
    [A, B, C, D]     A:B      C:D
    ==============  =======  =======

    or any object that has a ``workspace`` attribute, see :ref:`mobile-workspace`.

    Example:

    .. runblock:: pycon

        >>> from roboticstoolbox import LandmarkMap
        >>> map = LandmarkMap(20)
        >>> print(map)
        >>> print(map[3])  # coordinates of landmark 3

    The object is an iterator that returns consecutive landmark coordinates.

    :Reference:

        Robotics, Vision & Control in Python, 3rd edition, Chap 6,
        Peter Corke,
        Springer 2023

    :seealso: :class:`~roboticstoolbox.mobile.sensors.RangeBearingSensor` :class:`~roboticstoolbox.mobile.EKF.EKF`
    """

    def __init__(
        self,
        landmarks: Points2 | int | None = None,
        workspace: WorkspaceLike = 10,
        seed: int | None = 0,
        verbose: bool | None = None,
        map: Points2 | None = None,
    ) -> None:
        """
        Create a landmark map

        :param landmarks: the landmark coordinates, one column per landmark, or the
            number of landmarks to place at random in the workspace
        :type landmarks: ndarray(2, N) or int
        :param workspace: workspace or map bounds, defaults to 10
        :param seed: random number seed used when placing landmarks at random,
            defaults to 0
        :param verbose: deprecated, it had no effect
        :param map: deprecated, use ``landmarks``
        :raises ValueError: ``landmarks`` is not given, or is not a 2xN array or
            an integer

        .. deprecated:: 1.4.5
            ``verbose`` has no effect and ``map`` is renamed ``landmarks``.
        """
        if map is not None:
            warnings.warn(
                "map is deprecated, use landmarks instead",
                DeprecationWarning,
                stacklevel=2,
            )
            if landmarks is None:
                landmarks = map
        if verbose is not None:
            warnings.warn(
                "verbose is deprecated and has no effect",
                DeprecationWarning,
                stacklevel=2,
            )
        if landmarks is None:
            raise ValueError("landmarks must be an array of coordinates or a count")

        resolved = resolve_workspace(workspace)
        if resolved is None:
            raise ValueError("a landmark map requires a workspace")
        self._workspace: NDArray = resolved

        self._map: NDArray
        if isinstance(landmarks, np.ndarray) and base.ismatrix(landmarks, (2, None)):
            self._map = landmarks
            self._nlandmarks = landmarks.shape[1]
        elif isinstance(landmarks, numbers.Integral) and not isinstance(
            landmarks, bool
        ):
            self._nlandmarks = int(landmarks)

            random = np.random.default_rng(seed)
            x = random.uniform(self._workspace[0], self._workspace[1], self._nlandmarks)
            y = random.uniform(self._workspace[2], self._workspace[3], self._nlandmarks)
            self._map = np.c_[x, y].T

        else:
            raise ValueError("landmarks must be a 2xN array or an integer")

    def __str__(self) -> str:
        """
        Compact string description of the map

        :return: number of landmarks and the workspace
        """
        ws = self._workspace
        s = f"LandmarkMap object with {self._nlandmarks} landmarks, workspace="
        s += f"({ws[0]}: {ws[1]}, {ws[2]}: {ws[3]})"
        return s

    def __repr__(self) -> str:
        """
        Compact string description of the map, same as :meth:`__str__`
        """
        return str(self)

    def __len__(self) -> int:
        """
        Number of landmarks in map

        :return: number of landmarks in the map
        """
        return self._nlandmarks

    @property
    def landmarks(self) -> Points2:
        """
        xy-coordinates of all landmarks

        :return: xy-coordinates for landmark points, one column per landmark
        :rtype: ndarray(2, n)
        """
        return self._map

    @property
    def workspace(self) -> R4:
        """
        Size of map workspace

        :return: workspace bounds [xmin, xmax, ymin, ymax]

        Returns the bounds of the workspace as specified by constructor
        option ``workspace``.
        """
        return self._workspace

    def __getitem__(self, k: int) -> R2:
        """
        Get landmark coordinates from map

        :param k: landmark index
        :return: coordinate :math:`(x,y)` of k'th landmark
        """
        return self._map[:, k]

    def plot(
        self, labels: bool = False, block: bool | None = None, **kwargs: Any
    ) -> None:
        """
        Plot landmark map

        :param labels: number the points on the plot, defaults to False
        :param block: ``None`` leaves showing the figure to the environment (no
            ``plt.show`` call), ``False`` shows it and returns at once, ``True`` shows
            it and waits until it is closed; ignored by notebook backends,
            defaults to None
        :param kwargs: :meth:`~matplotlib.axes.Axes.plot` options, merged over
            :data:`LANDMARK_STYLE`

        Plot landmark points using Matplotlib options.  Default style is black
        crosses, and any options given override the corresponding default.
        """

        ax = base.plotvol2(self._workspace)
        ax.set_aspect("equal")

        ax.set_xlabel("x")
        ax.set_ylabel("y")

        kwargs = {"label": "landmark"} | LANDMARK_STYLE | kwargs

        text = "#{}" if labels else None
        base.plot_point(self._map, text=text, **kwargs)
        plt.grid(True)
        apply_block(block)
