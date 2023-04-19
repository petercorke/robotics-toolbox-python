from abc import ABC
import numpy as np
import scipy as sp
from math import pi, sin, cos
import matplotlib.pyplot as plt
from spatialmath import base
import roboticstoolbox as rtb


class LandmarkMap:
    """
    Map of planar point landmarks

    :param map: map or number of landmarks
    :type map: ndarray(2, N) or int
    :param workspace: workspace or map bounds, defaults to 10
    :type workspace: scalar, array_like(2), array_like(4), optional
    :param verbose: display debug information, defaults to True
    :type verbose: bool, optional
    :param seed: random number seed, defaults to 0
    :type seed: int, optional
    :return: a landmark map object
    :rtype: LandmarkMap

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

    or any object that has a ``workspace`` attribute.

    Example:

    .. runblock:: pycon

        >>> from roboticstoolbox import LandmarkMap
        >>> map = LandmarkMap(20)
        >>> print(map)
        >>> print(map[3])  # coordinates of landmark 3

    The object is an iterator that returns consecutive landmark coordinates.

    :Reference:

        Robotics, Vision & Control, Chap 6,
        Peter Corke,
        Springer 2011

    See also :class:`~roboticstoolbox.mobile.sensors.RangeBearingSensor` :class:`~roboticstoolbox.mobile.EKF`
    """

    def __init__(self, map, workspace=10, verbose=True, seed=0):

        try:
            self._workspace = workspace.workspace
        except:
            self._workspace = base.expand_dims(workspace)

        if base.ismatrix(map, (2, None)):
            self._map = map
            self._nlandmarks = map.shape[1]
        elif isinstance(map, int):
            self._nlandmarks = map

            random = np.random.default_rng(seed)
            x = random.uniform(self._workspace[0], self._workspace[1], self._nlandmarks)
            y = random.uniform(self._workspace[2], self._workspace[3], self._nlandmarks)
            self._map = np.c_[x, y].T

        else:
            raise ValueError("bad type for map")

        self._verbose = verbose

    def __str__(self):
        # s = M.char() is a string showing map parameters in
        # a compact human readable format.
        ws = self._workspace
        s = f"LandmarkMap object with {self._nlandmarks} landmarks, workspace="
        s += f"({ws[0]}: {ws[1]}, {ws[2]}: {ws[3]})"
        return s

    def __repr__(self):
        return str(self)

    def __len__(self):
        """
        Number of landmarks in map

        :return: number of landmarks in the map
        :rtype: int
        """
        return self._nlandmarks

    @property
    def landmarks(self):
        """
        xy-coordinates of all landmarks

        :return: xy-coordinates for landmark points
        :rtype: ndarray(2, n)
        """
        return self._map

    @property
    def workspace(self):
        """
        Size of map workspace

        :return: workspace bounds [xmin, xmax, ymin, ymax]
        :rtype: ndarray(4)

        Returns the bounds of the workspace as specified by constructor
        option ``workspace``.
        """
        return self._workspace

    def __getitem__(self, k):
        """
        Get landmark coordinates from map

        :param k: landmark index
        :type k: int
        :return: coordinate :math:`(x,y)` of k'th landmark
        :rtype: ndarray(2)
        """
        return self._map[:, k]

    def plot(self, labels=False, block=None, **kwargs):
        """
        Plot landmark map

        :param labels: number the points on the plot, defaults to False
        :type labels: bool, optional
        :param block: block until figure is closed, defaults to False
        :type block: bool, optional
        :param kwargs: :meth:`~matplotlib.axes.Axes.plot` options

        Plot landmark points using Matplotlib options.  Default style is black
        crosses.
        """

        ax = base.plotvol2(self._workspace)
        ax.set_aspect("equal")

        ax.set_xlabel("x")
        ax.set_ylabel("y")

        if len(kwargs) == 0:
            kwargs = {
                "linewidth": 0,
                "marker": "x",
                "color": "black",
                "linestyle": "None",
            }

        if "label" not in kwargs:
            kwargs["label"] = "landmark"

        # plt.plot(self._map[0,:], self._map[1,:], , **kwargs)
        if labels:
            labels = "#{}"
        else:
            labels = None
        base.plot_point(self._map, text=labels, **kwargs)
        plt.grid(True)
        if block is not None:
            plt.show(block=block)


if __name__ == "__main__":
    import unittest
