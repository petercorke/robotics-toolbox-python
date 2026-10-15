from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as sp
import spatialmath.base as smb
from spatialmath.base.types import R2, R4, ArrayLike2, ArrayLike4
from spatialmath.geom2d import Polygon2
from typing_extensions import Self

from roboticstoolbox.mobile.workspace import WorkspaceLike, resolve_workspace
from roboticstoolbox.tools.types import NDArray

if TYPE_CHECKING:
    from matplotlib.axes import Axes


class BaseMap(ABC):
    def __init__(
        self,
        workspace: WorkspaceLike | None = None,
        name: str | None = None,
    ) -> None:
        """
        Abstract base class for maps

        :param workspace: dimensions of the map in world coordinates, a scalar,
            2-vector, 4-vector or an object with a ``workspace`` attribute, see
            :ref:`mobile-workspace`, defaults to None
        :param name: name of the map, defaults to None

        The workspace can be specified in several ways:

        ==============  =======  =======
        ``workspace``   x-range  y-range
        ==============  =======  =======
        A (scalar)      -A:A     -A:A
        [A, B]           A:B      A:B
        [A, B, C, D]     A:B      C:D
        ==============  =======  =======
        """
        # a scalar, 2-vector, 4-vector or an object with a workspace attribute
        self._workspace: NDArray | None = resolve_workspace(workspace)
        self._name = name

    @property
    def width(self) -> float | None:
        """
        Width of the workspace in world units

        :return: x-extent of the workspace, None if there is no workspace
        """
        if self._workspace is None:
            return None
        return self._workspace[1] - self._workspace[0]

    @property
    def height(self) -> float | None:
        """
        Height of the workspace in world units

        :return: y-extent of the workspace, None if there is no workspace
        """
        if self._workspace is None:
            return None
        return self._workspace[3] - self._workspace[2]


class BaseOccupancyGrid(BaseMap):
    def __init__(
        self,
        grid: NDArray | None = None,
        origin: ArrayLike2 = (0, 0),
        value: float | bool = 0,
        cellsize: float = 1,
        workspace: WorkspaceLike | None = None,
        name: str | None = None,
    ) -> None:
        """
        Occupancy grid (superclass)

        :param grid: occupancy grid as a NumPy array, one row per y and one
            column per x, defaults to None
        :type grid: ndarray(N,M)
        :param origin: world coordinates of the grid element [0,0], defaults to (0, 0)
        :param value: initial value of cells if the grid is created from
            ``workspace``, defaults to 0
        :param cellsize: cell size in world units, defaults to 1
        :param workspace: extent of the grid in world coordinates, used if
            ``grid`` is not given, see :ref:`mobile-workspace`
        :param name: name of the map, defaults to None
        :raises ValueError: neither ``grid`` nor ``workspace`` is given

        This object supports a user-defined coordinate system and grid size.
        World coordinates are converted to grid coordinates to lookup the
        occupancy status.

        The grid can be initialized by:

        - a 2D NumPy array, ``grid``, with ``origin`` and ``cellsize``
        - specifying ``workspace``, ``cellsize`` and ``value`` arguments
        """
        super().__init__(workspace=workspace, name=name)

        self._grid: NDArray
        self._origin: NDArray
        if grid is not None:
            self._grid = grid
            self._origin = smb.getvector(origin, 2)

        elif self._workspace is not None:
            # the array is indexed [row, column] = [y, x]
            width = self._workspace[1] - self._workspace[0]
            height = self._workspace[3] - self._workspace[2]
            self._grid = np.full(
                np.floor(np.r_[height, width] / cellsize).astype(int) + 1, value
            )
            self._origin = np.r_[self._workspace[0], self._workspace[2]]

        else:
            raise ValueError("an occupancy grid needs a grid array or a workspace")

        self._cellsize = cellsize

    def copy(self) -> Self:
        """
        Copy an occupancy grid (superclass)

        :return: copy of the occupancy grid
        """
        return self.__class__(
            self._grid.copy(),
            cellsize=self._cellsize,
            origin=self._origin,
            name=self._name,
        )

    def __repr__(self) -> str:
        """
        Compact string description of occupancy grid, same as :meth:`__str__`
        """
        return str(self)

    def __str__(self) -> str:
        """
        Compact string description of occupancy grid (superclass)

        :return: summary of occupancy grid characteristics
        :rtype: str
        """
        s = self.__class__.__name__
        if self._name is not None:
            s += f"[{self._name}]"
        s += f": {self._grid.shape[1]} x {self._grid.shape[0]}"
        s += f", cell size={self._cellsize}"
        s += f", x = [{self.xmin}, {self.xmax}], y = [{self.ymin}, {self.ymax}]"
        return s

    @property
    def grid(self) -> NDArray:
        """
        Occupancy grid as a NumPy array (superclass)

        :return: binary occupancy grid
        :rtype: ndarray(N,M) of bool

        If :meth:`inflate` has been called, this will return the inflated
        occupancy grid.
        """
        return self._grid

    @property
    def xmin(self) -> float:
        """
        Minimum x-coordinate of this grid (superclass)

        :return: minimum world x-coordinate
        :rtype: float
        """
        return self._origin[0]

    @property
    def xmax(self) -> float:
        """
        Maximum x-coordinate of this grid (superclass)

        :return: maximum world x-coordinate
        :rtype: float
        """
        return (self._grid.shape[1] - 1) * self._cellsize + self._origin[0]

    @property
    def ymin(self) -> float:
        """
        Minimum y-coordinate of this grid (superclass)

        :return: minimum world y-coordinate
        :rtype: float
        """
        return self._origin[1]

    @property
    def ymax(self) -> float:
        """
        Maximum y-coordinate of this grid (superclass)

        :return: maximum world y-coordinate
        :rtype: float
        """
        return (self._grid.shape[0] - 1) * self._cellsize + self._origin[1]

    @property
    def shape(self) -> tuple[int, ...]:
        """
        Shape of the occupancy grid array (superclass)

        :return: shape of the occupancy grid array
        :rtype: 2-tuple

        This is the shape of the NumPy array that holds the occupancy grid.
        """
        return self._grid.shape

    @property
    def maxdim(self) -> float:
        """
        Maximum dimension of grid in world coordinates (superclass)

        :return: maximum side length of the occupancy grid
        :rtype: float
        """
        return max(self.grid.shape) * self._cellsize

    @property
    def cellsize(self) -> float:
        """
        Cell size in world units (superclass)

        :return: side length of a grid cell
        """
        return self._cellsize

    @property
    def workspace(self) -> R4:
        """
        Bounds of the occupancy grid in world coordinates (superclass)

        :return: workspace bounds [xmin, xmax, ymin, ymax]

        Returns the bounds of the occupancy grid in world coordinates.
        """
        return np.r_[self.xmin, self.xmax, self.ymin, self.ymax]

    @property
    def name(self) -> str | None:
        """
        Occupancy grid name (superclass)

        :return: name of the occupancy grid
        :rtype: str
        """
        return self._name

    @name.setter
    def name(self, name: str | None) -> None:
        """
        Set occupancy grid name (superclass)

        :param name: new name of the occupancy grid
        :type name: str
        """
        self._name = name

    def set(self, region: ArrayLike4, value: float | bool) -> None:
        """
        Set region of map (superclass)

        :param region: the region in world coordinates, [xmin, xmax, ymin, ymax]
        :param value: value to set cells to

        The cells that contain the corners of the region, and all those between
        them, are set.
        """
        bl = self.w2g([region[0], region[2]])
        tr = self.w2g([region[1], region[3]])
        self.grid[bl[1] : tr[1] + 1, bl[0] : tr[0] + 1] = value

    def grid_to_world(self, p: ArrayLike2) -> R2:
        """
        Convert grid coordinate to world coordinate (superclass)

        :param p: grid coordinate (column, row)
        :return: world coordinate (x, y)

        The grid cell size and offset are used to convert occupancy grid
        coordinate ``p`` to a world coordinate.

        ``g2w`` is a shorthand for this method.
        """
        pv = smb.getvector(p, 2)
        return pv * self._cellsize + self._origin

    # the short names are used throughout the planners and pose graph
    g2w = grid_to_world

    def world_to_grid(self, p: ArrayLike2) -> R2:
        """
        Convert world coordinate to grid coordinate (superclass)

        :param p: world coordinate (x, y)
        :return: grid coordinate (column, row)

        The grid cell size and offset are used to convert ``p`` to an occupancy
        grid coordinate.  The grid coordinate is rounded and cast to integer
        value. No check is made on the validity of the coordinate.

        ``w2g`` is a shorthand for this method.
        """
        pv = smb.getvector(p, 2)
        return (np.round((pv - self._origin) / self._cellsize)).astype(int)

    w2g = world_to_grid

    def plot(
        self,
        map: NDArray | None = None,
        ax: Axes | None = None,
        block: bool | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Plot the occupancy grid (superclass)

        :param map: array which is plotted instead of the grid, must be same
            size as the occupancy grid, defaults to None
        :type map: ndarray(N,M)
        :param ax: Matplotlib axes to plot into, defaults to None (the current axes)
        :param block: block until plot is dismissed, defaults to None
        :param kwargs: arguments passed to ``imshow``

        The grid is plotted as an image but with axes in world coordinates.

        The grid is a NumPy boolean array which has values 0 (false=unoccupied)
        and 1 (true=occupied).  Passing a `cmap` option to imshow can be used
        to control the displayed color of free space and obstacles.

        """

        ax = smb.axes_logic(ax, 2)

        if map is None:
            map = self._grid
            kwargs["extent"] = self.workspace

        ax.imshow(map, origin="lower", interpolation=None, **kwargs)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        if block is not None:
            plt.show(block=block)

    def line_world(self, p1: ArrayLike2, p2: ArrayLike2) -> NDArray:
        """
        Get index of cells along a line segment (superclass)

        :param p1: start, in world coordinates
        :param p2: end, in world coordinates
        :return: index into the raveled grid
        :rtype: ndarray(N)

        Get the indices of cells along a line segment defined by the end
        points given in world coordinates.

        The returned indices can be applied to a raveled view of the grid.

        ``line_w`` is a shorthand for this method.

        :seealso: :meth:`ravel` :meth:`world_to_grid`
        """

        gp1 = self.world_to_grid(p1)
        gp2 = self.world_to_grid(p2)

        return self._line(gp1, gp2)

    line_w = line_world

    def _line(self, p1: ArrayLike2, p2: ArrayLike2) -> NDArray:
        x, y = smb.bresenham(p1, p2)
        z = np.ravel_multi_index(np.vstack((y, x)), self.grid.shape)
        return z

    @property
    def ravel(self) -> NDArray:
        """
        Ravel the grid (superclass)

        :return: 1D view of the occupancy grid
        :rtype: ndarray(N)
        """
        return self._grid.reshape(-1)


class BinaryOccupancyGrid(BaseOccupancyGrid):
    def __init__(
        self,
        grid: NDArray | BinaryOccupancyGrid | None = None,
        origin: ArrayLike2 = (0, 0),
        value: float | bool = 0,
        cellsize: float = 1,
        workspace: WorkspaceLike | None = None,
        name: str | None = None,
    ) -> None:
        """
        Create a binary occupancy grid instance

        :param grid: occupancy grid as a NumPy array, or another binary
            occupancy grid, defaults to None
        :type grid: ndarray(N,M) or BinaryOccupancyGrid
        :param origin: world coordinates of the grid element [0,0], defaults to (0, 0)
        :param value: initial value of cells if the grid is created from
            ``workspace``, defaults to 0
        :param cellsize: cell size in world units, defaults to 1
        :param workspace: extent of the grid in world coordinates, used if
            ``grid`` is not given, see :ref:`mobile-workspace`
        :param name: name of the map, defaults to None
        :raises ValueError: ``grid`` is not an array or an occupancy grid, or
            neither ``grid`` nor ``workspace`` is given

        The array is kept internally as a bool array. Cells are set to True
        (occupied) corresponding to input values > 0.

        This object supports a user-defined coordinate system and grid size.
        World coordinates are converted to grid coordinates to lookup the
        occupancy status.

        Example:

        .. runblock:: pycon

            >>> from roboticstoolbox import BinaryOccupancyGrid
            >>> import numpy as np
            >>> og = BinaryOccupancyGrid(np.zeros((5,5)))
            >>> print(og)
            >>> og = BinaryOccupancyGrid(workspace=[-5,5], cellsize=0.1, value=0)
            >>> print(og)

        :seealso: :class:`OccupancyGrid`
        """

        if grid is not None:
            if isinstance(grid, np.ndarray):
                grid = grid.astype(bool)
            elif isinstance(grid, BinaryOccupancyGrid):
                grid = grid.grid
            else:
                raise ValueError("argument must be NumPy array or BinaryOccupancyGrid")

        super().__init__(
            grid=grid,
            origin=origin,
            value=value,
            cellsize=cellsize,
            workspace=workspace,
            name=name,
        )

    def __str__(self) -> str:
        """
        Compact string description of the occupancy grid, including the
        percentage of cells that are occupied
        """
        s = super().__str__()

        ncells = np.prod(self._grid.shape)
        nobs = self._grid.sum()
        s += f", {nobs / ncells * 100:.1f}% occupied"
        return s

    def isoccupied(self, p: ArrayLike2) -> bool:
        """
        Test if coordinate is occupied

        :param p: world coordinate (x, y)
        :return: occupancy status of corresponding grid cell

        The grid cell size and offset are used to convert ``p`` to an occupancy
        grid coordinate.  The grid coordinate is rounded and cast to integer
        value.  If the coordinate is outside the bounds of the occupancy grid
        it is considered to be occupied.

        :seealso: :meth:`w2g`
        """
        c, r = self.w2g(p)
        try:
            return self._grid[r, c]
        except IndexError:
            return True

    def inflate(self, radius: float) -> None:
        """
        Inflate obstacles

        :param radius: radius of circular structuring element in world units

        A circular structuring element is created and used to dilate the
        stored occupancy grid.

        Successive calls to ``inflate`` will compound the inflation.

        :seealso: :func:`scipy.ndimage.binary_dilation`
        """
        # Generate a circular structuring element
        r = round(radius / self._cellsize)
        Y, X = np.meshgrid(np.arange(-r, r + 1), np.arange(-r, r + 1))
        SE = X**2 + Y**2 <= r**2
        SE = SE.astype(int)

        # do the inflation using SciPy
        self._grid = sp.binary_dilation(self._grid, SE)


class OccupancyGrid(BaseOccupancyGrid):
    """
    General occupancy grid


    The elements of the array are floats and can represent occupancy
    probability or traversal cost.

    Example:

    .. runblock:: pycon

        >>> from roboticstoolbox import OccupancyGrid
        >>> import numpy as np
        >>> og = OccupancyGrid(np.zeros((5,5)))
        >>> print(og)
        >>> og = OccupancyGrid(workspace=[-5,5], cellsize=0.1, value=0.5)
        >>> print(og)

    :seealso: :class:`BinaryOccupancyGrid`
    """

    def __str__(self) -> str:
        """
        Compact string description of the occupancy grid, including the data
        type and the minimum, maximum and mean of the cell values
        """
        s = super().__str__()

        g = self._grid
        s += f", dtype {g.dtype}"
        s += f", min {g.min()}, max {g.max()}, mean {g.mean()}"
        return s


class PolygonMap(BaseMap):
    def __init__(
        self,
        workspace: WorkspaceLike | None = None,
        polygons: list[Polygon2] | None = None,
    ) -> None:
        """
        Polygonal obstacle map

        :param workspace: dimensions of 2D plot area, a scalar, 2-vector,
            4-vector or an object with a ``workspace`` attribute, see
            :ref:`mobile-workspace`, defaults to None
        :param polygons: obstacle polygons, defaults to None (no polygons)

        The obstacle polygons are specified as instances of
        :class:`~spatialmath.geom2d.Polygon2`.  Use :meth:`add` to add a polygon
        given as an ndarray(2,N).

        The workspace can be specified in several ways:

        ==============  =======  =======
        ``workspace``   x-range  y-range
        ==============  =======  =======
        A (scalar)      -A:A     -A:A
        [A, B]           A:B      A:B
        [A, B, C, D]     A:B      C:D
        ==============  =======  =======

        Workspace is used only to set plot bounds.
        """
        super().__init__(workspace=workspace)

        # a copy, not the caller's list or a list shared between maps
        self.polygons: list[Polygon2] = list(polygons) if polygons is not None else []

    def add(self, polygon: Polygon2 | NDArray) -> None:
        """
        Add a polygon to map

        :param polygon: a polygon, or its vertices
        :type polygon: :class:`~spatialmath.geom2d.Polygon2` or ndarray(2,N)
        """

        if isinstance(polygon, Polygon2):
            self.polygons.append(polygon)
        else:
            self.polygons.append(Polygon2(polygon))

    def iscollision(self, polygon: Polygon2 | NDArray) -> bool:
        """
        Test for collision

        :param polygon: a polygon, or its vertices
        :type polygon: :class:`~spatialmath.geom2d.Polygon2` or ndarray(2,N)
        :return: collision

        The ``polygon`` is tested against polygons in the map, and returns True
        on the first collision.

        :seealso: :meth:`add` :class:`~spatialmath.geom2d.Polygon2`
        """
        if not isinstance(polygon, Polygon2):
            polygon = Polygon2(polygon)
        return polygon.intersects(self.polygons)

    def plot(self, block: bool | None = None) -> None:
        """
        Plot the polygonal obstacles

        :param block: block until plot is dismissed, defaults to None

        The obstacles are plotted in red, in axes that have the limits of the
        workspace.
        """
        smb.plotvol2(self.workspace)

        for polygon in self.polygons:
            polygon.plot(color="r")

        if block is not None:
            plt.show(block=block)

    def isoccupied(self, p: ArrayLike2) -> bool:
        """
        Test if point lies inside an obstacle

        :param p: a 2D point
        :return: True if the point lies inside any obstacle

        The point is tested for enclosure by polygons in the map, and returns True
        on the first enclosure.
        """
        for polygon in self.polygons:
            if polygon.contains(p):
                return True

        return False

    @property
    def workspace(self) -> NDArray | None:
        """
        Bounds of the map

        :return: workspace bounds [xmin, xmax, ymin, ymax], None if not set
        :rtype: ndarray(4) or None

        Returns the bounds of the workspace given to the constructor, which is
        used only to set the limits of plots.
        """
        return self._workspace
