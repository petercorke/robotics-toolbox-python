import numpy as np
import matplotlib.pyplot as plt
from spatialmath import base
import scipy.ndimage as sp
from abc import ABC
from spatialmath import base

from spatialmath.geom2d import Polygon2

class BaseMap(ABC):
    def __init__(self, workspace=None, name=None, **unused):
        if workspace is not None:
            workspace = base.expand_dims(workspace)
            self._workspace = workspace
            self.dx = workspace[1] - workspace[0]
            self.dy = workspace[3] - workspace[2]
        self._name = name

class BaseOccupancyGrid(BaseMap):

    def __init__(self, grid=None, origin=(0, 0), 
            value=0, cellsize=1, **kwargs):
        """
        Create an occupancy grid instance

        :param grid: occupancy grid as a NumPy array
        :type grid: ndarray(N,M)
        :param size: cell size, defaults to 1
        :type size: float, optional
        :param origin: world coordinates of the grid element [0,0], defaults to (0, 0)
        :type origin: array_like(2), optional

        The array is kept internally as a bool array, True if occupied,
        corresponding to input values > 0.

        This object supports a user-defined coordinate system and grid size.
        World coordinates are converted to grid coordinates to lookup the 
        occupancy status.
        """
        super().__init__(**kwargs)

        if grid is not None:
            self._grid = grid
            self._origin = base.getvector(origin, 2)

        elif self._workspace is not None:
            self._grid = np.full(np.floor(np.r_[self.dx, self.dy] / cellsize).astype(int) + 1, value)
            self._origin = np.r_[self._workspace[0], self._workspace[2]]

        self._cellsize = cellsize
        

    def copy(self):
        """
        Copy an occupancy grid

        :return: copy of the ocupancy grid
        :rtype: OccGrid
        """
        return self.__class__(self._grid.copy(), cellsize=self._cellsize, origin=self._origin, name=self._name)

    def __str__(self):
        s = self.__class__.__name__
        if self._name is not None:
            s += f"[{self._name}]"
        s += f": {self._grid.shape[1]} x {self._grid.shape[0]}"
        s += f", cell size={self._cellsize}"
        s += f", x = [{self.xmin}, {self.xmax}], y = [{self.ymin}, {self.ymax}]"
        return s

    @property
    def grid(self):
        """
        Get the occupancy grid array

        :return: binary occupancy grid
        :rtype: ndarray(N,M) of bool

        If :meth:`inflate` has been called, this will return the inflated
        occupancy grid.
        """
        return self._grid

    @property
    def xmin(self):
        """
        Minimum x-coordinate of this grid

        :return: minimum world x-coordinate
        :rtype: float
        """
        return self._origin[0]

    @property
    def xmax(self):
        """
        Maximum x-coordinate of this grid

        :return: maximum world x-coordinate
        :rtype: float
        """
        return (self._grid.shape[1] -1 ) * self._cellsize + self._origin[0]

    @property
    def ymin(self):
        """
        Minimum y-coordinate of this grid

        :return: minimum world y-coordinate
        :rtype: float
        """
        return self._origin[1]

    @property
    def ymax(self):
        """
        Maximum y-coordinate of this grid

        :return: maximum world y-coordinate
        :rtype: float
        """
        return (self._grid.shape[0] -1 ) * self._cellsize + self._origin[1]

    @property
    def shape(self):
        """
        Shape of the occupancy grid array

        :return: shape of the occupancy grid array
        :rtype: 2-tuple
        """
        return self._grid.shape


    @property
    def maxdim(self):
        """
        Maximum dimension of grid in world coordinates

        :return: maximum side length of the occupancy grid
        :rtype: float
        """
        return max(self.grid.shape) * self._cellsize

    @property
    def workspace(self):
        """
        Bounds of the occupancy grid

        :return: workspace bounds [xmin, xmax, ymin, ymax]
        :rtype: ndarray(4)

        Returns the bounds of the occupancy grid.
        """
        return np.r_[self.xmin, self.xmax, self.ymin, self.ymax]

    @property
    def name(self):
        """
        Occupancy grid name

        :return: name of the occupancy grid
        :rtype: str
        """
        return self._name

    @name.setter
    def name(self, name):
        """
        Set occupancy grid name

        :param name: new name of the occupancy grid
        :type name: str
        """
        self._name = name

    def set(self, region, value):
        bl = self.w2g([region[0], region[2]])
        tr = self.w2g([region[1], region[3]])
        self.grid[bl[1]:tr[1]+1,bl[0]:tr[0]+1] = value

    def g2w(self, p):
        """
        Convert grid coordinate to world coordinate

        :param p: grid coordinate (column, row)
        :type p: array_like(2)
        :return: world coordinate (x, y)
        :rtype: ndarray(2)

        The grid cell size and offset are used to convert occupancy grid
        coordinate ``p`` to a world coordinate.
        """
        p = base.getvector(p, 2)
        return p * self._cellsize + self._origin

    def w2g(self, p):
        """
        Convert world coordinate to grid coordinate

        :param p: world coordinate (x, y)
        :type p: array_like(2)
        :return: grid coordinate (column, row)
        :rtype: ndarray(2)

        The grid cell size and offset are used to convert ``p`` to an occupancy
        grid coordinate.  The grid coordinate is rounded and cast to integer
        value. No check is made on the validity of the coordinate.
        """
        return (np.round((p - self._origin) / self._cellsize)).astype(int)


    def plot(self, map=None, ax=None, block=False, **kwargs):
        """
        Plot the occupancy grid

        :param map: array which is plotted instead of the grid, must be same
            size as the occupancy grid,defaults to None
        :type map: ndarray(N,M), optional
        :param ax: matplotlib axes to plot into, defaults to None
        :type ax: Axes2D, optional
        :param block: block until plot is dismissed, defaults to False
        :type block: bool, optional
        :param kwargs: arguments passed to ``imshow``

        The grid is plotted as an image but with axes in world coordinates.

        The grid is a NumPy boolean array which has values 0 (false=unoccupied)
        and 1 (true=occupied).  Passing a `cmap` option to imshow can be used
        to control the displayed color of free space and obstacles.
 
        """

        ax = base.axes_logic(ax, 2)
        
        if map is None:
            map = self._grid
            kwargs['extent'] = self.workspace

        ax.imshow(map, origin='lower', interpolation=None, **kwargs) #extent=extent, 
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.show(block=block)


    def line_w(self, p1, p2):

        gp1 = self.w2g(p1)
        gp2 = self.w2g(p2)

        return line(gp1, gp2)

    def line(self, p1, p2):

        x, y = base.bresenham(p1, p2, array=self.grid)
        z =  np.ravel_multi_index(np.vstack((y, x)), self.grid.shape)
        return z

    @property
    def ravel(self):
        return self._grid.reshape(-1)

class BinaryOccupancyGrid(BaseOccupancyGrid):

    def __init__(self, grid=None, **kwargs):

        if grid is not None:
            grid = grid.astype(bool)
        super().__init__(grid=grid, **kwargs)

    def __str__(self):
        s = super().__str__()

        ncells = np.prod(self._grid.shape)
        nobs = self._grid.sum()
        s += f", {nobs/ncells*100:.1f}% occupied"
        return s

    def isoccupied(self, p):
        """
        Test if coordinate is occupied

        :param p: world coordinate (x, y)
        :type p: array_like(2)
        :return: occupancy status of corresponding grid cell
        :rtype: bool

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

    def inflate(self, radius):
        """
        Inflate obstales

        :param radius: radius of circular structuring element in world units
        :type radius: float

        A circular structuring element is created and used to dilate the
        stored occupancy grid.

        Successive calls to ``inflate`` will compound the inflation. 

        :seealso: :func:`scipy.ndimage.binary_dilation`
        """
        # Generate a circular structuring element
        r = round(radius / self._cellsize)
        Y, X = np.meshgrid(np.arange(-r, r+1), np.arange(-r, r+1))
        SE = X**2 + Y**2 <= r**2
        SE = SE.astype(int)

        # do the inflation using SciPy
        self._grid = sp.binary_dilation(self._grid, SE)
class OccupancyGrid(BaseOccupancyGrid):
    def __str__(self):
        s = super().__str__()

        g = self._grid
        s += f", dtype {g.dtype}"
        s += f", min {g.min()}, max {g.max()}, mean {g.mean()}"
        return s

class PolygonMap(BaseMap):

    def __init__(self, workspace=None, polygons=[]):
        super().__init__(workspace=workspace)

        self.polygons = polygons

    def add(self, polygon):

        if isinstance(polygon, Polygon2):
            self.polygons.append(polygon)
        else:
            self.polygons.append(Polygon2(polygon))

    def iscollision(self, polygon):
        return polygon.intersects(self.polygons)

    def plot(self, block=False):
        base.plotvol2(self.workspace)

        for polygon in self.polygons:
            polygon.plot(color='r')

        plt.show(block=block)

    def isoccupied(self, p):
        for polygon in self.polygons:
            if polygon.contains(p):
                return True

        return False

    @property
    def workspace(self):
        """
        Bounds of the occupancy grid

        :return: workspace bounds [xmin, xmax, ymin, ymax]
        :rtype: ndarray(4)

        Returns the bounds of the occupancy grid.
        """
        return self._workspace

if __name__ == "__main__":

    # g = np.zeros((100, 100))
    # g[20:30, 50:80] = 1

    # og = OccGrid(g, size=0.1, origin=(2,4),name='bob')
    # print(og)
    # print(og.xmin, og.xmax, og.ymin, og.ymax)
    # print(og.isoccupied((8.5, 6.5)))
    # print(og.isoccupied((6, 6)))
    # print(og.isoccupied((500, 500)))
    # og.plot(block=False)
    # og2 = og.copy()
    # print(og2)
    # og2.inflate(0.5)
    # plt.figure()
    # og2.plot(block=True)

    # g = np.zeros((10,10))
    # g[2:3, 4:5] = 1
    # og = BinaryOccupancyGrid(g)
    # print(og)

    # r = og.ravel
    # print(r[24])

    # og = BinaryOccupancyGrid(workspace=[2,3,4,5], cellsize=0.2)
    # print(og)

    # og = BinaryOccupancyGrid(workspace=[2,3,4,5], cellsize=0.2, value=True)
    # print(og)

    # og = OccupancyGrid(workspace=[2,3,4,5], cellsize=0.2, value=3)
    # print(og)

    # og = OccupancyGrid(workspace=[2,3,4,5], cellsize=0.2, value=3.0)
    # print(og)

    map = PolygonMap(workspace=[0, 10])
    map.add([(5, 50), (5, 6), (6, 6), (6, 50)])
    map.add([(5, 4), (5, -50), (6, -50), (6, 4)])
    map.plot()

    og = BinaryOccupancyGrid(workspace=[-5, 5, -5, 5], value=False)
    # np.set_printoptions(linewidth=300)
    # og = BinaryOccupancyGrid(workspace=[-10, 10, -10, 10], value=False)
    # print(og)
    # og.set([1,10, -10, 10], True)
    # print(og.grid)
    # print(og.isoccupied((0,0)))