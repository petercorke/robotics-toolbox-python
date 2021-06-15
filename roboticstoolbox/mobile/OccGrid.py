import numpy as np
import matplotlib.pyplot as plt
from spatialmath import base
import scipy.ndimage as sp

class OccGrid:

    def __init__(self, grid, size=1, origin=(0, 0), name=None):
        """
        Create an occupancy grid instance

        :param grid: occupancy grid as a NumPy array
        :type grid: ndarray(N,M)
        :param size: cell size, defaults to 1
        :type size: float, optional
        :param origin: world coordinates of the grid element [0,0], defaults to (0, 0)
        :type origin: array_like(2), optional

        The array is kept internally as a bool array, True if occupied, corresponding
        to input values > 0.
        """
        self._size = size
        self._origin = base.getvector(origin, 2)
        self._grid = grid.astype(bool)
        self._name = name

    def copy(self):
        return self.__class__(self._grid.copy(), self._size, self._origin, self._name)

    def __str__(self):
        s = "OccupancyGrid"
        if self._name is not None:
            s += f"[{self._name}]"
        s += f": {self._grid.shape[1]} x {self._grid.shape[0]}"
        ncells = np.prod(self._grid.shape)
        nobs = self._grid.sum()
        s += f", {nobs/ncells*100:.1f}% occupied"
        s += f", cell size={self._size}"
        s += f", x: {self.xmin} - {self.xmax}, y: {self.ymin} - {self.ymax}"
        return s

    @property
    def grid(self):
        """
        Get the occupancy grid array

        :return: binary occupancy grid
        :rtype: ndarray(N,M) of bool
        """
        return self._grid

    @property
    def xmin(self):
        """
        Minimum x-coordinate of this grid

        :return: minimum x-coordinate
        :rtype: float
        """
        return self._origin[0]

    @property
    def xmax(self):
        """
        Maximum x-coordinate of this grid

        :return: maximum x-coordinate
        :rtype: float
        """
        return (self._grid.shape[1] -1 ) * self._size + self._origin[0]

    @property
    def ymin(self):
        """
        Minimum y-coordinate of this grid

        :return: minimum y-coordinate
        :rtype: float
        """
        return self._origin[1]

    @property
    def ymax(self):
        """
        Maximum y-coordinate of this grid

        :return: maximum y-coordinate
        :rtype: float
        """
        return (self._grid.shape[0] -1 ) * self._size + self._origin[1]

    @property
    def shape(self):
        """
        Shape of the occupancy grid array

        :return: shape of the occupancy grid array
        :rtype: 2-tuple
        """
        return self._grid.shape

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

    def g2w(self, p):
        """
        Convert grid coordinate to world coordinate

        :param p: grid coordinate (column, row)
        :type p: array_like(2)
        :return: world coordinate (x, y)
        :rtype: ndarray(2)
        """
        p = base.getvector(p, 2)
        return p * self._size + self._origin

    def w2g(self, p):
        """
        Convert world coordinate to grid coordinate

        :param p: world coordinate (x, y)
        :type p: array_like(2)
        :return: grid coordinate (column, row)
        :rtype: ndarray(2)

        The grid coordinate is rounded and cast to integer value.
        """
        return (np.round(p - self._origin) / self._size).astype(int)

    def isoccupied(self, p):
        """
        Test if coordinate is occupied

        :param p: world coordinate (x, y)
        :type p: array_like(2)
        :return: occupancy status of corresponding grid cell
        :rtype: bool

        The world coordinate is rounded to integer grid coordinates and the
        occupancy status of that cell is returned.

        :seealso: :meth:`w2g`
        """
        c, r = self.w2g(p)
        try:
            return self._grid[r, c]
        except IndexError:
            return True

    def plot(self, map=None, ax=None, block=False, **kwargs):

        ax = base.axes_logic(ax, 2)
        extent = [self.xmin, self.xmax, self.ymin, self.ymax]
        if map is None:
            ax.imshow(self._grid, extent=extent, origin='lower', **kwargs)
        else:
            ax.imshow(map, extent=extent, origin='lower', **kwargs)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        plt.show(block=block)

    def inflate(self, radius):
        """
        Inflate obstales

        :param radius: radius of circular structuring element in world units
        :type radius: float

        A circular structuring element is created and used to dilate the
        occupancy grid.

        :seealso: :func:`scipy.ndimage.binary_dilation`
        """
        # Generate a circular structuring element
        r = round(radius / self._size)
        Y, X = np.meshgrid(np.arange(-r, r+1), np.arange(-r, r+1))
        SE = X**2 + Y**2 <= r**2
        SE = SE.astype(int)

        # do the inflation using SciPy
        self._grid = sp.binary_dilation(self._grid, SE)

if __name__ == "__main__":

    g = np.zeros((100, 100))
    g[20:30, 50:80] = 1

    og = OccGrid(g, size=0.1, origin=(2,4),name='bob')
    print(og)
    print(og.xmin, og.xmax, og.ymin, og.ymax)
    print(og.isoccupied((8.5, 6.5)))
    print(og.isoccupied((6, 6)))
    print(og.isoccupied((500, 500)))
    og.plot(block=False)
    og2 = og.copy()
    print(og2)
    og2.inflate(0.5)
    plt.figure()
    og2.plot(block=True)