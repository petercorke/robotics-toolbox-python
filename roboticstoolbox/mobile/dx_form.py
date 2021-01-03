"""
Python ReedShepp Planner
@Author: Kristian Gibson
TODO: Comments + Sphynx Docs Structured Text
TODO: Bug-fix, testing

Not ready for use yet.
"""
from numpy import disp
from scipy import integrate
from spatialmath.base.transforms2d import *
from spatialmath.base.vectors import *
from spatialmath.pose2d import SE2
from spatialmath import base
from scipy.ndimage import *
from matplotlib import cm
from roboticstoolbox.mobile.navigation import Navigation


class DXform(Navigation):
    def __init__(self, occ_grid=None, metric="euclidean", distance_map=None, **kwargs):

        super().__init__(occ_grid=occ_grid, **kwargs)
        self._metric = metric
        self._distance_map = None

    @property
    def metric(self):
        return self._metric

    @property
    def distance_map(self):
        return self._distance_map

    def __str__(self):
        s = super().__str__()
        s += f"\n  Distance metric: {self._metric}"
        if self._distance_map is not None:
            s += ", Distance map: computed "
        else:
            s += ", Distance map: empty "

        return s

    def goal_change(self, goal):
        self._distance_map = np.array([])

    def plan(self, goal=None, animate=False):
        show = None
        if animate:
            show = 0.05
        else:
            show = 0

        if goal is not None:
            self.goal = goal

        if self._goal is None:
            raise ValueError('No goal specified here or in constructor')

        self._distance_map = distancexform(self.occ_grid_nav, self._goal, self._metric)

    # Use plot from parent class

    def next(self, robot):
        if self._distance_map is None:
            Error("No distance map computed, you need to plan.")

        directions = np.array([
            # dy  dx
            [-1, -1],
            [ 0, -1],
            [ 1, -1],
            [-1,  0],
            [ 0,  0],
            [ 1,  0],
            [ 0,  1],
            [ 1,  1],
        ])

        x = robot[0]
        y = robot[1]

        min_dist = np.inf
        for d in directions:
            try:
                if self._distance_map[y + d[0], x + d[1]] < min_dist:
                    min_dir = d
                    min_dist = self.distance_map[y + d[0], x + d[1]]
            except:
                # come here if the neighbouring cell is outside the map bounds
                raise RuntimeError(f"Unexpected error finding next min dist at {d}")

        if np.isinf(min_dist):
            raise RuntimeError("no minimum found, shouldn't happen")

        x = x + min_dir[1]
        y = y + min_dir[0]

        next = np.r_[x, y]
        if all(next == self._goal):
            return None
        else:
            return next

    def plot_3d(self, p=None, ls=None):
        fig = plt.figure()
        ax = fig.gca(projection='3d')

        surf = ax.plot_surface(self._distance_map, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)

        if p is not None:
            k = sub2ind(np.shape(self._distance_map), p[:, 1], p[:, 0])
            height = self._distance_map[k]
            ax.plot(p[:, 0], p[:, 1], height)

        plt.show()

    def plot(self, **kwargs):
        super().plot(distance=self._distance_map, **kwargs)

# Sourced from: https://stackoverflow.com/questions/28995146/matlab-ind2sub-equivalent-in-python/28995315#28995315
def sub2ind(array_shape, rows, cols):
    ind = rows*array_shape[1] + cols
    ind[ind < 0] = -1
    ind[ind >= array_shape[0]*array_shape[1]] = -1
    return ind


def ind2sub(array_shape, ind):
    ind[ind < 0] = -1
    ind[ind >= array_shape[0]*array_shape[1]] = -1
    rows = (ind.astype('int') / array_shape[1])
    cols = ind % array_shape[1]
    return rows, cols


import numpy as np

def distancexform(occgrid, goal, metric='cityblock', show=False):
    """
    Distance transform for path planning

    :param occgrid: Occupancy grid
    :type occgrid: NumPy ndarray
    :param goal: Goal position (x,y)
    :type goal: 2-element array-like
    :param metric: distance metric, defaults to 'cityblock'
    :type metric: str, optional
    :return: Distance transform matrix
    :rtype: NumPy ndarray

    Returns a matrix the same size as the occupancy grid ``occgrid`` where each
    cell contains the distance to the goal according to the chosen ``metric``.

        - Obstacle cells will be set to ``nan``.  
        - Unreachable cells, ie. free cells _inside obstacles_ will be set 
          to ``inf``. 

    The cells of the passed occupancy grid are:
        - zero, cell is free or driveable
        - one, cell is an obstacle, or not driveable
    """

    # build the matrix for performing distance transform:
    # - obstacles are nan
    # - other cells are inf
    # - goal is zero

    goal = base.getvector(goal, 2, dtype=np.int)
    occgrid = occgrid.astype(np.float32)
    occgrid[occgrid > 0] = np.nan  # assign nan to obstacle cells
    nans = np.isnan(occgrid)  # keep record of where the NaNs are

    occgrid[occgrid==0] = np.inf  # assign inf to other cells
    occgrid[goal[1], goal[0]] = 0  # assign zero to goal
    
    # create the appropriate distance matrix D
    if metric.lower() in ('manhattan', 'cityblock'):
        D = np.array([
                [ np.inf,   1,   np.inf],
                [      1,   0,        1],
                [ np.inf,   1,   np.inf]
            ])
    elif metric.lower() == 'euclidean':
        r2 = np.sqrt(2)
        D = np.array([
                [ r2,   1,   r2],
                [  1,   0,    1],
                [ r2,   1,   r2]
                ])

    # get ready to iterate
    count = 0
    ninf = np.inf  # number of infinities in the map

    while True:
        
        occgrid = dxstep(occgrid, D)
        occgrid[nans] = np.nan

        count += 1
        # if opt.animate
        #     cmap = [1 0 0; gray(count)];
        #     colormap(cmap)
        #     image(occgrid+1, 'CDataMapping', 'direct');
        #     set(gca, 'Ydir', 'normal');
        #     xlabel('x');
        #     ylabel('y');
        #     if opt.animate
        #         anim.add();
        #     else
        #         pause(opt.delay);

        ninfnow = sum(np.isinf(occgrid.flatten()))  # current number of Infs
        if ninfnow == ninf:
            # stop if the number of Infs left in the map had stopped reducing
            # it may never get to zero if there are unreachable cells in the map
            break

        ninf = ninfnow

    print(f"{count:d} iterations, {ninf:d} unreachable cells")
    return occgrid

def dxstep(G, D):

    # pad with inf
    H = np.pad(G, max(D.shape) // 2, 'constant', constant_values=np.inf)
    rows, columns = G.shape
    
    # inspired by https://landscapearchaeology.org/2018/numpy-loops/
    minimum = np.full(G.shape, np.inf)
    for y in range(3):
        for x in range(3):
            v = H[y : rows + y, x : columns + x] + D[y, x]
            # we use fmin() because it ignores NaNs, behaves like MATLAB min()
            minimum = np.fmin(minimum, v)

    return minimum


if __name__ == "__main__":


    # # make a simple map, as per the MOOCs
    # occgrid = np.zeros((10,10))
    # occgrid[3:6,2:7] = 1
    # occgrid[4,3:5] = 0  # hole in the obstacle
    # # occgrid[7:8,7] = 1  # extra bit

    # print(occgrid)
    # print()

    # goal=[5,7]
    # np.set_printoptions(precision=1)
    # dx = distancexform(occgrid, goal, metric='Euclidean')
    # print(dx)


    from roboticstoolbox import DXform
    from scipy.io import loadmat

    vars = loadmat("/Users/corkep/code/robotics-toolbox-python/data/house.mat", squeeze_me=True, struct_as_record=False)
    house = vars['house']
    place = vars['place']

    dx = DXform(house)
    print(dx)
    dx.goal = [1,2]
    dx.plan(place.kitchen)
    dx.plot()