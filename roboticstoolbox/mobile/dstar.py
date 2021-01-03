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
from scipy.ndimage import *
from matplotlib import cm
from roboticstoolbox.mobile.navigation import Navigation

class DStar(Navigation):
    def __init__(self, occ_grid=None, 
                 reset=False, **kwargs):
        self._g = None
        self._b = None
        self._t = None
        self._h = None
        self._valid_plan = None
        self._open_list = None
        self._open_list_maxlen = None
        self._new = 0
        self._open = 1
        self._closed = 2
        self._niter = None
        self._costmap = occ_grid.astype(np.float32)
        self._m = 'euclidean'

        super().__init__(occ_grid=occ_grid, **kwargs)

    @property
    def niter(self):
        return self._niter

    @property
    def costmap(self):
        return self._costmap

    def dstar(self, map, goal=None, metric=None, inflate=None, progress=None):
        self._occ_grid = map
        self._goal = goal

        self.reset()
        if self._goal is not None or self._goal is not np.array([]):
            self.goal_change()

        self.reset()

        return self

    def reset(self):
        self._b = np.zeros(self._costmap.shape)
        self._t = np.zeros(self._costmap.shape)
        self._h = np.inf * np.ones(self._costmap.shape)
        
        self._open_list = np.zeros(2, 0)

        self._open_list_maxlen = -np.inf

        self.occ_grid2costmap(self._occ_grid_nav)

        self._valid_plan = False

    def __str__(self):
        s = super().__str__()
        s += f"\n  Distance metric: {self._metric}"
        if self._distance_map is not None:
            s += ", Cost map: computed "
        else:
            s += ", Cost map: empty "
        if self._valid_plan:
            s += "  \nPlan: valid"
        else:
            s += "  \nPlan: stale"
        return s


    # Just feed self._h into plot function from parent as p

    def next(self, current):
        if not self._valid_plan:
            Error("Cost map has changed, replan")
        x = sub2ind(np.shape(self._costmap), current[1], current[0])
        x = self._b[x]

        n = None
        if x == 0:
            n = np.array([])
        else:
            r, c = ind2sub(np.shape(self._costmap), current[1], current[0])
            n = np.array([c, r])

        return n

    def plan(self, goal=None, animate=False, progress=True):
        if goal is not None:
            self.goal = goal
            self.reset()

        assert(goal is not None and goal is not np.array([]))

        goal = self._goal

        self._g = sub2ind(np.shape(self._occ_grid_nav), goal[1], goal[0])
        self.__insert(self._g, 0, 'goalset')
        self._h[self._g] = 0

        self._niter = 0
        h_prog = None
        if progress:
            h_prog = self.progress_init('D* Planning')

        n_free = np.prod(np.shape(self._occ_grid_nav)) - np.sum(np.sum(self._occ_grid_nav > 0))
        n_update = np.round(n_free / 100)

        while True:
            self._niter = self._niter + 1
            if progress and np.mod(self._niter, n_update) == 0:
                self.progress(h_prog, self._niter/n_free)

                if animate:
                    self.show_distance(self._h)

            if self.process_state() < 0:
                break

            if progress:
                self.progress_delete(h_prog)
            self._valid_plan = True
            disp(self._niter + " iterations\n")

    def set_cost(self, cost_map):
        if not np.all(np.shape(cost_map) == np.shape(self._occ_grid_nav)):
            Error("Costmap must be the same size of the occupancy grid")

        self._costmap = cost_map
        self._valid_plan = False

    def modify_cost(self, xy, new_cost):
        if np.all(np.shape[xy] == np.array([2, 2])) and np.size(new_cost) == 1:
            for xx in range(xy[0, 0], xy[0, 1]):
                for yy in range(xy[1, 0], xy[1, 1]):
                    self.modify(xx, yy, new_cost)
        elif len(xy[0]) == np.size(new_cost):
            for i in range(0, len[xy[0]]):
                self.modify(xy[0, i], xy[1, i], new_cost[i])
        else:
            Error("Number of columns of P and C must match")

        self._valid_plan = False

    def modify(self, x, y, new_cost):
        x = sub2ind(np.shape(self._costmap), y, x)
        self._costmap[x] = new_cost

        if self._t[x] == self._closed:
            self.__insert(x, self._h[x], 'modifycost')

    # These are private methods ... kinda
    def __occ_grid2cost_map(self, og=None, cost=None):
        if cost is None:
            cost = 1
        self._costmap = og
        self._costmap[self._costmap==1] = np.Inf
        self._costmap[self._costmap==0] = cost

    def __process_state(self):
        x = self.__min_state()
        r = None

        if x is None or x is np.array([]):
            r = -1
            return r

        k_old = self.__get_kmin()
        self.__delete(x)

        if k_old < self.h[x]:
            self.message('k_old == h[x]: ' + k_old)
            for y in self.__neighbours(x):
                if (self._h[y] <= k_old) and (self._h[x] > self._h[y] + self.__c(y, x)):
                    self._b[x] = y
                    self._h[x] = self._h[y] + self.__c(y, x)

        if k_old == self._h[x]:
            self.message('k_old == h[x]: ' + k_old)
            for y in self.__neighbours(x):
                if (self._t[y] == self._new) or ( (self._b[y] == x) and
                    (self._h[y] != (self._h[x] + self.c[x, y]))) or  ((self._b[y] != x)
                     and (self._h[y] > (self._h[x] + self.c[x, y]))):

                    self._b[y] = x
                    self.__insert(y, self._h(x)+self._c[x, y], 'L13')
        else:
            self.message("k_old == h[x]: " + k_old)
            for y in self.__neighbours(x):
                if (self._t[y] == self.__new) or ((self._b[y] == x) and
                        (self._h[y] != (self._h[x] + self.__c(x, y)))):
                    self._b[y] = x
                    self.__insert(y, self._h[x] + self.__c(x, y), "L18")
                else:
                    if (self._b[y] != x) and (self._h[y] > (self._h[x] + self.__c(x, y))):
                        self.__insert(x, self._h[x], "L21")
                    else:
                        if (self._b[y] != x) and (self._h[x] > self._h[y] + self.__c(y, x)) and \
                                (self._t[y] == self.__closed) and self._h[y] > k_old:
                            self.__insert(y, self._h[y], 'L25')

        r = 0
        return r

    def __k(self, x):
        i = self._open_list[0,:] == x
        kk = self._open_list[1, i]
        return kk

    def __insert(self, x, h_new, where):
        self.message("Insert (" + where + ") " + x + " = " + h_new + "\n")

        i = np.argwhere(self._open_list[0,:] == x)
        if len(i) > 1:
            Error("D*:Insert: state in open list " + x + " times.\n")

        k_new = None
        if self._t[x] == self._new:
            k_new = h_new
            data = np.array([(x), (k_new)])
            self._open_list = np.array([(self._open_list), data] )
        elif self._t[x] == self._open:
            k_new = np.min(self._open_list[1, i], h_new)
        elif self._t(x) == self._closed:
            k_new = np.min(self._h[x], h_new)
            data = np.array([(x), (k_new)])
            self._open_list = np.array([self._open_list, data])

        if len(self._open_list[0]) > self._open_list_maxlen:
            self._open_list_maxlen = len(self._open_list[0])

        self._h[x] = h_new
        self._t[x] = self._open

    def __delete(self, x):
        self.message("Delete " + x)
        i = np.argwhere(self._open_list[0, :] == x)
        if len(i) != 1:
            Error("D*:Delete: state " + x + " doesn't exist.")
        if len(i) > 1:
            disp("Del")
        self._open_list = np.delete(self._open_list, i-1, 1)
        self._t = self._closed

    def __min_state(self):
        ms = None
        if self._open_list is None or self._open_list is np.array([]):
            ms = np.array([])
        else:
            d, i = np.minimum(self._open_list[1, :])
            ms = self._open_list[0, i]
        return ms

    def __get_kmin(self):
        kmin = np.minimum(self._open_list[1, :])
        return kmin

    def __c(self, x, y):
        r, c = ind2sub(np.shape(self.costmap), np.array([[x], [y]]))
        dist = np.sqrt(np.sum(np.diff(np.square(np.array([r, c])))))
        d_cost = (self._costmap[x] + self._costmap[y])/2

        cost = dist * d_cost
        return cost

    def __neighbours(self, x):
        dims = np.shape(self._costmap)
        r, c = ind2sub(dims, x)

        y = np.array([[r-1, r-1, r-1, r, r,  r+1, r+1, r+1],
                      [c-1, c, c+1, c-1, c+1, c-1, c, c+1]])

        k = (np.minimum(y) > 0) & (y[0, :] <= dims[0]) & (y[1, :] <= dims[1])
        y = y[:, k-1]
        y = np.transpose(sub2ind(dims, np.transpose(y[0, :]), np.transpose(y[1, :])))
        return y


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


if __name__ == "__main__":

    from roboticstoolbox import DStar
    from scipy.io import loadmat

    vars = loadmat("/Users/corkep/code/robotics-toolbox-python/data/map1.mat", squeeze_me=True, struct_as_record=False)
    map = vars['map']
    ds = DStar(map, verbose=True)
    ds = ds.plan([50, 35])
    path = ds.query([20, 10])
