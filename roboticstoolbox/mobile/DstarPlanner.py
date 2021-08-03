"""

D* grid planning

author: Nirnay Roy

See Wikipedia article (https://en.wikipedia.org/wiki/D*)

"""
import math

from sys import maxsize
from collections import namedtuple
from enum import IntEnum, auto
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from roboticstoolbox.mobile.Planner import Planner
from roboticstoolbox.mobile.OccGrid import OccupancyGrid


show_animation = True

class Tag(IntEnum):
    NEW = auto()
    OPEN = auto()
    CLOSED = auto()
class State:

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None # 'back pointer' to next state
        self.t = Tag.NEW  # open closed new
        self.h = 0  # cost to goal
        self.k = 0  # estimate of shortest path cost

    def __str__(self):
        return f"({self.x}, {self.y})" #[{self.h:.1f}, {self.k:.1f}]"

    def __repr__(self):
        return self.__str__()

class Map:

    def __init__(self, row, col):
        self.row = row
        self.col = col
        self.map = self.init_map()

    def init_map(self):
        map_list = []
        for i in range(self.row):
            tmp = []
            for j in range(self.col):
                tmp.append(State(i, j))
            map_list.append(tmp)
        return map_list

    _neighbours = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    def get_neighbors(self, state):
        state_list = []
        # for i in [-1, 0, 1]:
        #     for j in [-1, 0, 1]:
                # if i == 0 and j == 0:
                #     continue
                # if state.x + i < 0 or state.x + i >= self.row:
                #     continue
                # if state.y + j < 0 or state.y + j >= self.col:
                #     continue
                # state_list.append(self.map[state.x + i][state.y + j])
        for i, j in self._neighbours:
            try:
                state_list.append(self.map[state.x + i][state.y + j])
            except IndexError:
                pass
        return state_list


    def set_cost(self, region, cost, modify=None):

        xmin = max(0, region[0])
        xmax = min(self.col, region[1])
        ymin = max(0, region[2])
        ymax = min(self.row, region[3])

        self.costmap[ymin:ymax, xmin:xmax] = cost

        if modify is not None:
            for x in range(xmin, xmax):
                for y in range(ymin, ymax):
                    modify.modify_cost(self.map[x][y])

    _root2 = np.sqrt(2)

    def cost(self, state1, state2):
        c = (self.costmap[state1.y, state1.x] + self.costmap[state2.y, state2.x]) / 2

        dx = state1.x - state2.x
        dy = state1.y - state2.y
        if dx == 0 or dy == 0:
            # NSEW movement, distance of 1
            return c
        else:
            # diagonal movement, distance of sqrt(2)
            return c * self._root2
        # return c * math.sqrt(()**2 +
        #                  ()**2)

    def show_h(self):
        h = np.empty((self.col, self.row))

        for x in range(self.col):
            for y in range(self.row):
                h[y, x] = self.map[x][y].h
        print(h)

class Dstar:
    def __init__(self, map):
        self.map = map
        self.open_list = set()
        self.nexpand = 0

    def process_state(self, verbose=False):
        if verbose:
            print('FRONTIER ', ', '.join([str(x) for x in self.open_list]))
        
        # get state from frontier
        x = self.min_state()
        if x is None:
            if verbose:
                print('  x is None ')
            return -1
        self.nexpand += 1

        # get its current cost
        k_old = self.get_kmin()
        self.remove(x)

        if verbose:
            print('EXPAND ', x, k_old)

        if x.h > k_old + 1e-6:
            # RAISE state
            if verbose:
                print('  raise')
            for y in self.map.get_neighbors(x):
                if y.t is not Tag.NEW and y.h <= k_old and x.h > y.h + self.map.cost(x, y):
                    x.parent = y
                    x.h = y.h + self.map.cost(x, y)

        if np.isclose(x.h, k_old):
            # normal state
            if verbose:
                print('  normal')
            for y in self.map.get_neighbors(x):
                if y.t is Tag.NEW \
                   or (y.parent == x and not np.isclose(y.h, x.h + self.map.cost(x, y))) \
                   or (y.parent != x and y.h > x.h + self.map.cost(x, y) + 1e-6):
                    if verbose:
                        print(f"  reparent {y} from {y.parent} to {x}")
                    y.parent = x
                    self.insert(y, x.h + self.map.cost(x, y))
        else:
            # RAISE or LOWER state
            if verbose:
                print('  lower')
            for y in self.map.get_neighbors(x):
                if y.t is Tag.NEW or (y.parent == x and not np.isclose(y.h, x.h + self.map.cost(x, y))):
                    y.parent = x
                    self.insert(y, x.h + self.map.cost(x, y))
                else:
                    if y.parent != x and (y.h > x.h + self.map.cost(x, y)) and x.t is Tag.CLOSED:
                        self.insert(x, x.h)
                    else:
                        if y.parent != x and x.h > y.h + self.map.cost(y, x) + 1e-6 \
                                and y.t is Tag.CLOSED and y.h > k_old + 1e-6:
                            self.insert(y, y.h)
        if verbose:
            print()
        return self.get_kmin()

    def min_state(self):
        if not self.open_list:
            return None
        min_state = min(self.open_list, key=lambda x: x.k)
        return min_state

    def get_kmin(self):
        if not self.open_list:
            return -1
        k_min = min([x.k for x in self.open_list])
        return k_min

    ninsert = 0
    nin = 0

    def insert(self, state, h_new):
        self.ninsert += 1
        if state in self.open_list:
            self.nin += 1
            if state.t is not Tag.OPEN:
                print('already in but not open')
        if state.t is Tag.NEW:
            state.k = h_new
        elif state.t is Tag.OPEN:
            state.k = min(state.k, h_new)
        elif state.t is Tag.CLOSED:
            state.k = min(state.h, h_new)
        state.h = h_new
        state.t = Tag.OPEN

        self.open_list.add(state)

    def remove(self, state):
        if state.t is Tag.OPEN:
            state.t = Tag.CLOSED
        else:
            state.t = Tag.CLOSED
            print('removing non open state')
        self.open_list.remove(state)

    def modify_cost(self, x, newcost):
        self.map.costmap[x.y, x.x] = newcost
        if x.t is Tag.CLOSED:
            self.insert(x, x.parent.h + self.map.cost(x, x.parent))
        return self.get_kmin()

    def modify(self, state):
        self.modify_cost(state)
        while True:
            k_min = self.process_state()
            if k_min == -1 or k_min >= state.h:
                break

class DstarPlanner(Planner):
    r"""
    D* path planner

    :param occgrid: occupancy grid
    :type curvature: OccGrid or ndarray(w,h)
    :param Planner: D* path planner
    :type Planner: DstarPlanner instance

    ==================   ========================
    Feature              Capability
    ==================   ========================
    Plan                 Cartesian space
    Obstacle avoidance   Yes
    Curvature            Discontinuous
    Motion               Forwards only
    ==================   ========================

    Also known as wavefront, grassfire or brushfire planning algorithm.

    Creates a planner that finds the path between two points in the
    plane using forward motion.  The path comprises a set of points in 
    adjacent cells.

    :author: Peter Corke_
    :seealso: :class:`Planner`
    """
    def __init__(self, occ_grid=None, 
                 reset=False, **kwargs):
        super().__init__(ndims=2, **kwargs)

        self.costmap = np.where(self.occgrid.grid > 0, np.inf, 1)
        self.map = Map(self.costmap.shape[1], self.costmap.shape[0])
        self.map.costmap = self.costmap
        self.dstar = Dstar(self.map)


    def plan(self, goal, animate=False, progress=True):
        r"""
        Plan D* path

        :param goal: goal position :math:`(x, y)` or configuration :math:`(x, y, \theta)`, defaults to None
        :type goal: array_like(2) or array_like(3), optional
        :param animate: animate the planning algorithm iterations, defaults to False
        :type animate: bool, optional
        :param progress: show progress bar, defaults to True
        :type progress: bool, optional

        The implementation depends on the particular planner.  Some may have
        no planning phase.  The plan may also depend on just the start or goal.
        """
        self.goal = goal
        goalstate = self.map.map[goal[0]][goal[1]]
        self.goalstate = goalstate

        # self.dstar.open_list.add(goalstate)
        self.dstar.insert(goalstate, 0)

        while True:
            ret = self.dstar.process_state()
            # print('plan', ret, len(self.dstar.open_list))

            if ret == -1:
                break
        
        print(self.dstar.ninsert, self.dstar.nin)


    def query(self, start, update=None, changes=None):
        self.start = start
        startstate = self.map.map[start[0]][start[1]]
        s = startstate
        s = s.parent
        tmp = startstate

        cost = tmp.h

        for x in range(self.map.col):
            for y in range(self.map.row):
                self.dstar.map.map[x][y].t == 'new'
        self.goalstate.h = 0

        path = []
        while True:
            path.append((tmp.x, tmp.y))
            if tmp == self.goalstate:
                break
            
            # x, y = tmp.parent.x, tmp.parent.y
            if update is not None and update == tmp:
                # make changes now
                for x, y, newcost in changes:
                    X = self.dstar.map.map[x][y]
                    print('difference at ', x, y)
                    val = self.dstar.modify_cost(X, newcost)
                    while val != -1 and val < tmp.h:
                        val = self.dstar.process_state()
            tmp = tmp.parent
            print('done')

        status = namedtuple('DstarStatus', ['cost',])
        
        return np.array(path).T, status(cost)

    # # Just feed self._h into plot function from parent as p

    # def next(self, current):
    #     if not self._valid_plan:
    #         Error("Cost map has changed, replan")
    #     # x = sub2ind(np.shape(self._costmap), current[1], current[0])
    #     # x = self._b[x]
    #     i = np.ravel_multi_index([current[1], current[0]], self._costmap.shape)
    #     i = self._b[i]

    #     if i == 0:
    #         return None  # we have arrived
    #     else:
    #         x = np.unravel_index((i), self._costmap.shape)
    #         return x[1], x[0]

if __name__ == "__main__":

    og = np.zeros((10,10))
    og[4:8, 3:6] = 1
    print(og)

    ds = DstarPlanner(occgrid=og)
    print(ds.costmap)

    start = (1,1)
    goal = (7,6)

    ds.plan(goal=goal)
    ds.map.show_h()

    # path, status = ds.query(start=start)
    # print(path)
    # print(status)

    # ds.plot(path=path)


    changes = []
    for x in range(3, 6):
        for y in range(0, 4):
            changes.append((x, y, 5))
    # print(costmap2)

    path2, status2 = ds.query(start=start, update=ds.dstar.map.map[3][3], changes=changes)
    print(ds.map.costmap)
    ds.map.show_h()

    # ds.dstar.replan()

    # path2, status = ds.query(start=start)
    print(path2)
    print(status2)

    ds.plot(path=path2)
    # ds.plot(path=path2, block=True)

    # obstacle zone
    # m.set_cost([30, 60, 20, 60], np.inf)

    # start = [10, 10]
    # goal = [70, 70]
    # if show_animation:
    #     m.plot()
    #     plt.plot(start[0], start[1], "og")
    #     plt.plot(goal[0], goal[1], "xb")
    #     plt.axis("equal")

    # start = m.map[start[0]][start[1]]
    # end = m.map[goal[0]][goal[1]]
    # dstar = Dstar(m)
    # rx, ry = dstar.run(start, end)

    # if show_animation:
    #     plt.plot(rx, ry, "-r")
    #     plt.show(block=True)

    # # costly zone
    # m.set_cost([30, 40, 60, 80], 1.70, modify=dstar)

    # dstar.replan()

    plt.show(block=True)

