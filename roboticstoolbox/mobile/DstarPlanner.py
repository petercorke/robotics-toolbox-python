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
from roboticstoolbox.mobile.PlannerBase import PlannerBase
from roboticstoolbox.mobile.OccGrid import OccupancyGrid
import heapq
import bisect
import math

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

    def __lt__(self, other):
        return True

class Map:

    def __init__(self, row, col):
        self.row = row
        self.col = col
        self.map = self.init_map()

    def init_map(self):
        map_list = []  # list of rows
        for i in range(self.row):
            # for each row, make a list
            tmp = []
            for j in range(self.col):
                tmp.append(State(j, i))  # append column to the row
            map_list.append(tmp) # append row to map
        return map_list

    _neighbours = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    def get_neighbors(self, state):
        state_list = []

        for i, j in self._neighbours:
            try:
                if state.x + i >= 0 and state.y + j >= 0:
                    state_list.append(self.map[state.y + j][state.x + i])
            except IndexError:
                pass
        return state_list

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
                h[y, x] = self.map[y][x].h
        print(h)

# for performance reasons there are some important differences compared to
# the version from Python Robotics
#
# replace the classic D* functions min_state(), get_kmin(), insert(), remove()
# with heapq.heappush() and heapq.heappop(). The open_list is now a list of 
# tuples (k, state) maintained by heapq, rather than a set
#
# lots of unnecessary inserting/deleting from the open_list due to arithmetic
# rounding in tests for costs h and k:
#  - replace equality tests with math.isclose() which is faster than np.isclose()
#  - add an offset to inequality tests, X > Y becomes X > Y + tol
class Dstar:
    def __init__(self, map, tol=1e-6):
        self.map = map
        # self.open_list = set()
        self.open_list = []
        self.nexpand = 0
        self.tol = tol

    def process_state(self, verbose=False):
        if verbose:
            print('FRONTIER ', ' '.join([f"({x[1].x}, {x[1].y})" for x in self.open_list]))

        # get state from frontier
        if len(self.open_list) == 0:
            if verbose:
                print('  x is None ')
            return -1
        k_old, x = heapq.heappop(self.open_list)
        x.t = Tag.CLOSED

        self.nexpand += 1

        if verbose:
            print(f"EXPAND {x}, {k_old:.1f}")

        if x.h > k_old + self.tol:
            # RAISE state
            if verbose:
                print('  raise')
            for y in self.map.get_neighbors(x):
                if (y.t is not Tag.NEW and
                        y.h <= k_old - self.tol and 
                        x.h > y.h + self.map.cost(x, y) + self.tol):
                    if verbose:
                        print(f"  {x} cost from {x.h:.1f} to {y.h + self.map.cost(x, y):.1f}; parent from {x.parent} to {y}")
                    x.parent = y
                    x.h = y.h + self.map.cost(x, y)

        if math.isclose(x.h, k_old, rel_tol=0, abs_tol=self.tol):
            # normal state
            if verbose:
                print('  normal')
            for y in self.map.get_neighbors(x):
                if y.t is Tag.NEW \
                   or (y.parent == x and not math.isclose(y.h, x.h + self.map.cost(x, y), rel_tol=0, abs_tol=self.tol)) \
                   or (y.parent != x and y.h > x.h + self.map.cost(x, y) + self.tol):
                    if verbose:
                        print(f"  reparent {y} from {y.parent} to {x}")
                    y.parent = x
                    self.insert(y, x.h + self.map.cost(x, y))
        else:
            # RAISE or LOWER state
            if verbose:
                print('  raise/lower')
            for y in self.map.get_neighbors(x):
                if y.t is Tag.NEW or (y.parent == x and not math.isclose(y.h, x.h + self.map.cost(x, y), rel_tol=0, abs_tol=self.tol)):
                    if verbose:
                        print(f"  {y} cost from {y.h:.1f} to {y.h + self.map.cost(x, y):.1f}; parent from {y.parent} to {x}; add to frontier")
                    y.parent = x
                    self.insert(y, x.h + self.map.cost(x, y))
                else:
                    if y.parent != x and y.h > x.h + self.map.cost(x, y) + self.tol and x.t is Tag.CLOSED:
                        self.insert(x, x.h)
                        if verbose:
                            print(f"  {x}, {x.h:.1f} add to frontier")
                    else:
                        if y.parent != x and x.h > y.h + self.map.cost(y, x) + self.tol \
                                and y.t is Tag.CLOSED and y.h > k_old + self.tol:
                            self.insert(y, y.h)
                        if verbose:
                            print(f"  {y}, {y.h:.1f} add to frontier")
        if verbose:
            print()

        if len(self.open_list) == 0:
            return -1
        else:
            return self.open_list[0][0]

    ninsert = 0
    nin = 0

    def insert(self, state, h_new):
        self.ninsert += 1

        if state.t is Tag.NEW:
            state.k = h_new

        elif state.t is Tag.OPEN:
            k_new = min(state.k, h_new)
            if state.k == k_new:
                # k hasn't changed, and vertex already in frontier
                # just update h and be done
                state.h = h_new
                return
            else:
                # k has changed, we need to remove the vertex from the list
                # and re-insert it
                state.k = k_new

                # scan the list to find vertex, then remove it
                # this is quite slow
                for i, item in enumerate(self.open_list):
                    if item[1] is state:
                        del self.open_list[i]
                        break

            # state.k = min(state.k, h_new)
            # # remove the item from the open list
            # # print('node already in open list, remove it first')
            # #TODO use bisect on old state.k to find the entry
            # for i, item in enumerate(self.open_list):
            #     if item[1] is state:
            #         del self.open_list[i]
            #         break  

        elif state.t is Tag.CLOSED:
            state.k = min(state.h, h_new)

        state.h = h_new
        state.t = Tag.OPEN

        # self.open_list.add(state)
        heapq.heappush(self.open_list, (state.k, state))

    def modify_cost(self, x, newcost):
        self.map.costmap[x.y, x.x] = newcost
        if x.t is Tag.CLOSED:
            self.insert(x, x.parent.h + self.map.cost(x, x.parent))
        if len(self.open_list) == 0:
            return -1
        else:
            # lowest priority item is always at index 0 according to docs
            return self.open_list[0][0]

    def showparents(self):
        for y in range(self.map.row-1, -1, -1):
            if y == self.map.row-1:
                print("   ", end='')
                for x in range(self.map.col):
                    print(f"  {x}   ", end='')
                print()
            print(f"{y}: ", end='')
            for x in range(self.map.col):
                x = self.map.map[y][x]
                par = x.parent
                if par is None:
                    print('  G   ', end='')
                else:
                    print(f"({par.x},{par.y}) ", end='')
            print()
        print()
class DstarPlanner(PlannerBase):
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
        self.map = Map(self.costmap.shape[0], self.costmap.shape[1])
        self.map.costmap = self.costmap
        self.dstar = Dstar(self.map) #, tol=0)


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
        goalstate = self.map.map[goal[1]][goal[0]]
        self.goalstate = goalstate

        # self.dstar.open_list.add(goalstate)
        self.dstar.insert(goalstate, 0)

        while True:
            ret = self.dstar.process_state()
            # print('plan', ret, len(self.dstar.open_list))

            if ret == -1:
                break
        
        print(self.dstar.ninsert, self.dstar.nin)


    @property
    def nexpand(self):
        return self.dstar.nexpand

    def query(self, start, sensor=None, animate=False, verbose=False):
        self.start = start
        startstate = self.map.map[start[1]][start[0]]
        s = startstate
        s = s.parent
        tmp = startstate

        cost = tmp.h
        self.goalstate.h = 0

        path = []
        while True:
            path.append((tmp.x, tmp.y))
            if tmp == self.goalstate:
                break
            
            # x, y = tmp.parent.x, tmp.parent.y

            if sensor is not None:
                changes = sensor((tmp.x, tmp.y))
                if changes is not None:
                    # make changes to the plan
                    for x, y, newcost in changes:
                        X = self.dstar.map.map[y][x]
                        # print(f"change cost at ({x}, {y}) to {newcost}")
                        val = self.dstar.modify_cost(X, newcost)
                    # propagate the changes to plan
                    print('propagate')
                    # self.dstar.showparents()
                    while val != -1 and val < tmp.h:
                        val = self.dstar.process_state(verbose=verbose)
                        # print('open set', len(self.dstar.open_list))
                    # self.dstar.showparents()
                    
            tmp = tmp.parent
            # print('move to ', tmp)

        status = namedtuple('DstarStatus', ['cost',])
        
        return np.array(path), status(cost)

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


    def sensorfunc(pos):
        if pos == (3, 3):
            changes = []
            for x in range(3, 6):
                for y in range(0, 4):
                    changes.append((x, y, 100))
            return changes

    path2, status2 = ds.query(start=start, sensor=sensorfunc, verbose=False)
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

