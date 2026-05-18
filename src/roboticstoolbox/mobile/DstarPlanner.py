"""



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
from roboticstoolbox.mobile.OccGrid import BinaryOccupancyGrid, OccupancyGrid
import heapq
import bisect
import math

show_animation = True

# ======================================================================== #

# The following code is modified from Python Robotics
# https://github.com/AtsushiSakai/PythonRobotics/tree/master/PathPlanning
# D* grid planning
# Author: Nirnay Roy
# Copyright (c) 2016 - 2022 Atsushi Sakai and other contributors: https://github.com/AtsushiSakai/PythonRobotics/contributors
# Released under the MIT license: https://github.com/AtsushiSakai/PythonRobotics/blob/master/LICENSE

# for performance reasons there are some important differences compared to
# the version from Python Robotics:
#
# 1. replace the classic D* functions min__State(), get_kmin(), insert(), remove()
#    with heapq.heappush() and heapq.heappop(). The open_list is now a list of
#    tuples (k, _State) maintained by heapq, rather than a set
#
# 2. use enums rather than strings for cell state
#
# 3. lots of unnecessary inserting/deleting from the open_list due to arithmetic
#    rounding in tests for costs h and k:
#    - replace equality tests with math.isclose() which is faster than np.isclose()
#    - add an offset to inequality tests, X > Y becomes X > Y + tol


class _Tag(IntEnum):
    NEW = auto()
    OPEN = auto()
    CLOSED = auto()


class _State:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None  # 'back pointer' to next _State
        self.t = _Tag.NEW  # open closed new
        self.h = 0  # cost to goal
        self.k = 0  # estimate of shortest path cost

    def __str__(self):
        return f"({self.x}, {self.y})"  # [{self.h:.1f}, {self.k:.1f}]"

    def __repr__(self):
        return self.__str__()

    def __lt__(self, other):
        return True


class _Map:
    def __init__(self, row, col):
        self.row = row
        self.col = col
        self._Map = self.init__Map()

    def init__Map(self):
        _Map_list = []  # list of rows
        for i in range(self.row):
            # for each row, make a list
            tmp = []
            for j in range(self.col):
                tmp.append(_State(j, i))  # append column to the row
            _Map_list.append(tmp)  # append row to _Map
        return _Map_list

    _neighbours = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    def get_neighbors(self, _State):
        _State_list = []

        for i, j in self._neighbours:
            try:
                if _State.x + i >= 0 and _State.y + j >= 0:
                    _State_list.append(self._Map[_State.y + j][_State.x + i])
            except IndexError:
                pass
        return _State_list

    _root2 = np.sqrt(2)

    def cost(self, _State1, _State2):
        c = (
            self.costmap[_State1.y, _State1.x] + self.costmap[_State2.y, _State2.x]
        ) / 2

        dx = _State1.x - _State2.x
        dy = _State1.y - _State2.y
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
                h[y, x] = self._Map[y][x].h
        print(h)


class _Dstar:
    def __init__(self, _Map, tol=1e-6):
        self._Map = _Map
        # self.open_list = set()
        self.open_list = []
        self.nexpand = 0
        self.tol = tol

    def process__State(self, verbose=False):
        if verbose:
            print(
                "FRONTIER ", " ".join([f"({x[1].x}, {x[1].y})" for x in self.open_list])
            )

        # get _State from frontier
        if len(self.open_list) == 0:
            if verbose:
                print("  x is None ")
            return -1
        k_old, x = heapq.heappop(self.open_list)
        x.t = _Tag.CLOSED

        self.nexpand += 1

        if verbose:
            print(f"EXPAND {x}, {k_old:.1f}")

        if x.h > k_old + self.tol:
            # RAISE _State
            if verbose:
                print("  raise")
            for y in self._Map.get_neighbors(x):
                if (
                    y.t is not _Tag.NEW
                    and y.h <= k_old - self.tol
                    and x.h > y.h + self._Map.cost(x, y) + self.tol
                ):
                    if verbose:
                        print(
                            f"  {x} cost from {x.h:.1f} to {y.h + self._Map.cost(x, y):.1f}; parent from {x.parent} to {y}"
                        )
                    x.parent = y
                    x.h = y.h + self._Map.cost(x, y)

        if math.isclose(x.h, k_old, rel_tol=0, abs_tol=self.tol):
            # normal _State
            if verbose:
                print("  normal")
            for y in self._Map.get_neighbors(x):
                if (
                    y.t is _Tag.NEW
                    or (
                        y.parent == x
                        and not math.isclose(
                            y.h, x.h + self._Map.cost(x, y), rel_tol=0, abs_tol=self.tol
                        )
                    )
                    or (y.parent != x and y.h > x.h + self._Map.cost(x, y) + self.tol)
                ):
                    if verbose:
                        print(f"  reparent {y} from {y.parent} to {x}")
                    y.parent = x
                    self.insert(y, x.h + self._Map.cost(x, y))
        else:
            # RAISE or LOWER _State
            if verbose:
                print("  raise/lower")
            for y in self._Map.get_neighbors(x):
                if y.t is _Tag.NEW or (
                    y.parent == x
                    and not math.isclose(
                        y.h, x.h + self._Map.cost(x, y), rel_tol=0, abs_tol=self.tol
                    )
                ):
                    if verbose:
                        print(
                            f"  {y} cost from {y.h:.1f} to {y.h + self._Map.cost(x, y):.1f}; parent from {y.parent} to {x}; add to frontier"
                        )
                    y.parent = x
                    self.insert(y, x.h + self._Map.cost(x, y))
                else:
                    if (
                        y.parent != x
                        and y.h > x.h + self._Map.cost(x, y) + self.tol
                        and x.t is _Tag.CLOSED
                    ):
                        self.insert(x, x.h)
                        if verbose:
                            print(f"  {x}, {x.h:.1f} add to frontier")
                    else:
                        if (
                            y.parent != x
                            and x.h > y.h + self._Map.cost(y, x) + self.tol
                            and y.t is _Tag.CLOSED
                            and y.h > k_old + self.tol
                        ):
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

    def insert(self, _State, h_new):
        self.ninsert += 1

        if _State.t is _Tag.NEW:
            _State.k = h_new

        elif _State.t is _Tag.OPEN:
            k_new = min(_State.k, h_new)
            if _State.k == k_new:
                # k hasn't changed, and vertex already in frontier
                # just update h and be done
                _State.h = h_new
                return
            else:
                # k has changed, we need to remove the vertex from the list
                # and re-insert it
                _State.k = k_new

                # scan the list to find vertex, then remove it
                # this is quite slow
                for i, item in enumerate(self.open_list):
                    if item[1] is _State:
                        del self.open_list[i]
                        break

            # _State.k = min(_State.k, h_new)
            # # remove the item from the open list
            # # print('node already in open list, remove it first')
            # #TODO use bisect on old _State.k to find the entry
            # for i, item in enumerate(self.open_list):
            #     if item[1] is _State:
            #         del self.open_list[i]
            #         break

        elif _State.t is _Tag.CLOSED:
            _State.k = min(_State.h, h_new)

        _State.h = h_new
        _State.t = _Tag.OPEN

        # self.open_list.add(_State)
        heapq.heappush(self.open_list, (_State.k, _State))

    def modify_cost(self, x, newcost):
        self._Map.costmap[x.y, x.x] = newcost
        if x.t is _Tag.CLOSED:
            self.insert(x, x.parent.h + self._Map.cost(x, x.parent))
        if len(self.open_list) == 0:
            return -1
        else:
            # lowest priority item is always at index 0 according to docs
            return self.open_list[0][0]

    def showparents(self):
        for y in range(self._Map.row - 1, -1, -1):
            if y == self._Map.row - 1:
                print("   ", end="")
                for x in range(self._Map.col):
                    print(f"  {x}   ", end="")
                print()
            print(f"{y}: ", end="")
            for x in range(self._Map.col):
                x = self._Map._Map[y][x]
                par = x.parent
                if par is None:
                    print("  G   ", end="")
                else:
                    print(f"({par.x},{par.y}) ", end="")
            print()
        print()


# ====================== RTB wrapper ============================= #

# Copyright (c) 2022 Peter Corke: https://github.com/petercorke/robotics-toolbox-python
# Released under the MIT license: https://github.com/AtsushiSakai/PythonRobotics/blob/master/LICENSE
class DstarPlanner(PlannerBase):
    r"""
    D* path planner

    :param costmap: traversability costmap
    :type costmap: OccGrid or ndarray(w,h)
    :param kwargs: common planner options, see :class:`PlannerBase`

    ==================   ========================
    Feature              Capability
    ==================   ========================
    Plan                 :math:`\mathbb{R}^2`, discrete
    Obstacle avoidance   Yes, occupancy grid
    Curvature            Discontinuous
    Motion               Omnidirectional
    ==================   ========================

    Creates a planner that finds the minimum-cost path between two points in the
    plane using omnidirectional motion.  The path comprises a set of 8-way
    connected points in adjacent cells.

    The map is described by a 2D ``costmap`` whose elements indicate the cost
    of traversing that cell.  The cost of diagonal traverse is :math:`\sqrt{2}`
    the value of the cell.  An infinite cost indicates an untraversable cell
    or obstacle.

    Example:

    .. runblock:: pycon

        >>> from roboticstoolbox import DstarPlanner
        >>> import numpy as np
        >>> costmap = np.ones((6, 6));
        >>> costmap[2:5, 3:5] = 10
        >>> ds = DstarPlanner(costmap, goal=(1, 1));
        >>> ds.plan()
        >>> path, status = ds.query(start=(5, 4))
        >>> print(path.T)
        >>> print(status)

    :thanks: based on D* grid planning included from `Python Robotics <https://github.com/AtsushiSakai/PythonRobotics/tree/master/PathPlanning>`_
    :seealso: :class:`PlannerBase`
    """

    def __init__(self, costmap=None, **kwargs):
        super().__init__(ndims=2, **kwargs)

        if isinstance(costmap, np.ndarray):
            self.costmap = costmap
        elif isinstance(costmap, OccupancyGrid):
            self.costmap = costmap.grid
        elif self.occgrid is not None:
            self.costmap = np.where(self.occgrid.grid > 0, np.inf, 1)
        else:
            raise ValueError("unknown type of map")
        self._Map = _Map(self.costmap.shape[0], self.costmap.shape[1])
        self._Map.costmap = self.costmap
        self._Dstar = _Dstar(self._Map)  # , tol=0)

    def plan(self, goal=None, animate=False, progress=True, summary=False):
        r"""
        Plan D* path

        :param goal: goal position :math:`(x, y)`, defaults to previously set value
        :type goal: array_like(2), optional
        :param animate: animate the planning algorithm iterations, defaults to False
        :type animate: bool, optional
        :param progress: show progress bar, defaults to True
        :type progress: bool, optional

        Compute the minimum-cost obstacle-free distance to the goal from all
        points in the grid.
        """
        if goal is not None:
            self.goal = goal

        if self._goal is None:
            raise ValueError("No goal specified here or in constructor")

        self._goal = self._goal.astype(int)

        goal_State = self._Map._Map[self._goal[1]][self._goal[0]]
        self.goal_State = goal_State

        # self._Dstar.open_list.add(goal_State)
        self._Dstar.insert(goal_State, 0)

        while True:
            ret = self._Dstar.process__State()
            # print('plan', ret, len(self._Dstar.open_list))

            if ret == -1:
                break

        if summary:
            print(self._Dstar.ninsert, self._Dstar.nin)

    @property
    def nexpand(self):
        """
        Number of node expansions

        :return: number of expansions
        :rtype: int

        This number will increase during initial planning, and also if
        replanning is invoked during the :meth:`query`.
        """
        return self._Dstar.nexpand

    def query(self, start, sensor=None, animate=False, verbose=False):
        """
        Find path with replanning

        :param start: start position :math:`(x,y)`
        :type start: array_like(2)
        :param sensor: sensor function, defaults to None
        :type sensor: callable, optional
        :param animate: animate the motion of the robot, defaults to False
        :type animate: bool, optional
        :param verbose: display detailed diagnostic information about D* operations, defaults to False
        :type verbose: bool, optional
        :return: path from start to goal, one point :math:`(x, y)` per row
        :rtype: ndarray(N,2)

        If ``sensor`` is None then the plan determined by the ``plan`` phase
        is used unaltered.

        If ``sensor`` is not None it must be callable, and is called at each
        step of the path with the current robot coordintes:

            sensor((x, y))

        and mimics the behaviour of a simple sensor onboard the robot which can
        dynamically change the costmap. The function return a list (0 or more)
        of 3-tuples (x, y, newcost) which are the coordinates of cells and their
        cost.  If the cost has changed this will trigger D* incremental
        replanning.  In this case the value returned by :meth:`nexpand` will
        increase, according to the severity of the replanning.

        :seealso: :meth:`plan`
        """
        self.start = start
        start_State = self._Map._Map[start[1]][start[0]]
        s = start_State
        s = s.parent
        tmp = start_State

        if sensor is not None and not callable(sensor):
            raise ValueError("sensor must be callable")

        cost = tmp.h
        self.goal_State.h = 0

        path = []
        while True:
            path.append((tmp.x, tmp.y))
            if tmp == self.goal_State:
                break

            # x, y = tmp.parent.x, tmp.parent.y

            if sensor is not None:
                changes = sensor((tmp.x, tmp.y))
                if changes is not None:
                    # make changes to the plan
                    for x, y, newcost in changes:
                        X = self._Dstar._Map._Map[y][x]
                        # print(f"change cost at ({x}, {y}) to {newcost}")
                        val = self._Dstar.modify_cost(X, newcost)
                    # propagate the changes to plan
                    print("propagate")
                    # self._Dstar.showparents()
                    while val != -1 and val < tmp.h:
                        val = self._Dstar.process__State(verbose=verbose)
                        # print('open set', len(self._Dstar.open_list))
                    # self._Dstar.showparents()

            tmp = tmp.parent
            # print('move to ', tmp)

        status = namedtuple(
            "_DstarStatus",
            [
                "cost",
            ],
        )

        return np.array(path), status(cost)

    # # Just feed self._h into plot function from parent as p

    # def next(self, current):
    #     if not self._valid_plan:
    #         Error("Cost _Map has changed, replan")
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

    og = np.zeros((10, 10))
    og[4:8, 3:6] = 1
    print(og)

    ds = DstarPlanner(occgrid=og)
    print(ds.costmap)

    start = (1, 1)
    goal = (7, 6)

    ds.plan(goal=goal)
    ds._Map.show_h()

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
    print(ds._Map.costmap)

    ds._Map.show_h()

    # ds._Dstar.replan()

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

    # start = m._Map[start[0]][start[1]]
    # end = m._Map[goal[0]][goal[1]]
    # _Dstar = _Dstar(m)
    # rx, ry = _Dstar.run(start, end)

    # if show_animation:
    #     plt.plot(rx, ry, "-r")
    #     plt.show(block=True)

    # # costly zone
    # m.set_cost([30, 40, 60, 80], 1.70, modify=_Dstar)

    # _Dstar.replan()

    plt.show(block=True)
