"""



See Wikipedia article (https://en.wikipedia.org/wiki/D*)

"""

from __future__ import annotations

import heapq
import math
import warnings
from collections.abc import Callable
from enum import IntEnum, auto
from typing import Any, NamedTuple

import matplotlib.pyplot as plt
import numpy as np
from spatialmath.base.types import ArrayLike2

from roboticstoolbox.mobile.OccGrid import BinaryOccupancyGrid, OccupancyGrid
from roboticstoolbox.mobile.PlannerBase import PlannerBase
from roboticstoolbox.tools.types import NDArray

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
    def __init__(self, x: int, y: int) -> None:
        self.x = x
        self.y = y
        self.parent: _State | None = None  # 'back pointer' to next _State
        self.t = _Tag.NEW  # open closed new
        self.h: float = 0  # cost to goal
        self.k: float = 0  # estimate of shortest path cost

    def __str__(self) -> str:
        return f"({self.x}, {self.y})"  # [{self.h:.1f}, {self.k:.1f}]"

    def __repr__(self) -> str:
        return self.__str__()

    def __lt__(self, other: object) -> bool:
        return True


class _Map:
    costmap: NDArray  # assigned by DstarPlanner

    def __init__(self, row: int, col: int) -> None:
        self.row = row
        self.col = col
        self._Map = self.init__Map()

    def init__Map(self) -> list[list[_State]]:
        _Map_list = []  # list of rows
        for i in range(self.row):
            # for each row, make a list
            tmp = []
            for j in range(self.col):
                tmp.append(_State(j, i))  # append column to the row
            _Map_list.append(tmp)  # append row to _Map
        return _Map_list

    _neighbours = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    def get_neighbors(self, _State: _State) -> list[_State]:
        _State_list = []

        for i, j in self._neighbours:
            try:
                if _State.x + i >= 0 and _State.y + j >= 0:
                    _State_list.append(self._Map[_State.y + j][_State.x + i])
            except IndexError:
                pass
        return _State_list

    _root2 = np.sqrt(2)

    def cost(self, _State1: _State, _State2: _State) -> float:
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

    def show_h(self) -> None:
        h = np.empty((self.col, self.row))

        for x in range(self.col):
            for y in range(self.row):
                h[y, x] = self._Map[y][x].h
        print(h)


class _Dstar:
    def __init__(self, _Map: _Map, tol: float = 1e-6) -> None:
        self._Map = _Map
        # self.open_list = set()
        self.open_list: list[tuple[float, _State]] = []
        self.nexpand = 0
        self.tol = tol

    def process__State(self, verbose: bool = False) -> float:
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

    def insert(self, _State: _State, h_new: float) -> None:
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

    def modify_cost(self, x: _State, newcost: float) -> float:
        self._Map.costmap[x.y, x.x] = newcost
        if x.t is _Tag.CLOSED:
            assert x.parent is not None
            self.insert(x, x.parent.h + self._Map.cost(x, x.parent))
        if len(self.open_list) == 0:
            return -1
        else:
            # lowest priority item is always at index 0 according to docs
            return self.open_list[0][0]

    def showparents(self) -> None:
        for y in range(self._Map.row - 1, -1, -1):
            if y == self._Map.row - 1:
                print("   ", end="")
                for x in range(self._Map.col):
                    print(f"  {x}   ", end="")
                print()
            print(f"{y}: ", end="")
            for x in range(self._Map.col):
                state = self._Map._Map[y][x]
                par = state.parent
                if par is None:
                    print("  G   ", end="")
                else:
                    print(f"({par.x},{par.y}) ", end="")
            print()
        print()


# ====================== RTB wrapper ============================= #


# Copyright (c) 2022 Peter Corke: https://github.com/petercorke/robotics-toolbox-python
# Released under the MIT license: https://github.com/AtsushiSakai/PythonRobotics/blob/master/LICENSE
class DstarStatus(NamedTuple):
    """Cost of the path"""

    cost: float


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
    per world unit. The cost of a step is the mean of the two cell costs
    multiplied by the distance between their centres. An infinite cost indicates
    an untraversable cell or obstacle. Endpoints and returned paths are in world
    coordinates; endpoints are rounded to cell centres.

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

    def __init__(
        self, costmap: NDArray | OccupancyGrid | None = None, **kwargs: Any
    ) -> None:
        super().__init__(ndims=2, **kwargs)

        if isinstance(costmap, np.ndarray):
            self.costmap = costmap
        elif isinstance(costmap, OccupancyGrid):
            self.costmap = costmap.grid
        elif self.occgrid is not None:
            self.costmap = np.where(self.occgrid.grid > 0, np.inf, 1)
        else:
            raise ValueError("unknown type of map")
        # Keep the cost grid's geometry instead of treating world positions as indices.
        geometry = costmap if isinstance(costmap, OccupancyGrid) else self.occgrid
        if geometry is None:
            geometry = OccupancyGrid(self.costmap)
        if geometry.shape != self.costmap.shape:
            raise ValueError("costmap and occupancy grid must have the same shape")
        if self.occgrid is not None and (
            geometry.cellsize != self.occgrid.cellsize
            or not np.array_equal(geometry.workspace, self.occgrid.workspace)
        ):
            raise ValueError("costmap and occupancy grid must have the same geometry")
        self._costgrid = geometry
        if self.occgrid is None:
            self._occgrid = BinaryOccupancyGrid(
                np.isinf(self.costmap),
                origin=geometry.g2w((0, 0)),
                cellsize=geometry.cellsize,
            )
            self._occgrid0 = self._occgrid
            self.validate_start_goal(self._start, self._goal)
        self._Map = _Map(self.costmap.shape[0], self.costmap.shape[1])
        self._Map.costmap = self.costmap * geometry.cellsize
        self._Dstar = _Dstar(self._Map)  # , tol=0)

    def plan(
        self,
        goal: ArrayLike2 | None = None,
        animate: bool | None = None,
        progress: bool = True,
        summary: bool = False,
    ) -> None:
        r"""
        Plan D* path

        :param goal: goal position :math:`(x, y)`, defaults to previously set value
        :param animate: deprecated, animating the planning is not implemented
        :param progress: deprecated, has no effect
        :param summary: print the number of insertions, defaults to False

        .. deprecated:: 1.4.5
            ``animate`` and ``progress`` were accepted but had no effect.

        Compute the minimum-cost obstacle-free distance to the goal from all
        points in the grid.
        """
        for name, value in (("animate", animate), ("progress", progress is not True)):
            if value:
                warnings.warn(
                    f"{name} is deprecated and has no effect",
                    DeprecationWarning,
                    stacklevel=2,
                )

        if goal is not None:
            self.goal = goal

        if self._goal is None:
            raise ValueError("No goal specified here or in constructor")

        c, r = self._costgrid.w2g(self._goal)
        goal_State = self._Map._Map[r][c]
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
    def nexpand(self) -> int:
        """
        Number of node expansions

        :return: number of expansions

        This number will increase during initial planning, and also if
        replanning is invoked during the :meth:`query`.
        """
        return self._Dstar.nexpand

    def query(
        self,
        start: ArrayLike2,
        sensor: Callable[..., Any] | None = None,
        animate: bool | None = None,
        verbose: bool = False,
        movie: str | tuple[str, float] | None = None,
    ) -> tuple[NDArray, DstarStatus]:
        """
        Find path with replanning

        :param start: start position :math:`(x,y)`
        :param sensor: sensor function, defaults to None
        :param animate: animate the motion of the robot, defaults to the
            planner's :attr:`animate`
        :param movie: save the animation as a movie file, filename or tuple
            (filename, frame interval), defaults to the planner's :attr:`movie`.
            Implies ``animate``.
        :param verbose: display detailed diagnostic information about D* operations, defaults to False
        :return: path from start to goal, one point :math:`(x, y)` per row
        :rtype: ndarray(N,2)

        If ``sensor`` is None then the plan determined by the ``plan`` phase
        is used unaltered.

        If ``sensor`` is not None it must be callable, and is called at each
        step of the path with the current robot coordinates in world units:

            sensor((x, y))

        and mimics the behaviour of a simple sensor onboard the robot which can
        dynamically change the costmap. The function return a list (0 or more)
        of 3-tuples (x, y, newcost) which are world coordinates of cells and their
        cost per world unit. If the cost has changed this will trigger D* incremental
        replanning.  In this case the value returned by :meth:`nexpand` will
        increase, according to the severity of the replanning.

        :seealso: :meth:`plan`
        """
        self.validate_start_goal(start)
        c, r = self._costgrid.w2g(start)
        start_State = self._Map._Map[r][c]
        tmp = start_State

        if sensor is not None and not callable(sensor):
            raise ValueError("sensor must be callable")

        cost = tmp.h
        self.goal_State.h = 0

        animating = self.animation_start(animate, movie, draw=self.plot)

        path = []
        while True:
            position = self._costgrid.g2w((tmp.x, tmp.y))
            path.append(position)
            if animating:
                plt.plot(*position, "y.", markersize=12)
                self.animation_frame()
            if tmp == self.goal_State:
                break

            # x, y = tmp.parent.x, tmp.parent.y

            if sensor is not None:
                changes = sensor(tuple(position))
                if changes is not None:
                    # make changes to the plan
                    val = -1.0
                    for x, y, newcost in changes:
                        c, r = self._costgrid.w2g((x, y))
                        if not (
                            0 <= r < self.costmap.shape[0]
                            and 0 <= c < self.costmap.shape[1]
                        ):
                            raise ValueError("sensor cell is outside the costmap")
                        X = self._Dstar._Map._Map[r][c]
                        self.costmap[r, c] = newcost
                        # print(f"change cost at ({x}, {y}) to {newcost}")
                        val = self._Dstar.modify_cost(
                            X, newcost * self._costgrid.cellsize
                        )
                    # propagate the changes to plan
                    self.message("propagate the cost changes")
                    # self._Dstar.showparents()
                    while val != -1 and val < tmp.h:
                        val = self._Dstar.process__State(verbose=verbose)
                        # print('open set', len(self._Dstar.open_list))
                    # self._Dstar.showparents()

            assert tmp.parent is not None
            tmp = tmp.parent
            # print('move to ', tmp)

        self.animation_end()

        return np.array(path), DstarStatus(cost)

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

    def sensorfunc(pos: tuple[int, int]) -> list[tuple[int, int, int]] | None:
        if pos == (3, 3):
            changes = []
            for x in range(3, 6):
                for y in range(0, 4):
                    changes.append((x, y, 100))
            return changes
        return None

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
