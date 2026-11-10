"""
Bug2 reactive navigation

@Author: Peter Corke, original MATLAB code and Python version
@Author: Kristian Gibson, initial MATLAB port
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from spatialmath import base
from spatialmath.base.types import R3, ArrayLike2

from roboticstoolbox.mobile.PlannerBase import NoPathError, PlannerBase
from roboticstoolbox.tools.types import NDArray

# The robot has crossed the m-line when it is within this distance of it, in
# cells.  Half a cell is not enough: a line at 45 degrees to the grid passes
# between two diagonal neighbours at sqrt(2)/2 from each, so the robot could step
# over it without being within half a cell of it at any point on its path.  The
# small addition absorbs rounding, since a diagonal neighbour is exactly
# sqrt(2)/2 away.
_M_LINE_TOLERANCE = np.sqrt(2) / 2 + 1e-9


class Bug2(PlannerBase):
    """
    Bug2 reactive navigator

    :param kwargs: common arguments for :class:`PlannerBase` superclass, in
        particular ``occgrid``
    :return: Bug2 reactive navigator

    Creates an object which simulates an automaton, capable of omnidirectional
    motion, finding a path across an occupancy grid using only a bump sensor.

    :reference: "Path-Planning Strategies for a Point Mobile Automaton Moving
        Amidst Unknown Obstacles of Arbitrary Shape", Lumelsky and Stepanov,
        Algorithmica (1987)2, pp.403-430

    .. note:: This class is not a planner, even though it subclasses
        :class:`PlannerBase`. It can produce very inefficient paths.

    :author: Kristian Gibson and Peter Corke, based on MATLAB version by Peter Corke

    .. seealso:: :class:`PlannerBase`
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(ndims=2, **kwargs)

        self.H: list[NDArray] = []  # hit points
        self._j = 0
        self._step = 1
        self._m_line: R3 | None = None
        self._edge: list[NDArray] | None = None
        self._k: int | None = None

    @property
    def m_line(self) -> R3 | None:
        """
        Get m-line

        :return: m-line in homogeneous form, None if :meth:`run` has not been
            called
        :rtype: ndarray(3)

        This is the m-line computed for the last :meth:`run`.
        """
        return self._m_line

    # the Bug2 automaton is run, rather than planned and then queried
    def run(
        self,
        start: ArrayLike2 | None = None,
        goal: ArrayLike2 | None = None,
        animate: bool | None = None,
        pause: float = 0.001,
        trail: bool = True,
        movie: str | tuple[str, float] | None = None,
        label: str = "robot path",
    ) -> NDArray:
        """
        Find a path using Bug2 reactive navigation algorithm

        :param start: starting position, defaults to the planner's start
        :param goal: goal position, defaults to the planner's goal
        :param animate: show animation of robot moving over the map,
            defaults to the planner's :attr:`animate`
        :param pause: time to pause between frames of the animation, defaults to
            0.001 seconds
        :param trail: show the path followed by the robot, defaults to True
        :param movie: save animation as a movie, defaults to None. Is either
            name of movie or a tuple (filename, frame interval).  Implies
            ``animate``.
        :type movie: str or tuple(str, float)
        :param label: legend label for the path followed by the robot, defaults
            to "robot path"
        :raises ValueError: ``start`` and ``goal`` are not both given, or lie
            inside an obstacle
        :raises NoPathError: the robot is trapped, or the goal is enclosed by
            an obstacle
        :return: path from ``start`` to ``goal``, one point per row
        :rtype: ndarray(N,2)

        Compute the path from ``start`` to ``goal`` assuming the robot is capable
        of 8-way motion from its current cell to the next.

        .. note:: ``start`` and ``goal`` are (x,y) world coordinates, rounded to
              cell centres using the occupancy grid's origin and cell size.
              The returned path and hit points use the same world coordinates.

        .. versionchanged:: 1.5.0
            The robot is considered to have crossed the m-line when it is within
            :math:`\\sqrt{2}/2` cells of it, previously half a cell.  A line at
            45 degrees to the grid passes between two diagonal neighbours at that
            distance from each, and could be stepped over without being seen, so
            that the robot was reported as trapped.  This shortens the house
            floorplan example, from the bedroom to the kitchen, by one point.

        .. versionchanged:: 1.5.0
            The robot leaves an obstacle only where the next step along the
            m-line is free.  Previously it could leave, find the obstacle in
            the way, and hit it again from the same cell, so that cell appeared
            twice in the path, and a goal enclosed by an obstacle was never
            given up on.  The cell where an obstacle is first hit is also no
            longer repeated in the path.  Together these remove two repeated
            points at the places the robot left an obstacle blocked, and two at
            the first hits, so that the house floorplan example from the bedroom
            to the kitchen has 1302 points rather than the 1307 of earlier
            versions.

        .. seealso:: :meth:`Bug2.plot`
        """

        # make sure start and goal are set and valid
        self.validate_start_goal(start, goal)
        if self._start is None or self._goal is None:
            raise ValueError("start and goal must be specified")

        if self.occgrid is not None:
            # The automaton moves between cell centres, in world coordinates.
            self._start = self.occgrid.g2w(self.occgrid.w2g(self._start))
            self._goal = self.occgrid.g2w(self.occgrid.w2g(self._goal))

        self.H = []
        self._step = 1

        if np.all(self._start == self._goal):
            # nothing to do, and there is no m-line
            self._m_line = None
            return np.atleast_2d(self._start)

        # compute the m-line
        #  create homogeneous representation of the line
        #  line*[x y 1]' = 0
        self._m_line = hom_line(self._start, self._goal)

        def draw() -> None:
            self.plot()
            self.plot_m_line()
            plt.pause(0.05)

        animating = self.animation_start(animate, movie, draw=draw)

        robot: NDArray | None = self._start
        path: NDArray = self._start

        trail_line = trail_head = None
        if animating:
            (trail_line,) = plt.plot(0, 0, "y.", label=label)
            (trail_head,) = plt.plot(0, 0, "ko", zorder=10)

        # iterate using the next() method until we reach the goal
        try:
            while True:
                assert robot is not None  # nosec B101: narrows the type for mypy
                if animating:
                    assert trail_head is not None and trail_line is not None  # nosec B101: narrows the type for mypy
                    trail_head.set_data([robot[0]], [robot[1]])
                    if trail:
                        x, y = np.atleast_2d(path).T
                        trail_line.set_data(x, y)

                    self.animation_frame(pause)

                # move to next point on path
                robot = self.next(robot)

                # are we there yet?
                if robot is None:
                    break
                else:
                    path = np.vstack((path, robot))
        finally:
            self.animation_end()
        if trail_head is not None:
            trail_head.remove()

        return path

    def plot_m_line(self, *args: Any, **kwargs: Any) -> None:
        """
        Plot m-line

        :param args: positional arguments passed to
            :meth:`~matplotlib.axes.Axes.plot`, for example a format string
        :param kwargs: keyword arguments passed to
            :meth:`~matplotlib.axes.Axes.plot`
        :raises ValueError: there is no m-line, :meth:`run` has not been called

        Plots the m-line on the current axes.  By default it is a black dashed
        line.
        """
        if self._m_line is None or self._start is None:
            raise ValueError("no m-line, call run() first")

        if not args and not kwargs:
            args = ("k--",)
        style: dict[str, Any] = {"zorder": 10, "label": "m-line"} | kwargs

        x_min, x_max = plt.gca().get_xlim()
        y_min, y_max = plt.gca().get_ylim()
        if self._m_line[1] == 0:
            # handle the case that the line is vertical
            plt.plot([self._start[0], self._start[0]], [y_min, y_max], *args, **style)
        else:
            # regular line
            x = np.array([[x_min, 1], [x_max, 1]])
            y = -x @ np.r_[self._m_line[0], self._m_line[2]]
            y = y / self._m_line[1]
            plt.plot([x_min, x_max], y, *args, **style)

    def _m_line_move(self, position: NDArray) -> tuple[float, float]:
        """
        Step along the m-line toward the goal

        :param position: current robot position
        :return: step (dx, dy) in world units, at most one cell along each axis
        """
        assert self._goal is not None and self._m_line is not None  # nosec B101: narrows the type for mypy
        m_line = self._m_line
        d = self._goal - position
        cellsize = self.occgrid.cellsize if self.occgrid is not None else 1
        if abs(d[0]) > abs(d[1]):
            # line slope less than 45 deg
            dx = cellsize if d[0] >= 0 else -cellsize
            y = -((position[0] + dx) * m_line[0] + m_line[2]) / m_line[1]
            dy = round((y - position[1]) / cellsize) * cellsize
        else:
            # line slope greater than 45 deg
            dy = cellsize if d[1] >= 0 else -cellsize
            x = -((position[1] + dy) * m_line[1] + m_line[2]) / m_line[0]
            dx = round((x - position[0]) / cellsize) * cellsize
        return dx, dy

    def next(self, position: NDArray) -> NDArray | None:
        """
        One step of the finite state automaton

        :param position: current robot position
        :type position: ndarray(2)
        :raises ValueError: there is no goal or m-line, :meth:`run` has not been
            called
        :raises NoPathError: robot is trapped
        :return: next robot position, None if the goal has been reached
        :rtype: ndarray(2)
        """
        goal = self._goal
        m_line = self._m_line
        if goal is None or m_line is None:
            raise ValueError("no goal or m-line, call run() first")

        if self.occgrid is not None:
            position = self.occgrid.g2w(self.occgrid.w2g(position))

        if np.all(goal == position):
            return None  # we have arrived

        if self._step == 1:
            # Step 1.  Move along the M-line toward the goal
            self.message(f"{position}: moving along the M-line (step 1)")
            # motion on line toward goal
            dx, dy = self._m_line_move(position)

            # detect if next step is an obstacle
            if self.isoccupied(position + np.r_[dx, dy]):
                self.message(f"  {position}: obstacle at {position + np.r_[dx, dy]}")
                # each hit point is closer to the goal than the one before, so
                # meeting the same one again means no progress is being made
                # and the goal is enclosed by the obstacle
                if any(np.all(position == h) for h in self.H):
                    raise NoPathError("robot is trapped")
                self.H.append(position)  # save hit point
                self._step = 2  # transition to step 2
                self.message(f"  {position}: change to step 2")

                # get a list of all the points around the obstacle
                assert self.occgrid is not None  # nosec B101: narrows the type for mypy
                edge, _ = edgelist(self.occgrid.grid == 0, self.occgrid.w2g(position))
                self._edge = [self.occgrid.g2w(p) for p in edge]
                # the first point of the edge list is the hit point, where the
                # robot already is
                self._k = 1
            else:
                n = position + np.array([dx, dy])
                if self.occgrid is not None:
                    n = self.occgrid.g2w(self.occgrid.w2g(n))

        if self._step == 2:
            # Step 2.  Move around the obstacle until we reach a point
            # on the M-line closer than when we started.

            self.message(f"{position}: moving around the obstacle (step 2)")
            assert self._edge is not None and self._k is not None  # nosec B101: narrows the type for mypy
            if self._k < len(self._edge):
                n = self._edge[self._k]  # next edge point
            else:
                # we are at the end of the list of edge points, we
                # are back where we started.  Step 2.c test.
                raise NoPathError("robot is trapped")

            # are we on the M-line now ?
            assert self.occgrid is not None  # nosec B101: narrows the type for mypy
            if (
                abs(np.inner(np.r_[position, 1], m_line))
                <= _M_LINE_TOLERANCE * self.occgrid.cellsize
            ):
                self.message(f"  {position}: crossed the M-line")

                # are we closer than when we encountered the obstacle, and can
                # we move toward the goal from here?  Leaving where the next
                # step along the m-line is an obstacle would only hit it again
                # at once, and for a goal enclosed by the obstacle would never
                # end.
                dx, dy = self._m_line_move(position)
                if base.norm(position - goal) < base.norm(
                    self.H[-1] - goal
                ) and not self.isoccupied(position + np.r_[dx, dy]):
                    self._step = 1  # transition to step 1
                    self.message(f"  {position}: change to step 1")
                    return n

            # no, keep going around
            self._k += 1

        return n


# Ported from Peter Corke's edgelist function found:
# https://github.com/petercorke/toolbox-common-matlab/blob/master/edgelist.m


#  these are directions of 8-neighbours in a clockwise direction, as (x, y)
#  with y increasing down the image: W SW S SE E NE N NW
# fmt: off
_dirs = np.array([
        [-1,  0],
        [-1,  1],
        [ 0,  1],
        [ 1,  1],
        [ 1,  0],
        [ 1, -1],
        [ 0, -1],
        [-1, -1],
    ])
# fmt: on


def _inside(im: NDArray, x: int, y: int) -> bool:
    """Is the (column, row) coordinate within the image"""
    return 0 <= x < im.shape[1] and 0 <= y < im.shape[0]


def edgelist(
    im: NDArray, p: ArrayLike2, direction: int = 1
) -> tuple[list[NDArray], list[int]]:
    """
    Find edge of a region

    :param im: binary image
    :type im: ndarray(h,w,int)
    :param p: initial point
    :param direction: direction to traverse region, +1 clockwise [default], -1
        counter-clockwise
    :raises ValueError: initial point is not within the image, or has no
        neighbour outside the blob
    :return: edge list, direction vector list
    :rtype: tuple of lists

    ``edge, dirs = edgelist(im, seed)`` is the boundary/contour/edge of a region
    in the binary image ``im``.  ``seed=[X,Y]`` is the coordinate of a point on
    the edge of the region of interest, but belonging to the region.

    ``edge`` is a list of coordinates (2) of edge pixels of a region in the
    image.  The elements of the edgelist are NumPy ndarray(2).

    ``dirs`` is a list of integers representing the direction of the edge from
    the corresponding point in ``edge`` to the next point in ``edge``.  The
    integers in the range 0 to 7 represent directions: W SW S SE E NE N NW
    respectively.

    - Coordinates are given and returned assuming the matrix is an image, so the
      indices are always in the form (x,y) or (column,row).
    - ``im`` is a binary image where 0 is assumed to be background, non-zero
      is an object.
    - ``p`` must be a point on the edge of the region.
    - The initial point is always the first and last element of the returned edgelist.
    - The area outside the image is not part of the region, so a region that
      touches the border of the image has the border as part of its edge.
    - 8-direction chain coding can give incorrect results when used with
      blobs founds using 4-way connectivity.

    :Reference:

    - METHODS TO ESTIMATE AREAS AND PERIMETERS OF BLOB-LIKE OBJECTS: A COMPARISON
      Luren Yang, Fritz Albregtsen, Tor Lønnestad and Per Grøttum
      IAPR Workshop on Machine Vision Applications Dec. 13-15, 1994, Kawasaki

    """
    pos: NDArray = np.asarray(base.getvector(p, 2, dtype=int))
    if direction > 0:
        neighbours = np.arange(0, 8, 1)
    else:
        neighbours = np.arange(7, -1, -1)

    if not _inside(im, int(pos[0]), int(pos[1])):
        raise ValueError("specified coordinate is not within image")
    pix0 = im[int(pos[1]), int(pos[0])]  # color of pixel we start at

    q = adjacent_point(im, pos, pix0)

    # find an adjacent point outside the blob
    if q is None:
        raise ValueError("no neighbour outside the blob")

    e: list[NDArray] = [pos]  # initialize the edge list
    directions: list[int] = []  # initialize the direction list
    pos0 = None

    while True:
        # find which direction is Q
        dq = q - pos
        for kq in range(0, 8):
            # get index of neighbour's direction in range [0,7]
            if np.all(dq == _dirs[kq]):
                break

        # now test for directions relative to Q
        for j in neighbours:
            # get index of neighbour's direction in range [0,7]
            k = (j + kq) % 8

            # compute coordinate of the k'th neighbour
            nk = pos + _dirs[k]

            # if this neighbour is in the blob it is the next edge pixel, a
            # neighbour outside the image is not in the blob
            if _inside(im, int(nk[0]), int(nk[1])) and im[nk[1], nk[0]] == pix0:
                pos = nk
                break

            q = nk

        directions.append(int(k))
        # check if we are back where we started
        if pos0 is None:
            pos0 = pos  # make a note of where we started
        else:
            if np.all(pos == pos0):
                break

        # keep going, add this point to the edgelist
        e.append(pos)

    return e, directions


# Ported from Peter Corke's adjacent_point function found:
# https://github.com/petercorke/toolbox-common-matlab/blob/master/edgelist.m


def adjacent_point(im: NDArray, seed: NDArray, pix0: Any) -> NDArray | None:
    """
    Find adjacent point

    :param im: occupancy grid
    :type im: ndarray(m,n)
    :param seed: initial point
    :type seed: ndarray(2)
    :param pix0: value of occupancy grid at ``seed`` coordinate
    :return: coordinate of a neighbour, None if there is none
    :rtype: ndarray(2) or None

    Is a neighbouring point of the coordinate ``seed`` that is not within the
    region containing the coordinate ``seed``, ie. it is a neighbour but
    outside.  The area outside the image is not part of the region, so a
    neighbour that is off the edge of the image counts as outside it.
    """
    for d in _dirs:
        p = seed + d
        if not _inside(im, int(p[0]), int(p[1])) or im[p[1], p[0]] != pix0:
            return p

    return None


# Implementation of Peter Corke's matlab homline function from:
# https://github.com/petercorke/spatialmath-matlab/blob/master/homline.m
def hom_line(p1: ArrayLike2, p2: ArrayLike2) -> R3:
    """
    Homogeneous line through two points

    :param p1: first point :math:`(x, y)`
    :param p2: second point :math:`(x, y)`
    :raises ValueError: the points are coincident
    :return: line in homogeneous form :math:`(a, b, c)` such that a point
        :math:`(x, y)` is on the line if :math:`ax + by + c = 0`

    The line is normalized so that ``line @ [x, y, 1]`` is the distance of the
    point :math:`(x, y)` from the line.
    """
    line = np.cross(np.r_[p1[0], p1[1], 1], np.r_[p2[0], p2[1], 1])

    norm = np.linalg.norm(line[0:2])
    if norm == 0:
        raise ValueError("the points are coincident, there is no line")

    # normalize so that the result of x*l' is the pixel distance
    # from the line
    return line / norm


if __name__ == "__main__":
    from roboticstoolbox import rtb_load_matfile

    vars = rtb_load_matfile("data/map1.mat")
    map = vars["map"]

    bug2 = Bug2(occgrid=map)
    path = bug2.run([20, 10], [50, 35], animate=True)
