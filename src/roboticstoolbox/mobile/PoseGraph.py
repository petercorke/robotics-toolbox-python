"""
Pose graph optimization for SLAM
"""

from __future__ import annotations

import math
import time
import warnings
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import pgraph
import scipy.sparse
import scipy.sparse.linalg
import spatialmath.base as smb
from progress.bar import FillingCirclesBar
from spatialmath import SE2

import roboticstoolbox as rtb
from roboticstoolbox.mobile.OccGrid import BaseOccupancyGrid
from roboticstoolbox.tools.types import ArrayLike, NDArray

#: default Matplotlib style of the vertices and edges drawn by :meth:`PoseGraph.plot`,
#: a user's options are merged over these
VERTEX_STYLE: dict[str, Any] = dict(
    markersize=8, markerfacecolor="blue", markeredgecolor="None"
)
EDGE_STYLE: dict[str, Any] = dict(linewidth=1)


class PGVertex(pgraph.Vertex):
    """
    Vertex of a pose graph

    :param type: ``"vertex"`` for a robot pose :math:`(x, y, \\theta)`, or
        ``"landmark"`` for a landmark position :math:`(x, y)`
    :param index: index of the vertex in the graph, in order of creation
    :param kwargs: options passed to :class:`pgraph.Vertex`
    """

    def __init__(self, type: str, index: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.type = type
        self.index = index


class PGEdge(pgraph.Edge):
    """
    Edge of a pose graph, a relative pose measurement

    :param v1: first vertex
    :param v2: second vertex
    :param mean: measured relative pose :math:`(x, y, \\theta)` of ``v2`` with
        respect to ``v1``
    :param info: information matrix, the inverse of the measurement covariance
    """

    def __init__(self, v1: Any, v2: Any, mean: NDArray, info: NDArray) -> None:
        super().__init__(v1, v2)
        self.mean = mean
        self.info = info

    # The following method was inspired by, and modeled on, the MATLAB/Octave file
    # ls-slam.m by Giorgio Grisetti and Gian Diego Tipaldi.
    #
    # The theory is covered in:
    #  G. Grisetti, R. Kümmerle, C. Stachniss and W. Burgard,
    #  "A Tutorial on Graph-Based SLAM,” in IEEE Intelligent Transportation Systems Magazine,
    #  vol. 2, no. 4, pp. 31-43, winter 2010, doi: 10.1109/MITS.2010.939925.

    def linear_factors(self) -> tuple[NDArray, NDArray, NDArray]:
        """
        Linearize the error of this edge

        :return: error and its Jacobians with respect to each vertex
        :rtype: ndarray(3), ndarray(3,3), ndarray(3,3)
        """
        # computes the Taylor expansion of the error function of the k_th edge
        # v: vertices positions
        # z: edge means
        #  k:	 edge number
        #  e:	 e_k(x)
        #  A:	 d e_k(x) / d(x_i)
        #  B:	 d e_k(x) / d(x_j)
        v = self.endpoints
        v_i = v[0].coord
        v_j = v[1].coord
        z_ij = self.mean

        # compute the homoeneous transforms of the previous solutions
        zt_ij = smb.trot2(z_ij[2], t=z_ij[0:2])
        vt_i = smb.trot2(v_i[2], t=v_i[0:2])
        vt_j = smb.trot2(v_j[2], t=v_j[0:2])

        # compute the displacement between x_i and x_j
        f_ij = smb.trinv2(vt_i) @ vt_j

        # this below is too long to explain, to understand it derive it by hand
        theta_i = v_i[2]
        ti = v_i[0:2]
        tj = v_j[0:2]
        dt_ij = tj - ti

        si = np.sin(theta_i)
        ci = np.cos(theta_i)

        # fmt: off
        A =  np.array([
            [-ci, -si, [-si,  ci] @ dt_ij],
            [ si, -ci, [-ci, -si] @ dt_ij],
            [ 0,   0,  -1 ],
                ])
        B  = np.array([
            [  ci, si, 0],
            [ -si, ci, 0],
            [  0,  0,  1 ],
                ])
        # fmt: on
        ztinv = smb.trinv2(zt_ij)
        T = ztinv @ f_ij
        e = smb.tr2xyt(T)
        ztinv[0:2, 2] = 0
        A = ztinv @ A
        B = ztinv @ B

        return e, A, B


class PoseGraph:
    r"""
    Pose graph optimization for SLAM

    :param filename: name of file to load, G2O or TORO format, can be zipped
    :param lidar: load lidar scans, defaults to False
    :param verbose: print a summary of the file when loaded and the cost of
        each optimization iteration, defaults to False
    :raises RuntimeError: file syntax error
    :raises ValueError: an edge references an unknown vertex, or a lidar
        record precedes any vertex

    A pose graph has a vertex for each robot pose :math:`(x, y, \theta)` and
    an edge for each relative pose measurement between two poses.  Optimization
    adjusts the poses to best satisfy all the measurements.

    The vertices are numbered in the order they appear in the file, and
    :meth:`pose`, :meth:`scan` and :meth:`time` take this number, not the
    vertex name which is the identifier in the file.

    Example:

    .. runblock:: pycon

        >>> from roboticstoolbox import PoseGraph
        >>> pg = PoseGraph("data/pg1.g2o")
        >>> pg.optimize(iterations=3)

    :references:

        - Robotics, Vision & Control for Python, §6.5, P. Corke, Springer 2023.
        - A Tutorial on Graph-Based SLAM,
          G. Grisetti, R. Kümmerle, C. Stachniss and W. Burgard,
          in IEEE Intelligent Transportation Systems Magazine,
          vol. 2, no. 4, pp. 31-43, winter 2010, doi: 10.1109/MITS.2010.939925.

    .. seealso:: :meth:`optimize` :meth:`plot` :meth:`scanmap`
    """

    def __init__(
        self, filename: str, lidar: bool = False, verbose: bool = False
    ) -> None:
        # parse the file data
        # we assume g2o format
        #    VERTEX* vertex_id X Y THETA
        #    EDGE* startvertex_id endvertex_id X Y THETA IXX IXY IYY IXT IYT ITT
        # vertex numbers start at 0

        self.lidar = lidar

        self.verbose = verbose
        self.graph = pgraph.UGraph(verbose=verbose)

        path = Path(rtb.rtb_path_to_datafile(filename))

        opener: Callable[..., Any]
        fname: Path | str
        if path.suffix == ".zip":
            zf = zipfile.ZipFile(path, "r")
            opener = cast("Callable[..., Any]", zf.open)
            fname = path.stem
        else:
            opener = open
            fname = path

        with opener(fname, "r") as f:
            toroformat = False
            nlidar = 0

            # indices into ROBOTLASER1 record for the 3x3 info matrix in column major
            # order
            # IXX IXY IXT IYY IYT ITT
            #   6   7   8   9  10  11
            #   0   1   2   3   4   5
            g2o = [0, 1, 2, 1, 3, 4, 2, 4, 5]
            # IXX IXY IYY ITT IXT IYT
            #   6   7   8   9  10  11
            #   0   1   2   3   4   5
            toro = [0, 1, 4, 1, 2, 5, 4, 5, 3]

            # we keep an array self. = vindex(gi) to map g2o vertex index to PGraph vertex index

            # vertex index to vertex, only for vertices with a lidar scan
            self.vindex: dict[int, PGVertex] = {}
            # all the vertices in order of index
            self._vertices: list[PGVertex] = []
            firstlidar = True
            v: PGVertex | None = None
            for line_number, line in enumerate(f, start=1):
                # for zip file, we get data as bytes not str
                if isinstance(line, bytes):
                    line = line.decode()

                # is it a comment?
                if line.startswith("#"):
                    continue

                tokens = line.split()

                # is it a blank line?
                if not tokens:
                    continue

                # g2o format records
                if tokens[0] == "VERTEX_SE2":
                    coord = np.array([float(x) for x in tokens[2:5]])
                    id = tokens[1]

                    v = self._add_vertex("vertex", coord, id)

                elif tokens[0] == "VERTEX_XY":
                    coord = np.array([float(x) for x in tokens[2:4]])
                    id = tokens[1]

                    v = self._add_vertex("landmark", coord, id)

                elif tokens[0] == "EDGE_SE2":
                    v1 = tokens[1]
                    v2 = tokens[2]

                    # create the edge data as a structure
                    #  X  Y  T
                    #  3  4  5
                    mean = np.array([float(x) for x in tokens[3:6]])

                    # IXX IXY IXT IYY IYT ITT
                    #   6   7   8   9  10  11
                    d = np.array([float(x) for x in tokens[6:12]])
                    info = d[g2o].reshape((3, 3))

                    # create the edge
                    try:
                        v1 = self.graph[v1]
                        v2 = self.graph[v2]
                    except KeyError as exc:
                        raise ValueError(
                            f"Edge references unknown vertex {exc.args[0]!r} in {filename} at line {line_number}"
                        ) from exc
                    e = PGEdge(v1, v2, mean, info)
                    v1.connect(v2, edge=e)

                ## TORO format records
                # https://openslam-org.github.io/toro.html
                elif tokens[0] == "VERTEX2":
                    toroformat = True

                    coord = np.array([float(x) for x in tokens[2:5]])
                    id = tokens[1]

                    v = self._add_vertex("vertex", coord, id)

                elif tokens[0] == "EDGE2":
                    toroformat = True

                    v1 = tokens[1]
                    v2 = tokens[2]

                    # create the edge data as a structure
                    #  X  Y  T
                    #  3  4  5
                    mean = np.array([float(x) for x in tokens[3:6]])

                    # IXX IXY IYY ITT IXT IYT
                    #   6   7   8   9  10  11
                    d = np.array([float(x) for x in tokens[6:12]])
                    info = d[toro].reshape((3, 3))

                    # create the edge
                    try:
                        v1 = self.graph[v1]
                        v2 = self.graph[v2]
                    except KeyError as exc:
                        raise ValueError(
                            f"Edge references unknown vertex {exc.args[0]!r} in {filename} at line {line_number}"
                        ) from exc
                    e = PGEdge(v1, v2, mean, info)
                    v1.connect(v2, edge=e)

                elif tokens[0] == "ROBOTLASER1":
                    if not lidar:
                        continue

                    if v is None:
                        raise ValueError(
                            f"Lidar record before any vertex in {filename} at line {line_number}"
                        )

                    # lidar records are associated with the immediately
                    # preceding VERTEX record
                    #
                    # not quite sure what all the fields are
                    # 1 ?
                    # 2 min scan angle
                    # 3 scan range
                    # 4 angular increment
                    # 5 maximum range possible
                    # ?
                    # 8 N = number of beams
                    # 9 to 9+N lidar range data
                    # ?
                    # 9+N+12 timestamp (*nix timestamp)
                    # 9+N+13 lidar type (str)

                    if firstlidar:
                        nbeams = int(tokens[8])
                        lidarmeta = tokens[2:6]
                        firstlidar = False

                    # add attributes to the most recent vertex created
                    v.theta = np.arange(0, nbeams) * float(tokens[4]) + float(tokens[2])
                    v.range = np.array([float(x) for x in tokens[9 : nbeams + 9]])
                    v.time = float(tokens[21 + nbeams])
                    self.vindex[v.index] = v
                    nlidar += 1
                else:
                    raise RuntimeError(f"Unexpected line  {line} in {filename}")

            if toroformat:
                filetype = "TORO/LAGO"
            else:
                filetype = "g2o"
            if verbose:
                print(
                    f"loaded {filetype} format file: {self.graph.n} vertices,"
                    f" {self.graph.ne} edges"
                )

            if nlidar > 0:
                lidar_f = [float(x) for x in lidarmeta]

                self._angmin = lidar_f[0]
                self._angmax = sum(lidar_f[0:2])
                self._maxrange = lidar_f[3]

                if verbose:
                    fov = np.degrees([self._angmin, self._angmax])
                    print(
                        f"  {nlidar} lidar scans: {nbeams} beams, fov {fov[0]:.1f}° to"
                        f" {fov[1]:.1f}°, max range {self._maxrange}"
                    )

    def _add_vertex(self, type: str, coord: NDArray, name: str) -> PGVertex:
        v = PGVertex(type, len(self._vertices), coord=coord, name=name)
        self.graph.add_vertex(v)
        self._vertices.append(v)
        return v

    def _vertex(self, i: int) -> PGVertex:
        if not 0 <= i < len(self._vertices):
            raise IndexError(
                f"no vertex {i}, the graph has {len(self._vertices)} vertices"
            )
        return self._vertices[i]

    def _scan_vertex(self, i: int) -> PGVertex:
        v = self._vertex(i)
        if not hasattr(v, "range"):
            raise ValueError(
                f"vertex {i} has no lidar scan, load the graph with lidar=True"
            )
        return v

    def scan(self, i: int) -> tuple[NDArray, NDArray]:
        """
        Lidar scan at a vertex

        :param i: vertex number
        :raises IndexError: there is no such vertex
        :raises ValueError: the vertex has no lidar scan
        :return: range and bearing of each beam
        :rtype: ndarray(N), ndarray(N)

        .. seealso:: :meth:`scanxy`
        """
        v = self._scan_vertex(i)
        return v.range, v.theta

    def scanxy(self, i: int) -> NDArray:
        """
        Lidar scan at a vertex as points in the sensor frame

        :param i: vertex number
        :raises IndexError: there is no such vertex
        :raises ValueError: the vertex has no lidar scan
        :return: points, one per column, NaN for beams that returned no
            range less than the maximum range of the sensor
        :rtype: ndarray(2,N)

        .. seealso:: :meth:`scan`
        """
        range, theta = self.scan(i)
        range = np.where(range < self._maxrange, range, np.nan)
        x = range * np.cos(theta)
        y = range * np.sin(theta)

        return np.c_[x, y].T

    def plot_scan(self, n: int | ArrayLike, pause: float = 1, **kwargs: Any) -> None:
        """
        Plot lidar scans

        :param n: vertex number, or a sequence of them
        :param pause: time to display each scan, defaults to 1 second
        :param kwargs: Matplotlib line options, merged over a marker size of 10

        Each scan is plotted in the sensor frame, in turn.

        .. seealso:: :meth:`scanxy`
        """
        if pause <= 0:
            raise ValueError("pause must be greater than zero")
        style = dict(markersize=10) | kwargs
        for i in smb.getvector(n):
            x, y = self.scanxy(int(i))
            plt.plot(x, y, ".", **style)
            plt.pause(pause)

    def pose(self, i: int) -> NDArray:
        """
        Pose of a vertex

        :param i: vertex number
        :raises IndexError: there is no such vertex
        :return: pose :math:`(x, y, \\theta)`, or position :math:`(x, y)` for a
            landmark, as stored in the graph so that it reflects any optimization
        :rtype: ndarray(3) or ndarray(2)
        """
        return self._vertex(i).coord

    def time(self, i: int) -> float:
        """
        Timestamp of the lidar scan at a vertex

        :param i: vertex number
        :raises IndexError: there is no such vertex
        :raises ValueError: the vertex has no lidar scan
        :return: timestamp, as a Unix time
        """
        return self._scan_vertex(i).time

    def plot(
        self,
        vertex_style: dict[str, Any] | None = None,
        edge_style: dict[str, Any] | None = None,
        *,
        vopt: dict[str, Any] | None = None,
        eopt: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Plot the pose graph

        :param vertex_style: Matplotlib options for the vertices, merged over
            :data:`VERTEX_STYLE`
        :param edge_style: Matplotlib options for the edges, merged over
            :data:`EDGE_STYLE`
        :param vopt: deprecated, use ``vertex_style``
        :param eopt: deprecated, use ``edge_style``
        :param kwargs: options passed to ``pgraph.UGraph.plot``

        .. deprecated:: 1.5.0
            ``vopt`` and ``eopt`` are deprecated, use ``vertex_style`` and
            ``edge_style``.

        .. seealso:: :meth:`optimize`
        """
        for old, new, value in (
            ("vopt", "vertex_style", vopt),
            ("eopt", "edge_style", eopt),
        ):
            if value is None:
                continue
            warnings.warn(
                f"{old} is deprecated, use {new} instead",
                DeprecationWarning,
                stacklevel=2,
            )
            if new == "vertex_style" and vertex_style is None:
                vertex_style = value
            elif new == "edge_style" and edge_style is None:
                edge_style = value

        # the option names of pgraph before 1.0 are forwarded to the new names
        if "text" in kwargs:
            text = kwargs.pop("text")
            if text is False:
                kwargs.setdefault("labels", False)
            elif isinstance(text, dict):
                kwargs.setdefault("text_opt", text)

        self.graph.plot(
            color_components=False,
            force_2d=True,
            vertex_opt=VERTEX_STYLE | (vertex_style or {}),
            edge_opt=EDGE_STYLE | (edge_style or {}),
            **kwargs,
        )
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid(True)

    def scanmatch(self, s1: int, s2: int) -> SE2:
        """
        Relative pose between two scans by scan matching

        :param s1: vertex number of the first scan
        :param s2: vertex number of the second scan
        :raises IndexError: there is no such vertex
        :raises ValueError: a vertex has no lidar scan
        :return: relative pose, found by iterative closest point

        .. seealso:: :func:`spatialmath.base.ICP2d`
        """
        p1 = self.scanxy(s1)
        p2 = self.scanxy(s2)
        T = smb.ICP2d(p1, p2)
        return SE2(T)

    def scanmap(
        self, occgrid: BaseOccupancyGrid, maxrange: float | None = None, step: int = 5
    ) -> None:
        """
        Build an occupancy grid from the lidar scans

        :param occgrid: occupancy grid to update
        :param maxrange: ignore range readings beyond this distance, defaults to
            None for all
        :param step: use every ``step``-th scan, defaults to 5
        :raises ValueError: the graph has no lidar scans

        Each scan is placed at the pose of its vertex, and the cells along each
        ray are incremented as free, and the cell it ends in is decremented as
        occupied.  Afterwards cells are -1 (occupied), +1 (free) or 0 (unknown).

        .. note:: Scans are taken in vertex order, so the result depends on the
            current vertex poses, and is best made after :meth:`optimize`.

        .. seealso:: :meth:`scanxy` :meth:`~roboticstoolbox.mobile.OccGrid.BaseOccupancyGrid.line_world`
        """
        indices = range(0, self.graph.n, step)
        bar = FillingCirclesBar(
            "Converting", max=len(indices), suffix="%(percent).1f%% - %(eta)ds"
        )

        grid1d = occgrid.ravel
        for i in indices:
            xy = self.scanxy(i)
            r, theta = self.scan(i)
            if maxrange is not None:
                toofar = np.where(r > maxrange)[0]
                xy = np.delete(xy, toofar, axis=1)
            xyt = self.pose(i)

            xy = SE2(xyt) * xy

            for p2 in xy.T:
                # all cells along the ray, from the sensor to the point
                try:
                    k = occgrid.line_world(xyt[:2], p2)
                except ValueError:
                    # silently ignore rays to points outside the grid map
                    continue

                # increment cells along the ray, these are free space
                grid1d[k[:-1]] += 1
                # decrement target cell
                grid1d[k[-1]] -= 1

            bar.next()

        grid1d[grid1d < 0] = -1
        grid1d[grid1d > 0] = 1
        bar.finish()

    # The following methods were inspired by, and modeled on, the MATLAB/Octave file
    # ls-slam.m by Giorgio Grisetti and Gian Diego Tipaldi.
    #
    # The theory is covered in:
    #  G. Grisetti, R. Kümmerle, C. Stachniss and W. Burgard,
    #  "A Tutorial on Graph-Based SLAM,” in IEEE Intelligent Transportation Systems Magazine,
    #  vol. 2, no. 4, pp. 31-43, winter 2010, doi: 10.1109/MITS.2010.939925.

    def optimize(
        self,
        iterations: int = 10,
        animate: bool = False,
        retain: bool = True,
        vertex_style: dict[str, Any] | None = None,
        edge_style: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Optimize the pose graph

        :param iterations: maximum number of iterations, defaults to 10
        :param animate: show the graph after each iteration, defaults to False
        :param retain: retain the graph of each iteration in the animation, in
            increasingly dark gray, defaults to True
        :param vertex_style: Matplotlib options for the vertices when animating
        :param edge_style: Matplotlib options for the edges when animating, the
            color is set by ``retain``
        :param kwargs: options passed to ``pgraph.UGraph.plot`` when animating
        :raises NotImplementedError: the graph has landmark vertices

        .. deprecated:: 1.5.0
            ``vopt`` and ``eopt`` are deprecated, use ``vertex_style`` and
            ``edge_style``.

        Each iteration solves the linearized least-squares problem for a
        correction to all the poses, which are updated in place.  Iteration
        stops early if the total error does not decrease.

        .. seealso:: :meth:`linearize_and_solve` :meth:`plot`
        """
        if any(v.type == "landmark" for v in self.graph):
            raise NotImplementedError("optimize does not support landmark vertices")

        for old, new in (("vopt", "vertex_style"), ("eopt", "edge_style")):
            if old in kwargs:
                warnings.warn(
                    f"{old} is deprecated, use {new} instead",
                    DeprecationWarning,
                    stacklevel=2,
                )
                value = kwargs.pop(old)
                if new == "vertex_style" and vertex_style is None:
                    vertex_style = value
                elif new == "edge_style" and edge_style is None:
                    edge_style = value

        eprev = math.inf
        eo = edge_style or {}

        if animate and retain:
            colors = plt.cm.Greys(np.linspace(0.3, 1, iterations))

        for i in range(iterations):
            if animate:
                if retain:
                    if i == 0:
                        ax = smb.axes_logic(None, 2)
                else:
                    ax = smb.axes_logic(None, 2, new=True)

                if not retain:
                    eopt = eo
                else:
                    eopt = {**eo, **dict(color=tuple(colors[i, :]), label=i)}
                if vertex_style is not None:
                    kwargs["vertex_opt"] = vertex_style
                self.graph.plot(
                    edge_opt=eopt,
                    force_2d=True,
                    color_components=False,
                    ax=ax,
                    **kwargs,
                )
                plt.pause(0.5)

            energy = self.linearize_and_solve()

            if energy >= eprev:
                break
            eprev = energy

    def linearize_and_solve(self) -> float:
        """
        One iteration of pose graph optimization

        :return: total cost, the sum of squared errors of all the edges, before
            the poses are updated

        If the graph is ``verbose`` the cost and the time taken are printed.

        The first vertex is anchored, since the measurements are all relative.
        The system matrix is assembled from sparse 3-by-3 blocks, so assembly
        storage grows with the graph's vertices and edges rather than the
        square of its vertex count.

        .. seealso:: :meth:`optimize`
        """
        t0 = time.time()

        g = self.graph

        # Each edge contributes four 3x3 blocks. Assemble only those blocks,
        # rather than allocating and scanning a dense (3N, 3N) matrix.
        blocks: dict[tuple[int, int], NDArray] = {(0, 0): np.zeros((3, 3))}
        b = np.zeros((g.n * 3, 1))

        etotal = 0
        for edge in g.edges():
            e, A, B = edge.linear_factors()
            omega = edge.info

            # compute the blocks of H^k
            bi = -A.T @ omega @ e
            bj = -B.T @ omega @ e
            Hii = A.T @ omega @ A
            Hij = A.T @ omega @ B
            Hjj = B.T @ omega @ B

            v = edge.endpoints
            i = v[0].index
            j = v[1].index

            islice = slice(i * 3, (i + 1) * 3)
            jslice = slice(j * 3, (j + 1) * 3)

            # RVC3, Appendix F.2.4.1: accumulate H_ij += ... and b_i += ...
            # over the edges, storing only the nonzero blocks of H.
            # Keep the original accumulation order: sorting duplicate COO
            # entries can change rounding in poorly conditioned pose graphs.
            for key, block in (
                ((i, i), Hii),
                ((j, j), Hjj),
                ((i, j), Hij),
                ((j, i), Hij.T),
            ):
                if key in blocks:
                    blocks[key] += block
                else:
                    blocks[key] = block.copy()

            b[islice, 0] += bi
            b[jslice, 0] += bj

            etotal += np.inner(e, e)

        # Relative constraints leave a global translation and rotation free.
        # Anchor the first pose by adding an identity block to its diagonal.
        blocks[0, 0] += np.eye(3)

        # Expand each block's vertex indices into scalar matrix coordinates
        # in the same row-major order as its nine flattened coefficients.
        indices = np.array(list(blocks))
        rows = (3 * indices[:, 0, None] + np.repeat(np.arange(3), 3)).ravel()
        cols = (3 * indices[:, 1, None] + np.tile(np.arange(3), 3)).ravel()
        SH = scipy.sparse.coo_matrix(
            (np.array(list(blocks.values())).ravel(), (rows, cols)),
            shape=(g.n * 3, g.n * 3),
        ).tocsr()
        # Match the nonzero structure of the original dense-to-CSR conversion.
        SH.eliminate_zeros()
        deltax = scipy.sparse.linalg.spsolve(SH, b)  # SH \ b

        # split the increments in nice 3x1 vectors and sum them up to the original matrix
        for vertex in g:
            i = vertex.index * 3
            vertex.coord += deltax[i : i + 3]
            # normalize the angles between -PI and PI
            vertex.coord[2] = smb.angdiff(vertex.coord[2])

        if self.verbose:
            dt = time.time() - t0
            print(f"done in {dt * 1e3:0.2f} msec.  Total cost {etotal:g}")

        return etotal


if __name__ == "__main__":
    pg = PoseGraph("data/pg1.g2o")
    pg.optimize(animate=True, retain=False)

    plt.show(block=True)

    # pg = PoseGraph("data/killian-small.toro");
    # pg.optimize()

    # scan = PoseGraph("killian.g2o", lidar=True, verbose=False)
    # print(scan.graph.nc)
    # # scan.plot()
    # w = scan.scanmap(maxrange=40)
    # scan.plot_occgrid(w)
