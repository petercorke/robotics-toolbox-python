# PoseGraph Pose graph

import roboticstoolbox as rtb
import pgraph
from spatialmath import SE2
import spatialmath.base as smb
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import zipfile
import time
import math
from pathlib import Path
from progress.bar import FillingCirclesBar


class PGVertex(pgraph.UVertex):

    nvertices = 0  # reset this before each PGGraph build

    def __init__(self, type, **kwargs):
        super().__init__(**kwargs)
        self.type = type
        self.index = PGVertex.nvertices
        PGVertex.nvertices += 1


class PGEdge(pgraph.Edge):
    def __init__(self, v1, v2, mean, info):
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

    def linear_factors(self):
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

    # properties
    #     graph

    #     ngrid
    #     center
    #     cellsize

    def __init__(self, filename, lidar=False, verbose=False):
        """
        Pose graph optimization for SLAM

        :param filename: name of file to load, G2O or TORO format, can be zipped
        :type filename: str
        :param lidar: process lidar data, defaults to False
        :type lidar: bool, optional
        :param verbose: show details of processing, defaults to False
        :type verbose: bool, optional
        :raises RuntimeError: file syntax error

        :references:

            - Robotics, Vision & Control for Python, §6.5, P. Corke, Springer 2023.
            - A Tutorial on Graph-Based SLAM,
              G. Grisetti, R. Kümmerle, C. Stachniss and W. Burgard,
              in IEEE Intelligent Transportation Systems Magazine,
              vol. 2, no. 4, pp. 31-43, winter 2010, doi: 10.1109/MITS.2010.939925.
        """

        # parse the file data
        # we assume g2o format
        #    VERTEX* vertex_id X Y THETA
        #    EDGE* startvertex_id endvertex_id X Y THETA IXX IXY IYY IXT IYT ITT
        # vertex numbers start at 0

        self.lidar = lidar

        self.graph = pgraph.UGraph(verbose=verbose)

        path = Path(rtb.rtb_path_to_datafile(filename))

        if path.suffix == ".zip":
            zf = zipfile.ZipFile(path, "r")
            opener = zf.open
            path = path.stem
        else:
            opener = open

        with opener(path, "r") as f:

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

            self.vindex = {}
            firstlidar = True
            PGVertex.nvertices = 0  # reset vertex counter in PGVertex class
            for line in f:
                # for zip file, we get data as bytes not str
                if isinstance(line, bytes):
                    line = line.decode()

                # is it a comment?
                if line.startswith("#"):
                    continue

                tokens = line.split()

                # g2o format records
                if tokens[0] == "VERTEX_SE2":
                    coord = np.array([float(x) for x in tokens[2:5]])
                    id = tokens[1]

                    v = PGVertex("vertex", coord=coord, name=id)
                    self.graph.add_vertex(v)

                elif tokens[0] == "VERTEX_XY":
                    coord = np.array([float(x) for x in tokens[2:4]])
                    id = tokens[1]

                    v = PGVertex("landmark", coord=coord, name=id)
                    self.graph.add_vertex(v)

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
                    v1 = self.graph[v1]
                    v2 = self.graph[v2]
                    e = PGEdge(v1, v2, mean, info)
                    v1.connect(v2, edge=e)

                ## TORO format records
                # https://openslam-org.github.io/toro.html
                elif tokens[0] == "VERTEX2":
                    toroformat = True

                    coord = np.array([float(x) for x in tokens[2:5]])
                    id = tokens[1]

                    v = PGVertex("vertex", coord=coord, name=id)
                    self.graph.add_vertex(v)

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
                    v1 = self.graph[v1]
                    v2 = self.graph[v2]
                    e = PGEdge(v1, v2, mean, info)
                    v1.connect(v2, edge=e)

                elif tokens[0] == "ROBOTLASER1":
                    if not lidar:
                        continue

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
            print(
                f"loaded {filetype} format file: {self.graph.n} vertices,"
                f" {self.graph.ne} edges"
            )

            if nlidar > 0:
                lidarmeta = [float(x) for x in lidarmeta]

                self._angmin = lidarmeta[0]
                self._angmax = sum(lidarmeta[0:2])
                self._maxrange = lidarmeta[3]

                fov = np.degrees([self._angmin, self._angmax])
                print(
                    f"  {nlidar} lidar scans: {nbeams} beams, fov {fov[0]:.1f}° to"
                    f" {fov[1]:.1f}°, max range {self._maxrange}"
                )

    def scan(self, i):
        v = self.vindex[i]
        return v.range, v.theta

    def scanxy(self, i):

        range, theta = self.scan(i)
        range = np.where(range < self._maxrange, range, np.nan)
        x = range * np.cos(theta)
        y = range * np.sin(theta)

        return np.c_[x, y].T

    def plot_scan(self, n):
        n = smb.getvector(n)
        for i in n:
            x, y = self.scanxy(i)
            plt.plot(x, y, ".", "MarkerSize", 10)
            plt.pause(1)

    def pose(self, i):
        return self.vindex[i].coord

    def time(self, i):
        return self.vindex[i].time

    def plot(self, **kwargs):
        if not "vopt" in kwargs:
            kwargs["vopt"] = dict(
                markersize=8, markerfacecolor="blue", markeredgecolor="None"
            )
        if not "eopt" in kwargs:
            kwargs["eopt"] = dict(linewidth=1)
        self.graph.plot(colorcomponents=False, force2d=True, **kwargs)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid(True)

    def scanmatch(self, s1, s2):
        p1 = self.scanxy(s1)
        p2 = self.scanxy(s2)
        T = smb.ICP2d(p1, p2)
        return SE2(T)

    def scanmap(self, occgrid, maxrange=None):
        # note about maxrange timing

        bar = FillingCirclesBar(
            "Converting", max=self.graph.n, suffix="%(percent).1f%% - %(eta)ds"
        )

        grid1d = occgrid.ravel
        for i in range(0, self.graph.n, 5):

            xy = self.scanxy(i)
            r, theta = self.scan(i)
            if maxrange is not None:
                toofar = np.where(r > maxrange)[0]
                xy = np.delete(xy, toofar, axis=1)
            xyt = self.vindex[i].coord

            xy = SE2(xyt) * xy

            # start of each ray
            p1 = occgrid.w2g(xyt[:2])

            for p2 in occgrid.w2g(xy.T):

                # all cells along the ray
                try:
                    k = occgrid._line(p1, p2)
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

    def w2g(self, w):
        return np.round((w - self._centre) / self._cellsize) + self._ngrid / 2

    def g2w(self, g):
        return (np.r_[g] - self._ngrid / 2) * self._cellsize + self._centre

    def plot_occgrid(self, w, block=None):

        bl = self.g2w([0, 0])
        tr = self.g2w([w.shape[1], w.shape[0]])

        w = np.where(w < 0, -1, w)  # free cells are -1
        w = np.where(w > 0, 1, w)  # occupied cells are +1
        w = -w

        # plot it
        plt.imshow(w, cmap="gray", extent=[bl[0], tr[0], bl[1], tr[1]])
        plt.xlabel("x")
        plt.ylabel("y")

        if block is not None:
            plt.show(block=block)

    # The following methods were inspired by, and modeled on, the MATLAB/Octave file
    # ls-slam.m by Giorgio Grisetti and Gian Diego Tipaldi.
    #
    # The theory is covered in:
    #  G. Grisetti, R. Kümmerle, C. Stachniss and W. Burgard,
    #  "A Tutorial on Graph-Based SLAM,” in IEEE Intelligent Transportation Systems Magazine,
    #  vol. 2, no. 4, pp. 31-43, winter 2010, doi: 10.1109/MITS.2010.939925.

    def optimize(self, iterations=10, animate=False, retain=True, **kwargs):

        eprev = math.inf
        eo = {}

        if animate and retain:
            colors = plt.cm.Greys(np.linspace(0.3, 1, iterations))
            if "eopt" in kwargs:
                eo = kwargs["eopt"]
                kwargs = {k: v for (k, v) in kwargs.items() if k != "eopt"}
            # else:
            #     eo = {}

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
                self.graph.plot(
                    eopt=eopt, force2d=True, colorcomponents=False, ax=ax, **kwargs
                )
                plt.pause(0.5)

            energy = self.linearize_and_solve()

            if energy >= eprev:
                break
            eprev = energy

    def linearize_and_solve(self):
        t0 = time.time()

        g = self.graph

        # H and b are respectively the system matrix and the system vector
        H = np.zeros((g.n * 3, g.n * 3))
        b = np.zeros((g.n * 3, 1))
        # this loop constructs the global system by accumulating in H and b the contributions
        # of all edges (see lecture)

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

            # accumulate the blocks in H and b
            try:
                H[islice, islice] += Hii
                H[jslice, jslice] += Hjj
                H[islice, jslice] += Hij
                H[jslice, islice] += Hij.T

                b[islice, 0] += bi
                b[jslice, 0] += bj
            except ValueError:
                print("linearize_and_solve failed")
                print(v)
                print(H.shape, b.shape, Hii.shape, Hjj.shape, Hij.shape)
                print(islice, jslice)

            etotal += np.inner(e, e)

        # note that the system (H b) is obtained only from
        # relative constraints. H is not full rank.
        # we solve the problem by anchoring the position of
        # the the first vertex.
        # this can be expressed by adding the equation
        #  deltax(1:3,1) = 0
        # which is equivalent to the following

        H[0:3, 0:3] += np.eye(3)

        SH = sp.sparse.csr_matrix(H)
        deltax = sp.sparse.linalg.spsolve(SH, b)  # SH \ b

        # split the increments in nice 3x1 vectors and sum them up to the original matrix
        for vertex in g:
            i = vertex.index * 3
            vertex.coord += deltax[i : i + 3]
            # normalize the angles between -PI and PI
            vertex.coord[2] = smb.angdiff(vertex.coord[2])

        dt = time.time() - t0
        print(f"done in {dt*1e3:0.2f} msec.  Total cost {etotal:g}")

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
