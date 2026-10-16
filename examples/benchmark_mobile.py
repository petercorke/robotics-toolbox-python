#!/usr/bin/env python
"""
Benchmark RRT planning and pose-graph optimisation without plotting.

Run this script with the same dependencies on each revision being compared.
Construction and file loading are excluded from the timings. Each repetition
starts with a fresh planner or graph, with a fixed RRT random seed.

Use ``--output before.npz`` and ``--output after.npz`` to save the final tree,
path and optimised poses for numerical comparison alongside the timings.
The graph defaults to the bundled ``killian-small.toro`` dataset.

For the larger examples in issue #755, use ``--npoints 400`` and
``--graph data/killian.g2o.zip``. Timings depend on the machine and dependency
versions; compare results from the same environment.
"""

import argparse
import json
import platform
from importlib.metadata import version
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
from spatialmath import Polygon2

from roboticstoolbox import Bicycle, PolygonMap, PoseGraph, RRTPlanner


def make_rrt(npoints: int, seed: int) -> RRTPlanner:
    """
    Build the RRT documentation example

    :param npoints: number of vertices to add to the tree
    :param seed: random seed for repeatable sampling
    :returns: unplanned RRT instance with polygonal obstacles
    """
    workspace = PolygonMap(workspace=[0, 10])
    workspace.add([(5, 50), (5, 6), (6, 6), (6, 50)])
    workspace.add([(5, 4), (5, -50), (6, -50), (6, 4)])
    outline = Polygon2([(-1.5, 0.75), (-1.5, -0.75), (1.5, -0.75), (1.5, 0.75)])
    return RRTPlanner(
        map=workspace,
        vehicle=Bicycle(steer_max=0.4, L=2, polygon=outline),
        npoints=npoints,
        seed=seed,
        progress=False,
    )


def main() -> None:
    """Measure repeated runs and optionally save their final numerical results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--npoints", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--graph", default="data/killian-small.toro")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.repeats, args.npoints, args.iterations) < 1:
        parser.error("repeats, npoints and iterations must be positive")

    rrt_times: list[float] = []
    graph_times: list[float] = []
    for _ in range(args.repeats):
        rrt = make_rrt(args.npoints, args.seed)
        start = perf_counter()
        rrt.plan(goal=(8, 2, -np.pi / 2), showsamples=False, showvalid=False)
        rrt_times.append(perf_counter() - start)

        pg = PoseGraph(args.graph)
        start = perf_counter()
        pg.optimize(iterations=args.iterations)
        graph_times.append(perf_counter() - start)

    print(
        json.dumps(
            {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "dependencies": {
                    name: version(name)
                    for name in (
                        "numpy",
                        "scipy",
                        "spatialmath-python",
                        "pgraph-python",
                    )
                },
                "rrt": {
                    "npoints": args.npoints,
                    "seed": args.seed,
                    "seconds": rrt_times,
                    "median": median(rrt_times),
                },
                "posegraph": {
                    "file": args.graph,
                    "vertices": pg.graph.n,
                    "edges": pg.graph.ne,
                    "iterations": args.iterations,
                    "seconds": graph_times,
                    "median": median(graph_times),
                },
            },
            indent=2,
        )
    )
    if args.output is not None:
        path, _ = rrt.query((2, 8, -np.pi / 2))
        np.savez(
            args.output,
            tree=np.array([v.coord for v in rrt.g]),
            path=path,
            poses=np.array([v.coord for v in pg.graph]),
        )


if __name__ == "__main__":
    main()
