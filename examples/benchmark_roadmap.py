#!/usr/bin/env python
"""
Compare PRM and lattice queries with and without pgraph's spatial index.

Run from a checkout with ``python examples/benchmark_roadmap.py``. Roadmaps
are built by the real planners once; planning is excluded from query timings.
The default cases use 1000 PRM vertices and 12 lattice iterations. Increase
``--npoints`` and ``--iterations`` to compare larger maps. PRM has a central
obstacle and a fixed sampling seed; lattice queries start at the root so that
their goals are reachable. Both use a fixed batch of distinct destinations.

Each repetition measures both lookup modes, alternating their order. The tree
is invalidated through the public coordinate setter before each measurement.
Index construction and two endpoint lookups are timed separately from complete
queries (which include A*). All paths, and lattice costs, must agree. Timings
are diagnostic, not test assertions, and depend on the dependency versions.

With pgraph 1.0.0, lattice angles just below zero can wrap to exactly 2*pi,
which scipy rejects as outside the periodic box. The JSON reports this as
``index_error`` and leaves indexed timings unset; it does not time a fallback
scan as an indexed query. Five lattice iterations already reproduce it.
"""

import argparse
import json
import platform
from importlib.metadata import version
from statistics import median
from time import perf_counter
from typing import Any

import numpy as np

from roboticstoolbox import LatticePlanner, PRMPlanner
from roboticstoolbox.tools.types import NDArray


def measure(
    planner: PRMPlanner | LatticePlanner,
    pairs: list[tuple[NDArray, NDArray]],
    repeats: int,
) -> dict[str, Any]:
    """
    Measure index construction, endpoint lookups and complete path queries.

    :param planner: planner with a completed roadmap
    :param pairs: reachable start and goal pairs
    :param repeats: repetitions of each mode
    :returns: timings in seconds, graph size and result comparison
    """
    graph = planner.graph
    vertex = next(iter(graph))
    vertex.coord = vertex.coord.copy()
    reference = [planner.query(start, goal) for start, goal in pairs]
    samples = {
        mode: {key: [] for key in ("build", "endpoints", "query")}
        for mode in ("scan", "kdtree")
    }
    index_error = None

    for repetition in range(repeats):
        modes = ("scan", "kdtree") if repetition % 2 == 0 else ("kdtree", "scan")
        for mode in modes:
            # Assignment invalidates the index without changing graph geometry.
            vertex.coord = vertex.coord.copy()
            start_time = perf_counter()
            if mode == "kdtree":
                try:
                    graph.build_kdtree()
                except ValueError as error:
                    # Report an unsupported graph instead of timing a fallback
                    # scan and presenting it as an indexed query.
                    index_error = str(error)
                    continue
            samples[mode]["build"].append(perf_counter() - start_time)

            # Warm each lookup path before measuring steady-state queries.
            for start, goal in pairs:
                graph.closest(start)
                graph.closest(goal)
            start_time = perf_counter()
            for start, goal in pairs:
                graph.closest(start)
                graph.closest(goal)
            samples[mode]["endpoints"].append(
                (perf_counter() - start_time) / len(pairs)
            )

            start_time = perf_counter()
            results = [planner.query(start, goal) for start, goal in pairs]
            samples[mode]["query"].append((perf_counter() - start_time) / len(pairs))
            for expected, actual in zip(reference, results):
                if isinstance(planner, LatticePlanner):
                    np.testing.assert_array_equal(actual[0], expected[0])
                    np.testing.assert_equal(actual[1].cost, expected[1].cost)
                else:
                    np.testing.assert_array_equal(actual, expected)

    medians = {
        mode: {
            key: median(values) if values else None for key, values in timings.items()
        }
        for mode, timings in samples.items()
    }
    saving = (
        medians["scan"]["query"] - medians["kdtree"]["query"]
        if index_error is None
        else 0
    )
    return {
        "vertices": graph.n,
        "edges": graph.ne,
        "queries_per_repeat": len(pairs),
        "paths_equal": True if index_error is None else None,
        "index_error": index_error,
        "seconds": samples,
        "median_seconds": medians,
        "query_speedup": (
            medians["scan"]["query"] / medians["kdtree"]["query"]
            if index_error is None
            else None
        ),
        "break_even_queries": (
            int(np.ceil(medians["kdtree"]["build"] / saving)) if saving > 0 else None
        ),
    }


def main() -> None:
    """Build reproducible roadmaps and print the measurements as JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npoints", type=int, default=1000)
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--queries", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if min(args.npoints, args.iterations, args.queries, args.repeats) < 1:
        parser.error("npoints, iterations, queries and repeats must be positive")

    grid = np.zeros((100, 100))
    grid[40:60, 40:60] = 1
    prm = PRMPlanner(grid, seed=args.seed, progress=False, dist_thresh=15)
    prm.plan(npoints=args.npoints)
    random = np.random.default_rng(args.seed)
    goals = [vertex.coord.copy() for vertex in prm.graph]
    indices = random.choice(
        len(goals), size=min(args.queries, len(goals)), replace=False
    )
    prm_pairs = [(np.array([5.0, 5.0]), goals[i]) for i in indices]

    lattice = LatticePlanner(progress=False)
    lattice.plan(iterations=args.iterations)
    goals = [vertex.coord.copy() for vertex in lattice.graph]
    indices = random.choice(
        len(goals), size=min(args.queries, len(goals)), replace=False
    )
    lattice_pairs = [(np.zeros(3), goals[i]) for i in indices]
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
                "seed": args.seed,
                "prm": measure(prm, prm_pairs, args.repeats),
                "lattice": measure(lattice, lattice_pairs, args.repeats),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
