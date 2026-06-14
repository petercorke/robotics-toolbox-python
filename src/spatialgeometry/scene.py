#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np
from numpy.typing import NDArray


ArrayF64 = NDArray[np.float64]


@dataclass
class _Node:
    T: ArrayF64
    wT: ArrayF64
    wq: ArrayF64
    parent: Optional["_Node"] = None
    children: list["_Node"] = field(default_factory=list)


def _r2q_xyzs(R: ArrayF64) -> ArrayF64:
    q = np.zeros(4, dtype=np.float64)
    tr = float(R[0, 0] + R[1, 1] + R[2, 2])

    if tr > 0.0:
        S = np.sqrt(tr + 1.0) * 2.0
        q[3] = 0.25 * S
        q[0] = (R[2, 1] - R[1, 2]) / S
        q[1] = (R[0, 2] - R[2, 0]) / S
        q[2] = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        q[3] = (R[2, 1] - R[1, 2]) / S
        q[0] = 0.25 * S
        q[1] = (R[0, 1] + R[1, 0]) / S
        q[2] = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        q[3] = (R[0, 2] - R[2, 0]) / S
        q[0] = (R[0, 1] + R[1, 0]) / S
        q[1] = 0.25 * S
        q[2] = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        q[3] = (R[1, 0] - R[0, 1]) / S
        q[0] = (R[0, 2] + R[2, 0]) / S
        q[1] = (R[1, 2] + R[2, 1]) / S
        q[2] = 0.25 * S

    return q


def _propogate_T(node: _Node, parent_wT: Optional[ArrayF64]) -> None:
    if parent_wT is None:
        node.wT[:] = node.T
    else:
        node.wT[:] = parent_wT @ node.T

    node.wq[:] = _r2q_xyzs(node.wT[:3, :3])

    for child in node.children:
        _propogate_T(child, node.wT)


def node_init(
    n_children: int,
    T: ArrayF64,
    wT: ArrayF64,
    wq: ArrayF64,
    parent: Optional[_Node],
    children: Iterable[_Node],
):
    node = _Node(T=T, wT=wT, wq=wq)
    node.parent = parent
    node.children = list(children)

    if n_children != len(node.children):
        node.children = node.children[:n_children]

    return node


def node_update(
    node: _Node,
    n_children: int,
    parent: Optional[_Node],
    children: Iterable[_Node],
):
    node.parent = parent
    node.children = list(children)

    if n_children != len(node.children):
        node.children = node.children[:n_children]


def scene_graph_children(node: _Node):
    _propogate_T(node, None)


def scene_graph_tree(node: _Node):
    while node.parent is not None:
        node = node.parent

    _propogate_T(node, None)
