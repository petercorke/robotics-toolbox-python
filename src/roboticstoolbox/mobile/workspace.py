"""
The workspace of a mobile robot world.

A *workspace* is the rectangle of the plane that an experiment takes place in,
``[xmin, xmax, ymin, ymax]``.  It is used for the limits of plots, as the region
that random goals, landmarks and particles are drawn from, and as the extent of
a map.

Classes accept a workspace as a scalar, a 2-vector, a 4-vector, or any object
that has a ``workspace`` attribute (a vehicle, a map, an occupancy grid).
:func:`resolve_workspace` converts these to the 4-vector and implements the
order of precedence between them.
"""

from __future__ import annotations

from typing import Any, Protocol, Union

from spatialmath import base
from spatialmath.base.types import R4

from roboticstoolbox.tools.types import ArrayLike


class HasWorkspace(Protocol):
    """An object, such as a vehicle or a map, that has a workspace."""

    @property
    def workspace(self) -> R4 | None: ...


#: anything that can be turned into a workspace by :func:`resolve_workspace`
WorkspaceLike = Union[float, ArrayLike, HasWorkspace]


def resolve_workspace(*sources: WorkspaceLike | None | Any) -> R4 | None:
    """
    Resolve a workspace from several candidate sources

    :param sources: candidates in order of precedence, each a scalar, a
        2-vector, a 4-vector, an object with a ``workspace`` attribute, or None
    :raises ValueError: the first usable candidate is not a valid workspace
    :return: workspace bounds ``[xmin, xmax, ymin, ymax]``, or None if no
        candidate has a workspace

    The first candidate that is not None, and is not an object whose
    ``workspace`` is None, is converted by
    :func:`~spatialmath.base.graphics.expand_dims`:

    ==============  =======  =======
    candidate       x-range  y-range
    ==============  =======  =======
    A (scalar)      -A:A     -A:A
    [A, B]           A:B      A:B
    [A, B, C, D]     A:B      C:D
    ==============  =======  =======

    For example, a class that takes an explicit workspace but otherwise uses
    that of its map and then its vehicle::

        workspace = resolve_workspace(workspace, map, vehicle)

    :seealso: :func:`~spatialmath.base.graphics.expand_dims`
    """
    for source in sources:
        if source is None:
            continue
        if hasattr(source, "workspace"):
            # an object with a workspace, which it might not have set
            source = source.workspace
            if source is None:
                continue
        return base.expand_dims(source)
    return None
