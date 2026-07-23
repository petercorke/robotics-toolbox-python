"""
Facade for the _frne_c C extension (Newton-Euler inverse dynamics).

Tries to import from the compiled extension.  When unavailable (Pyodide
without a WASM build, or CI testing the Python path) ``init`` returns
``None``; DHRobot.rne() checks for that sentinel and dispatches to
``rne_python()`` instead.
"""

from __future__ import annotations

try:
    from roboticstoolbox._frne_c import (  # type: ignore[import]
        init as _c_init,
        frne as _c_frne,
        delete as _c_delete,
    )

    _C_AVAILABLE = True
except ImportError:
    _C_AVAILABLE = False


def init(njoints, mdh, L):
    """Create the C Robot struct handle; returns None when C extension is absent."""
    if _C_AVAILABLE:
        return _c_init(njoints, mdh, L)
    return None


def frne(robot, q, qd, qdd, gravity, base_rot, fext):
    """Run Newton-Euler via C; caller must check that robot handle is not None.

    ``gravity`` is given in the world frame; ``base_rot`` (the robot's
    ``self.base.R``, flattened row-major) is applied internally to rotate
    it into the root link frame before use -- callers no longer need to do
    this by hand. Returns ``(tau, wbase)`` where ``wbase`` is the wrench
    the base experiences, in the base/world frame.
    """
    return _c_frne(robot, q, qd, qdd, gravity, base_rot, fext)


def delete(robot):
    """Free the C Robot struct; no-op when handle is None (C absent)."""
    if robot is not None:
        _c_delete(robot)
