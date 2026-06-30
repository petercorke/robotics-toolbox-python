"""
Facade for the _fknm_c C++ extension (FK, Jacobian, Hessian, IK, ET ops).

Tries to import from the compiled extension.  When unavailable (Pyodide
without a WASM build, CI forcing the Python path, etc.) pure-Python
implementations are used for ETS evaluation functions.  IK C solvers and
visualisation helpers require the extension and raise at call time when absent.
"""

from __future__ import annotations

import numpy as np
from spatialmath import SE3
from spatialmath.base import getvector, getmatrix, verifymatrix, tr2jac

try:
    from roboticstoolbox._fknm_c import (  # type: ignore[import]
        ETS_init as _c_ETS_init,
        ETS_fkine as _c_ETS_fkine,
        ETS_jacob0 as _c_ETS_jacob0,
        ETS_jacobe as _c_ETS_jacobe,
        ETS_hessian0 as _c_ETS_hessian0,
        ETS_hessiane as _c_ETS_hessiane,
        IK_NR_c as _c_IK_NR_c,
        IK_GN_c as _c_IK_GN_c,
        IK_LM_c as _c_IK_LM_c,
        ET_T as _c_ET_T,
        ET_init as _c_ET_init,
        ET_update as _c_ET_update,
        Angle_Axis as _c_Angle_Axis,
        Robot_link_T as _c_Robot_link_T,
        r2q as _c_r2q,
    )

    _C_AVAILABLE = True
except ImportError:
    _C_AVAILABLE = False


def _is_symbolic(q) -> bool:
    """True when q contains SymPy symbols (object dtype array)."""
    if q is None:
        return False
    return np.asarray(q).dtype == object


# ---------------------------------------------------------------------------
# Pure-Python implementations (used when C is absent or q is symbolic)
# ---------------------------------------------------------------------------


def _python_fkine(data, q, base, tool, include_base):
    q = getmatrix(q, (None, None))
    l, _ = q.shape  # noqa: E741
    end = data[-1]

    if isinstance(tool, SE3):
        tool = np.array(tool.A)
    if isinstance(base, SE3):
        base = np.array(base.A)

    if base is None or np.array_equal(base, np.eye(3)):
        bases = None
    else:
        bases = base

    if tool is None or np.array_equal(tool, np.eye(3)):
        tools = None
    else:
        tools = tool

    if l > 1:
        T = np.zeros((l, 4, 4), dtype=object)
    else:
        T = np.zeros((4, 4), dtype=object)

    for k, qk in enumerate(q):
        link = end
        jindex = 0 if link.jindex is None and link.isjoint else link.jindex
        Tk = link.A(qk[jindex])

        if tools is not None:
            Tk = Tk @ tools

        for i in range(len(data) - 2, -1, -1):
            link = data[i]
            jindex = 0 if link.jindex is None and link.isjoint else link.jindex
            A = link.A(qk[jindex])
            if A is not None:
                Tk = A @ Tk

        if include_base and bases is not None:
            Tk = bases @ Tk

        if l > 1:
            T[k, :, :] = Tk
        else:
            T = Tk

    if isinstance(T, np.ndarray) and T.dtype == object:
        try:
            T = T.astype(float)
        except (TypeError, ValueError):
            pass  # symbolic elements — keep as object
    return T


def _python_jacob0(data, n, q, tool):
    if tool is None:
        tools = np.eye(4)
    elif isinstance(tool, SE3):
        tools = np.array(tool.A)
    else:
        tools = np.asarray(tool)

    q = getvector(q, None)
    T = _python_fkine(data, q, None, None, False) @ tools

    U = np.eye(4)
    j = 0
    J = np.zeros((6, n), dtype="object")
    zero = np.array([0, 0, 0])
    end = data[-1]

    for link in data:
        jindex = 0 if link.jindex is None and link.isjoint else link.jindex

        if link.isjoint:
            U = U @ link.A(q[jindex])

            if link == end:
                U = U @ tools

            Tu = SE3(U, check=False).inv().A @ T
            n_vec = U[:3, 0]
            o = U[:3, 1]
            a = U[:3, 2]
            x = Tu[0, 3]
            y = Tu[1, 3]
            z = Tu[2, 3]

            if link.axis == "Rz":
                J[:3, j] = (o * x) - (n_vec * y)
                J[3:, j] = a
            elif link.axis == "Ry":
                J[:3, j] = (n_vec * z) - (a * x)
                J[3:, j] = o
            elif link.axis == "Rx":
                J[:3, j] = (a * y) - (o * z)
                J[3:, j] = n_vec
            elif link.axis == "tx":
                J[:3, j] = n_vec
                J[3:, j] = zero
            elif link.axis == "ty":
                J[:3, j] = o
                J[3:, j] = zero
            elif link.axis == "tz":
                J[:3, j] = a
                J[3:, j] = zero

            if link.isflip:
                J[:, j] = -J[:, j]

            j += 1
        else:
            A = link.A()
            if A is not None:
                U = U @ A

    try:
        return J.astype(float)
    except (TypeError, ValueError):
        return J  # symbolic elements — keep as object


def _python_jacobe(data, n, q, tool):
    T = _python_fkine(data, q, None, tool, False)
    return tr2jac(T.T) @ _python_jacob0(data, n, q, tool)


def _python_hessian(J):
    def cross(a, b):
        return np.array(
            [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
        )

    n = J.shape[1]
    H = np.zeros((n, 6, n))
    for j in range(n):
        for i in range(j, n):
            H[j, :3, i] = cross(J[3:, j], J[:3, i])
            H[j, 3:, i] = cross(J[3:, j], J[3:, i])
            if i != j:
                H[i, :3, j] = H[j, :3, i]
    return H


# ---------------------------------------------------------------------------
# Public facade: ETS operations
# ---------------------------------------------------------------------------


def ETS_init(fknm_data, n, m):
    """Create the C++ ETS handle; returns None when C extension is absent."""
    if _C_AVAILABLE:
        return _c_ETS_init(fknm_data, n, m)
    return None


def ETS_fkine(fknm, q, base, tool, include_base, _data=None):
    if fknm is not None and not _is_symbolic(q):
        return _c_ETS_fkine(fknm, q, base, tool, include_base)
    return _python_fkine(_data, q, base, tool, include_base)


def ETS_jacob0(fknm, q, tool, _data=None, _n=None):
    if fknm is not None and not _is_symbolic(q):
        return _c_ETS_jacob0(fknm, q, tool)
    return _python_jacob0(_data, _n, q, tool)


def ETS_jacobe(fknm, q, tool, _data=None, _n=None):
    if fknm is not None and not _is_symbolic(q):
        return _c_ETS_jacobe(fknm, q, tool)
    return _python_jacobe(_data, _n, q, tool)


def ETS_hessian0(fknm, q, J0, tool, _data=None, _n=None):
    if fknm is not None and not _is_symbolic(q) and not _is_symbolic(J0):
        return _c_ETS_hessian0(fknm, q, J0, tool)
    if J0 is None:
        if q is None:
            raise ValueError("Either J0 or q must be provided")
        q = getvector(q, None)
        J0 = _python_jacob0(_data, _n, q, tool)
    else:
        verifymatrix(J0, (6, _n))
    return _python_hessian(J0)


def ETS_hessiane(fknm, q, Je, tool, _data=None, _n=None):
    if fknm is not None and not _is_symbolic(q) and not _is_symbolic(Je):
        return _c_ETS_hessiane(fknm, q, Je, tool)
    if Je is None:
        if q is None:
            raise ValueError("Either Je or q must be provided")
        q = getvector(q, None)
        Je = _python_jacobe(_data, _n, q, tool)
    else:
        verifymatrix(Je, (6, _n))
    return _python_hessian(Je)


# ---------------------------------------------------------------------------
# Public facade: IK (C only — Python IK lives at a higher level)
# ---------------------------------------------------------------------------


def IK_NR_c(fknm, *args, **kwargs):
    if fknm is None:
        raise RuntimeError(
            "ik_NR requires the _fknm_c C extension; use ikine_NR for the Python solver"
        )
    return _c_IK_NR_c(fknm, *args, **kwargs)


def IK_GN_c(fknm, *args, **kwargs):
    if fknm is None:
        raise RuntimeError(
            "ik_GN requires the _fknm_c C extension; use ikine_GN for the Python solver"
        )
    return _c_IK_GN_c(fknm, *args, **kwargs)


def IK_LM_c(fknm, *args, **kwargs):
    if fknm is None:
        raise RuntimeError(
            "ik_LM requires the _fknm_c C extension; use ikine_LM for the Python solver"
        )
    return _c_IK_LM_c(fknm, *args, **kwargs)


# ---------------------------------------------------------------------------
# Public facade: ET operations
# ---------------------------------------------------------------------------


def ET_init(*args):
    """Create a C ET handle; returns None when C extension is absent."""
    if _C_AVAILABLE:
        return _c_ET_init(*args)
    return None


def ET_T(et_handle, q):
    """Evaluate an ET at q via C.  Raises TypeError when C absent so ET.A() falls back."""
    if et_handle is None:
        raise TypeError("_fknm_c not available")
    return _c_ET_T(et_handle, q)


def ET_update(et_handle, *args):
    """Update a C ET handle in-place; no-op when handle is None (C absent)."""
    if et_handle is not None:
        _c_ET_update(et_handle, *args)


# ---------------------------------------------------------------------------
# Public facade: misc helpers
# ---------------------------------------------------------------------------


def Angle_Axis(T, Td):
    """C angle-axis error; raises when C absent (p_servo.angle_axis has a Python fallback)."""
    if not _C_AVAILABLE:
        raise RuntimeError("_fknm_c not available")
    return _c_Angle_Axis(T, Td)


def Robot_link_T(link_ets, link_scene_nodes, q0, q):
    """Update per-link scene-node transforms via C; no-op when C absent (visualisation only)."""
    if _C_AVAILABLE:
        _c_Robot_link_T(link_ets, link_scene_nodes, q0, q)


def r2q(R):
    """Convert rotation matrix to quaternion via C; raises when C absent."""
    if not _C_AVAILABLE:
        raise RuntimeError("_fknm_c not available")
    return _c_r2q(R)
