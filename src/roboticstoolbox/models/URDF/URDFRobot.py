"""
URDF/xacro loading infrastructure for robot models.

Provides:
  - URDF_file(path)   — parse a URDF or xacro file, return (elinks, name)
  - URDF_read(path)   — same but returns (elinks, name, None, None) for
                        backward-compat with complex models that manipulate
                        the link list before calling Robot.__init__
  - URDFRobot         — base class for simple URDF-loaded robot models
"""

from pathlib import Path
import importlib
import warnings
from typing import TextIO

import numpy as np
from spatialmath import SE3
from spatialmath.base import unitvec_norm, angvec2r, tr2rpy

from xacrodoc import XacroDoc, packages

from roboticstoolbox.tools.urdf import URDF
from roboticstoolbox.robot.Link import Link
from roboticstoolbox.robot.ET import ET
from roboticstoolbox.robot.ETS import ETS
from roboticstoolbox.robot.Robot import Robot


def _register_rd_packages(urdf_path: Path) -> None:
    """Register all package:// names referenced in a URDF with the xacrodoc cache.

    Scans the URDF text for every ``package://NAME/`` reference, then walks up
    the directory tree from the file to find and register the matching ancestor
    directory for each name.  This handles arbitrarily nested cache layouts
    (e.g. ``example-robot-data/robots/panda_description/urdf/panda.urdf``).
    """
    import re

    pkg_names = set(re.findall(r"package://([^/]+)/", urdf_path.read_text()))
    for parent in urdf_path.parents:
        if parent.name in pkg_names:
            packages.update_package_cache({parent.name: str(parent)})
            pkg_names.discard(parent.name)
        if not pkg_names:
            break


_RD_URL = "https://github.com/robot-descriptions/robot_descriptions.py"


def _rd_link() -> str:
    """Render "robot_descriptions" as a clickable terminal hyperlink (OSC 8).

    Terminals that don't understand OSC 8 just display the plain text, so
    this degrades safely everywhere.
    """
    return f"\033]8;;{_RD_URL}\033\\robot_descriptions\033]8;;\033\\"


def _find_rd_rename(robot_name: str, tried: list[str]) -> str | None:
    """Look for another robot_descriptions entry for the same robot under a
    name we didn't try, e.g. an alternate/newer naming convention.

    Returns the matching name, or None if robot_descriptions has nothing
    registered for ``robot_name`` at all.
    """
    import robot_descriptions

    prefix = f"{robot_name}_"
    for key in robot_descriptions.DESCRIPTIONS:
        if key not in tried and key.startswith(prefix):
            return key
    return None


def _load_rd_module(robot_name: str):
    """Import the robot_descriptions submodule for ``robot_name``, trying the
    current name first and falling back to older/newer naming schemes.

    robot_descriptions has renamed some entries over time (e.g. ``ur5_description``
    -> ``ur5_official_description``); trying alternates here means callers and
    model classes don't need to track that migration themselves.
    """
    if "_mj_" in robot_name:
        raise ValueError(
            f"Toolbox uses {_rd_link()} to provide URDF robot models. The "
            f'requested model named "{robot_name}" is a MuJoCo model not a '
            "URDF model."
        )

    candidates = [f"{robot_name}_description", f"{robot_name}_official_description"]
    last_error: ImportError | None = None
    for candidate in candidates:
        try:
            with warnings.catch_warnings():
                # Deprecation notices name "robot_descriptions" directly, which
                # is an implementation detail users of roboticstoolbox never
                # opted into seeing.
                warnings.simplefilter("ignore", FutureWarning)
                return importlib.import_module(f"robot_descriptions.{candidate}")
        except ImportError as e:
            last_error = e
            continue

    renamed_to = _find_rd_rename(robot_name, candidates)
    if renamed_to is not None:
        raise ValueError(
            f"Toolbox uses {_rd_link()} to provide URDF robot models. The "
            f'requested model named "{robot_name}" is now named "{renamed_to}".'
        ) from last_error

    raise ValueError(
        f"Toolbox uses {_rd_link()} to provide URDF robot models. The "
        f'requested model named "{robot_name}" can not be found.'
    ) from last_error


def _load_urdf_from_RD(robot_name: str) -> "tuple[Path, dict | None]":
    """Fetch the URDF/xacro path from robot_descriptions and register its packages.

    robot_descriptions models expose different path attributes depending on their
    source format:
      - URDF_PATH  — a precompiled, ready-to-parse URDF file (e.g. panda, pr2, gen2)
      - XACRO_PATH — a xacro source that must be processed by xacrodoc
                     (e.g. j2n6s200, ur5, kinova family)
    Both are passed to XacroDoc.from_file(), which handles either format.

    Some xacro-based models also expose ``XACRO_ARGS`` — substitution
    values required to compile the file at all (e.g. Kinova Gen3's
    ``{"dof": "7"}``, which selects the 6dof/7dof arm variant; several UR
    and xArm variants have the same pattern). Returned alongside the path
    so the caller can forward them to xacrodoc as ``subargs`` — without
    them, xacro fails on an unresolved ``$(arg ...)``/property reference.
    """
    module = _load_rd_module(robot_name)

    if hasattr(module, "URDF_PATH"):
        urdf_path = Path(module.URDF_PATH)
    elif hasattr(module, "XACRO_PATH"):
        urdf_path = Path(module.XACRO_PATH)
    else:
        raise ValueError(
            f"Robot model '{robot_name}' in robot_descriptions has neither "
            "URDF_PATH nor XACRO_PATH."
        )
    _register_rd_packages(urdf_path)
    return urdf_path, getattr(module, "XACRO_ARGS", None)


def _parse_urdf(urdf_str: str):
    """Parse a URDF string into (elinks, name)."""
    urdf = URDF.loadstr(urdf_str, None)

    elinks = []
    elinkdict = {}

    for link in urdf._links:
        elink = Link(
            name=link.name,
            m=link.inertial.mass,
            r=link.inertial.origin[:3, 3] if link.inertial.origin is not None else None,
            I=link.inertial.inertia,
        )
        elinks.append(elink)
        elinkdict[link.name] = elink

        try:
            elink.geometry = [v.geometry.ob for v in link.visuals]
        except AttributeError:
            pass

        if link.collisions:
            shapes = []
            for col in link.collisions:
                try:
                    ob = col.geometry.ob
                except AttributeError:
                    ob = None
                if ob is not None:
                    shapes.append(ob)
            if not shapes:
                raise RuntimeError(
                    f"Collision geometry for link '{link.name}' failed to load. "
                    "Ensure all package:// URIs are registered in the xacrodoc cache."
                )
            elink.collision = shapes

    for joint in urdf._joints:
        childlink = elinkdict[joint.child]
        parentlink = elinkdict[joint.parent]

        childlink._parent = parentlink
        childlink._joint_name = joint.name

        trans = SE3(joint.origin).t
        rot = joint.rpy

        if np.count_nonzero(joint.axis) < 2:
            ets = ET.SE3(SE3(trans) * SE3.RPY(rot))
        else:
            v = joint.axis
            u, n = unitvec_norm(v)
            R = angvec2r(n, u)
            R_total = SE3.RPY(joint.rpy) * R
            rpy = tr2rpy(R_total)
            ets = ET.SE3(SE3(trans) * SE3.RPY(rpy))
            joint.axis = [0, 0, 1]

        var = None
        if joint.joint_type in ("revolute", "continuous"):
            if joint.axis[0] == 1:
                var = ET.Rx()
            elif joint.axis[0] == -1:
                var = ET.Rx(flip=True)
            elif joint.axis[1] == 1:
                var = ET.Ry()
            elif joint.axis[1] == -1:
                var = ET.Ry(flip=True)
            elif joint.axis[2] == 1:
                var = ET.Rz()
            elif joint.axis[2] == -1:
                var = ET.Rz(flip=True)
        elif joint.joint_type == "prismatic":
            if joint.axis[0] == 1:
                var = ET.tx()
            elif joint.axis[0] == -1:
                var = ET.tx(flip=True)
            elif joint.axis[1] == 1:
                var = ET.ty()
            elif joint.axis[1] == -1:
                var = ET.ty(flip=True)
            elif joint.axis[2] == 1:
                var = ET.tz()
            elif joint.axis[2] == -1:
                var = ET.tz(flip=True)

        if var is not None:
            ets = ets * var

        if isinstance(ets, ET):
            ets = ETS(ets)

        childlink.ets = ets

        try:
            if childlink.isjoint:
                if joint.limit.lower is not None and joint.limit.upper is not None:
                    childlink.qlim = [joint.limit.lower, joint.limit.upper]
                childlink.qdlim = joint.limit.velocity
                childlink.tlim = joint.limit.effort
        except AttributeError:
            pass

        try:
            if joint.dynamics.friction is not None:
                childlink.B = joint.dynamics.friction
        except AttributeError:
            pass

        for t in urdf.transmissions:
            if t.name == joint.name:
                childlink.G = t.actuators[0].mechanicalReduction

    return elinks, urdf.name


def URDF_file(file: "str | Path | TextIO", model: "str | None" = None) -> tuple:
    """Parse a URDF or xacro file, return (elinks, name).

    ``file`` may be:
    - an absolute or relative path (str or Path) to a .urdf or .xacro file
    - a bare name with no suffix, looked up via robot_descriptions
    - a file-like object whose .read() gives the URDF XML
    """
    import rtbdata

    xacro_root = Path(rtbdata.__file__).parent / "xacro"
    pkg_map = {d.name: str(d) for d in xacro_root.iterdir() if d.is_dir()}
    packages.update_package_cache(pkg_map)

    xacro_args = None
    if isinstance(file, str):
        file = Path(file)
        if file.suffix not in (".urdf", ".xacro"):
            file, xacro_args = _load_urdf_from_RD(str(file))

    resolved_path = None
    if isinstance(file, Path):
        if not file.is_absolute():
            file = xacro_root / file
        resolved_path = file
        doc = XacroDoc.from_file(file, subargs=xacro_args)
    else:
        doc = XacroDoc.from_string(file.read())

    elinks, name = _parse_urdf(doc.to_urdf_string())
    return elinks, name, resolved_path


def URDF_read(urdf_path: "str | Path") -> tuple:
    """Load a URDF/xacro file, return (elinks, name, filepath).

    ``filepath`` is the resolved filesystem path that was loaded, or None if
    the source was a file-like object.
    """
    return URDF_file(urdf_path)


class URDFRobot(Robot):
    """Base class for robot models loaded from a URDF or xacro file.

    Subclasses pass the path and manufacturer directly to ``super().__init__()``::

        class UR5(URDFRobot):
            def __init__(self):
                super().__init__(
                    "ur5",
                    manufacturer="Universal Robotics",
                    gripper_link_index=7,
                )
                self.qz = np.zeros(6)
                ...
    """

    def __init__(
        self,
        urdf_path: "str | Path",
        manufacturer: str = "",
        gripper_link_index: "int | None" = None,
        **kwargs,
    ):
        elinks, name, filepath = URDF_file(urdf_path)
        if gripper_link_index is not None:
            kwargs["gripper_links"] = elinks[gripper_link_index]
        super().__init__(elinks, name=name, manufacturer=manufacturer, **kwargs)
        self._urdf_filepath = str(filepath) if filepath is not None else ""
