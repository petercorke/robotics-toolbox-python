#!/usr/bin/env python
"""
@author (Original) Matthew Matl, Github: mmatl
@author (Adapted by) Jesse Haviland
"""

from __future__ import annotations

import copy
import os
import xml.etree.ElementTree as ETT
from io import BytesIO
from pathlib import PurePosixPath
from typing import Any

import numpy as np
import spatialgeometry as gm

from roboticstoolbox.tools.data import rtb_path_to_datafile

from .utils import configure_origin, parse_origin

# Global variable for the base path of the robot meshes
_base_path = None


class URDFError(ValueError):
    """
    A URDF or xacro file could not be loaded.

    Raised by :meth:`URDF.loadstr` and by the model loaders in
    :mod:`roboticstoolbox.models.URDF` instead of the bare ``ValueError``,
    ``KeyError`` or XML parser errors they used to leak. It subclasses
    :class:`ValueError`, so existing ``except ValueError`` handlers keep
    working. ``str(err)`` is a multi-line report; the fields below let code
    inspect the details.

    :param msg: what went wrong
    :param stage: ``"xml"`` (the text is not well-formed XML), ``"xacro"``
        (xacro could not expand the file) or ``"urdf"`` (the XML is fine but
        the robot description is not)
    :param source: the file, or a description of the input, being loaded
    :param line: 1-based line of the failure, when known
    :param column: column of the failure, when known
    :param element: the XML element being parsed when the failure happened;
        enclosing elements are added as the error propagates outward, see
        :meth:`add_context`
    :param location: xacro's file and macro stack at the time of the failure
    :param expanded_file: path of a saved copy of the xacro-expanded URDF,
        which is what ``line`` refers to when it is set
    """

    def __init__(
        self,
        msg: str,
        *,
        stage: str = "urdf",
        source: str | os.PathLike | None = None,
        line: int | None = None,
        column: int | None = None,
        element: str | None = None,
        location: list[str] | None = None,
        expanded_file: str | os.PathLike | None = None,
    ) -> None:
        super().__init__(msg)
        self.msg = msg
        self.stage = stage
        self.source = source
        self.line = line
        self.column = column
        self.elements: list[str] = [element] if element else []
        self.location: list[str] = list(location) if location else []
        self.expanded_file = expanded_file

    @property
    def element(self) -> str | None:
        """The innermost XML element being parsed when the failure happened."""
        return self.elements[0] if self.elements else None

    def add_context(self, element: str) -> None:
        """Record an enclosing element, called as the error propagates outward."""
        if not self.elements or self.elements[-1] != element:
            self.elements.append(element)

    def __str__(self) -> str:
        lines = [self.msg]
        if self.elements:
            lines.append("  in " + " inside ".join(self.elements))
        if self.line is not None:
            where = f"line {self.line}"
            if self.column is not None:
                where += f", column {self.column}"
            if self.expanded_file is not None:
                lines.append(f"  at {where} of the expanded URDF")
            elif self.source is not None:
                lines.append(f"  at {where} of {self.source}")
            else:
                lines.append(f"  at {where}")
        lines.extend(f"  {loc}" for loc in self.location)
        if self.expanded_file is not None:
            lines.append(f"  expanded URDF written to {self.expanded_file}")
        return "\n".join(lines)


def _describe(node: Any) -> str:
    """Short description of an XML element for error messages."""
    name = node.attrib.get("name")
    if name is None:
        return f"<{node.tag}>"
    return f'<{node.tag} name="{name}">'


def _parse_child(cls: Any, node: Any, path: Any) -> Any:
    """Call ``cls._from_xml(node, path)``, attaching element context to any error."""
    try:
        return cls._from_xml(node, path)
    except URDFError as e:
        e.add_context(_describe(node))
        raise
    except KeyError as e:
        raise URDFError(
            f"missing attribute {e.args[0]!r}", stage="urdf", element=_describe(node)
        ) from e
    except (ValueError, TypeError, AttributeError) as e:
        raise URDFError(
            str(e) or type(e).__name__, stage="urdf", element=_describe(node)
        ) from e


class URDFType():
    """Abstract base class for all URDF types.
    This has useful class methods for automatic parsing/unparsing
    of XML trees.
    There are three overridable class variables:
    - ``_ATTRIBS`` - This is a dictionary mapping attribute names to a tuple,
      ``(type, required)`` where ``type`` is the Python type for the
      attribute and ``required`` is a boolean stating whether the attribute
      is required to be present.
    - ``_ELEMENTS`` - This is a dictionary mapping element names to a tuple,
      ``(type, required, multiple)`` where ``type`` is the Python type for the
      element, ``required`` is a boolean stating whether the element
      is required to be present, and ``multiple`` is a boolean indicating
      whether multiple elements of this type could be present.
      Elements are child nodes in the XML tree, and their type must be a
      subclass of :class:`.URDFType`.
    - ``_TAG`` - This is a string that represents the XML tag for the node
      containing this type of object.
    """

    _ATTRIBS: dict[str, Any] = {}  # Map from attrib name to (type, required)
    _ELEMENTS: dict[str, Any] = {}  # Map from element name to (type, required, multiple)
    _TAG = ""  # XML tag for this element

    def __init__(self) -> None:  # pragma nocover
        pass

    @classmethod
    def _parse_attrib(cls, val_type: Any, val: Any) -> Any:
        """Parse an XML attribute into a python value.
        Parameters
        ----------
        val_type : :class:`type`
            The type of value to create.
        val : :class:`object`
            The value to parse.
        Returns
        -------
        val : :class:`object`
            The parsed attribute.
        """
        if val_type == np.ndarray:
            val = np.fromstring(val, sep=" ")
        else:
            val = val_type(val)
        return val

    @classmethod
    def _parse_simple_attribs(cls, node: Any) -> Any:
        """Parse all attributes in the _ATTRIBS array for this class.
        Parameters
        ----------
        node : :class:`lxml.etree.Element`
            The node to parse attributes for.
        Returns
        -------
        kwargs : dict
            Map from attribute name to value. If the attribute is not
            required and is not present, that attribute's name will map to
            ``None``.
        """
        kwargs = {}
        for a in cls._ATTRIBS:
            t, r = cls._ATTRIBS[a]  # t = type, r = required (bool)
            if r:
                if a not in node.attrib:
                    raise URDFError(
                        f'missing required attribute "{a}"',
                        stage="urdf",
                        element=_describe(node),
                    )
                try:
                    v = cls._parse_attrib(t, node.attrib[a])
                except (ValueError, TypeError) as e:
                    raise URDFError(
                        f'invalid value {node.attrib[a]!r} for attribute "{a}": {e}',
                        stage="urdf",
                        element=_describe(node),
                    ) from e
            else:
                v = None
                if a in node.attrib:
                    try:
                        v = cls._parse_attrib(t, node.attrib[a])
                    except (ValueError, TypeError) as e:
                        raise URDFError(
                            f'invalid value {node.attrib[a]!r} for attribute "{a}": '
                            f"{e}",
                            stage="urdf",
                            element=_describe(node),
                        ) from e
            kwargs[a] = v
        return kwargs

    @classmethod
    def _parse_simple_elements(cls, node: Any, path: Any) -> Any:
        """Parse all elements in the _ELEMENTS array from the children of
        this node.
        Parameters
        ----------
        node : :class:`lxml.etree.Element`
            The node to parse children for.
        path : str
            The string path where the XML file is located (used for resolving
            the location of mesh or image files).
        Returns
        -------
        kwargs : dict
            Map from element names to the :class:`URDFType` subclass (or list,
            if ``multiple`` was set) created for that element.
        """
        kwargs = {}
        for a in cls._ELEMENTS:
            t, r, m = cls._ELEMENTS[a]
            if not m:
                v = node.find(t._TAG)
                if v is None and r:
                    raise URDFError(
                        f"missing required <{t._TAG}> element",
                        stage="urdf",
                        element=_describe(node),
                    )
                if v is not None:
                    v = _parse_child(t, v, path)
            else:
                vs = node.findall(t._TAG)
                if len(vs) == 0 and r:
                    raise URDFError(
                        f"missing required <{t._TAG}> element(s)",
                        stage="urdf",
                        element=_describe(node),
                    )
                v = [_parse_child(t, n, path) for n in vs]
            kwargs[a] = v
        return kwargs

    @classmethod
    def _parse(cls, node: Any, path: Any) -> Any:
        """Parse all elements and attributes in the _ELEMENTS and _ATTRIBS
        arrays for a node.
        Parameters
        ----------
        node : :class:`lxml.etree.Element`
            The node to parse.
        path : str
            The string path where the XML file is located (used for resolving
            the location of mesh or image files).
        Returns
        -------
        kwargs : dict
            Map from names to Python classes created from the attributes
            and elements in the class arrays.
        """
        kwargs = cls._parse_simple_attribs(node)
        kwargs.update(cls._parse_simple_elements(node, path))
        return kwargs

    @classmethod
    def _from_xml(cls, node: Any, path: Any) -> Any:
        """Create an instance of this class from an XML node.
        Parameters
        ----------
        node : :class:`lxml.etree.Element`
            The node to parse.
        path : str
            The string path where the XML file is located (used for resolving
            the location of mesh or image files).
        Returns
        -------
        obj : :class:`URDFType`
            An instance of this class parsed from the node.
        """
        return cls(**cls._parse(node, path))


###############################################################################
# Link types
###############################################################################


class Box(URDFType):
    """A rectangular prism whose center is at the local origin.
    Parameters
    ----------
    size : (3,) float
        The length, width, and height of the box in meters.
    """

    _ATTRIBS = {"size": (np.ndarray, True)}
    _TAG = "box"

    def __init__(self, size: Any) -> None:
        self.size = size

    def __repr__(self) -> str:  # pragma nocover
        return f"Box(size={self.size})"
    
    @property
    def size(self) -> Any:
        """(3,) float : The length, width, and height of the box in meters."""
        return self._size

    @size.setter
    def size(self, value: Any) -> None:
        self._size = np.asanyarray(value).astype(np.float64)


class Cylinder(URDFType):
    """A cylinder whose center is at the local origin.
    Parameters
    ----------
    radius : float
        The radius of the cylinder in meters.
    length : float
        The length of the cylinder in meters.
    """

    _ATTRIBS = {
        "radius": (float, True),
        "length": (float, True),
    }
    _TAG = "cylinder"

    def __init__(self, radius: Any, length: Any) -> None:
        self.radius = radius
        self.length = length

    def __repr__(self) -> str:  # pragma nocover
        return f"Cylinder(radius={self.radius}, length={self.length})"

    @property
    def radius(self) -> Any:
        """float : The radius of the cylinder in meters."""
        return self._radius

    @radius.setter
    def radius(self, value: Any) -> None:
        self._radius = float(value)

    @property
    def length(self) -> Any:
        """float : The length of the cylinder in meters."""
        return self._length

    @length.setter
    def length(self, value: Any) -> None:
        self._length = float(value)


class Sphere(URDFType):
    """A sphere whose center is at the local origin.
    Parameters
    ----------
    radius : float
        The radius of the sphere in meters.
    """

    _ATTRIBS = {
        "radius": (float, True),
    }
    _TAG = "sphere"

    def __init__(self, radius: Any) -> None:
        self.radius = radius

    def __repr__(self) -> str:  # pragma nocover
        return f"Sphere(radius={self.radius})"

    @property
    def radius(self) -> Any:
        """float : The radius of the sphere in meters."""
        return self._radius

    @radius.setter
    def radius(self, value: Any) -> None:
        self._radius = float(value)


class Mesh(URDFType):
    """A triangular mesh object.
    Parameters
    ----------
    filename : str
        The path to the mesh that contains this object. This can be
        relative to the top-level URDF or an absolute path.
    scale : (3,) float, optional
        The scaling value for the mesh along the XYZ axes.
        If ``None``, assumes no scale is applied.
    """

    _ATTRIBS = {"filename": (str, True), "scale": (np.ndarray, False)}
    _TAG = "mesh"

    def __init__(self, filename: Any, scale: Any = None) -> None:
        self.filename = filename
        self.scale = scale

    def __repr__(self) -> str:  # pragma nocover
        return f"Mesh(filename={self.filename}, scale={self.scale})"

    @property
    def filename(self) -> Any:
        """str : The path to the mesh file for this object."""
        return self._filename

    @filename.setter
    def filename(self, value: Any) -> None:

        global _base_path

        if value.startswith("package://"):
            value = value.replace("package://", "")
        elif value.startswith("file://"):
            value = value.replace("file://", "")

        if _base_path is None:
            value = rtb_path_to_datafile("xacro", value)
        else:
            value = _base_path / PurePosixPath(value)

        self._filename = str(value)

    @property
    def scale(self) -> Any:
        """(3,) float : A scaling for the mesh along its local XYZ axes."""
        return self._scale

    @scale.setter
    def scale(self, value: Any) -> None:
        if value is not None:
            value = np.asanyarray(value).astype(np.float64)
        self._scale = value

    @classmethod
    def _from_xml(cls, node: Any, path: Any) -> Any:
        kwargs = cls._parse(node, path)
        return Mesh(**kwargs)


class Material(URDFType):
    """A material for some geometry.
    Parameters
    ----------
    name : str
        The name of the material.
    color : (4,) float, optional
        The RGBA color of the material in the range [0,1].
    texture : :class:`.Texture`, optional
        A texture for the material.
    """
    def __repr__(self) -> str:  # pragma nocover
        return f"Material(name={self.name}, color={self.color}, texture={self.texture})"

    _ATTRIBS = {"name": (str, True)}
    _ELEMENTS: dict[str, Any] = {}
    _TAG = "material"

    def __init__(self, name: Any, color: Any = None, texture: Any = None) -> None:
        # if color is None:
        #     color = name

        self.name = name
        self.color = color
        self.texture = texture

    @classmethod
    def _from_xml(cls, node: Any, path: Any) -> Any:  # pragma nocover
        kwargs = cls._parse(node, path)

        # Extract the color -- it's weirdly an attribute of a subelement
        color = node.find("color")
        if color is not None:
            color = np.fromstring(color.attrib["rgba"], sep=" ", dtype=np.float64)
        kwargs["color"] = color

        return Material(**kwargs)


class Geometry(URDFType):
    """A wrapper for all geometry types.
    Only one of the following values can be set, all others should be set
    to ``None``.
    Parameters
    ----------
    box : :class:`.Box`, optional
        Box geometry.
    cylinder : :class:`.Cylinder`
        Cylindrical geometry.
    sphere : :class:`.Sphere`
        Spherical geometry.
    mesh : :class:`.Mesh`
        Mesh geometry.
    """

    _ELEMENTS = {
        "box": (Box, False, False),
        "cylinder": (Cylinder, False, False),
        "sphere": (Sphere, False, False),
        "mesh": (Mesh, False, False),
    }
    _TAG = "geometry"

    def __init__(
        self, box: Any = None, cylinder: Any = None, sphere: Any = None, mesh: Any = None
    ) -> None:
        if (
            box is None and cylinder is None and sphere is None and mesh is None
        ):  # pragma nocover
            raise ValueError("At least one geometry element must be set")

        if box is not None:
            self.box = box
            self.ob = gm.Cuboid(box.size)

        if cylinder is not None:
            self.cylinder = cylinder
            self.ob = gm.Cylinder(cylinder.radius, cylinder.length)

        if sphere is not None:
            self.sphere = sphere
            self.ob = gm.Sphere(sphere.radius)

        if mesh is not None:
            self.mesh = mesh
            self.ob = gm.Mesh(mesh.filename, scale=mesh.scale)

    def __repr__(self) -> str:  # pragma nocover
        return f"Geometry(box={self.box}, cylinder={self.cylinder}, sphere={self.sphere}, mesh={self.mesh})"
    
    @property
    def box(self) -> Any:
        """:class:`.Box` : Box geometry."""
        return self._box

    @box.setter
    def box(self, value: Any) -> None:
        if value is not None and not isinstance(value, Box):  # pragma nocover
            raise TypeError("Expected Box type")
        self._box = value

    @property
    def cylinder(self) -> Any:
        """:class:`.Cylinder` : Cylinder geometry."""
        return self._cylinder

    @cylinder.setter
    def cylinder(self, value: Any) -> None:
        if value is not None and not isinstance(value, Cylinder):
            raise TypeError("Expected Cylinder type")  # pragma nocover
        self._cylinder = value

    @property
    def sphere(self) -> Any:
        """:class:`.Sphere` : Spherical geometry."""
        return self._sphere

    @sphere.setter
    def sphere(self, value: Any) -> None:
        if value is not None and not isinstance(value, Sphere):
            raise TypeError("Expected Sphere type")  # pragma nocover
        self._sphere = value

    @property
    def mesh(self) -> Any:
        """:class:`.Mesh` : Mesh geometry."""
        return self._mesh

    @mesh.setter
    def mesh(self, value: Any) -> None:
        if value is not None and not isinstance(value, Mesh):
            raise TypeError("Expected Mesh type")  # pragma nocover
        self._mesh = value

    @property
    def geometry(self) -> Any:  # pragma nocover
        """:class:`.Box`, :class:`.Cylinder`, :class:`.Sphere`, or
        :class:`.Mesh` : The valid geometry element.
        """
        if self.box is not None:
            return self.box
        if self.cylinder is not None:
            return self.cylinder
        if self.sphere is not None:
            return self.sphere
        if self.mesh is not None:
            return self.mesh
        return None


class Collision(URDFType):
    """Collision properties of a link.
    Parameters
    ----------
    geometry : :class:`.Geometry`
        The geometry of the element
    name : str, optional
        The name of the collision geometry.
    origin : (4,4) float, optional
        The pose of the collision element relative to the link frame.
        Defaults to identity.
    """

    _ATTRIBS = {"name": (str, False)}
    _ELEMENTS = {
        "geometry": (Geometry, False, False),
    }
    _TAG = "collision"

    def __init__(self, name: Any, origin: Any, geometry: Any) -> None:
        self.geometry = geometry
        self.name = name
        self.origin = origin
        self.geometry.ob.T = origin

    def __repr__(self) -> str:  # pragma nocover
        return f"Collision(name={self.name}, origin={self.origin}, geometry={self.geometry})"
    
    @property
    def geometry(self) -> Any:
        """:class:`.Geometry` : The geometry of this element."""
        return self._geometry

    @geometry.setter
    def geometry(self, value: Any) -> None:
        if not isinstance(value, Geometry):  # pragma nocover
            raise TypeError("Must set geometry with Geometry object")
        self._geometry = value

    @property
    def name(self) -> Any:
        """str : The name of this collision element."""
        return self._name

    @name.setter
    def name(self, value: Any) -> None:
        if value is not None:
            value = str(value)
        self._name = value

    @property
    def origin(self) -> Any:
        """(4,4) float : The pose of this element relative to the link frame."""
        return self._origin

    @origin.setter
    def origin(self, value: Any) -> None:
        self._origin = configure_origin(value)

    @classmethod
    def _from_xml(cls, node: Any, path: Any) -> Any:
        kwargs = cls._parse(node, path)
        kwargs["origin"], _ = parse_origin(node)
        return Collision(**kwargs)


class Visual(URDFType):
    """Visual properties of a link.
    Parameters
    ----------
    geometry : :class:`.Geometry`
        The geometry of the element
    name : str, optional
        The name of the visual geometry.
    origin : (4,4) float, optional
        The pose of the visual element relative to the link frame.
        Defaults to identity.
    material : :class:`.Material`, optional
        The material of the element.
    """

    _ATTRIBS = {"name": (str, False)}
    _ELEMENTS = {
        "geometry": (Geometry, False, False),
        "material": (Material, False, False),
    }
    _TAG = "visual"

    def __init__(
        self, geometry: Any, name: Any = None, origin: Any = None, material: Any = None
    ) -> None:
        self.geometry = geometry
        geometry.ob.T = origin
        self.name = name
        self.origin = origin
        self.material = material

        # Do not set material color yet. The top level URDF colors have not
        # been parsed/defined yet so we do not know what 'Grey' or 'Blue2'
        # mean yet. We set colors after these top level definitions come in

        # Do set it if the color was defined in line by the URDF
        if material is not None and material.color is not None:
            self.geometry.ob.color = material.color


    def __repr__(self) -> str:  # pragma nocover
        return f"Visual(name={self.name}, origin={self.origin}, geometry={self.geometry}, material={self.material})"

    @property
    def geometry(self) -> Any:
        """:class:`.Geometry` : The geometry of this element."""
        return self._geometry

    @geometry.setter
    def geometry(self, value: Any) -> None:
        if not isinstance(value, Geometry):  # pragma nocover
            raise TypeError("Must set geometry with Geometry object")
        self._geometry = value

    @property
    def name(self) -> Any:
        """str : The name of this visual element."""
        return self._name

    @name.setter
    def name(self, value: Any) -> None:
        if value is not None:
            value = str(value)
        self._name = value

    @property
    def origin(self) -> Any:
        """(4,4) float : The pose of this element relative to the link frame."""
        return self._origin

    @origin.setter
    def origin(self, value: Any) -> None:
        self._origin = configure_origin(value)

    @classmethod
    def _from_xml(cls, node: Any, path: Any) -> Any:
        kwargs = cls._parse(node, path)
        kwargs["origin"], _ = parse_origin(node)
        return Visual(**kwargs)


class Inertial(URDFType):
    """The inertial properties of a link.
    Parameters
    ----------
    mass : float
        The mass of the link in kilograms.
    inertia : (3,3) float
        The 3x3 symmetric rotational inertia matrix.
    origin : (4,4) float, optional
        The pose of the inertials relative to the link frame.
        Defaults to identity if not specified.
    """

    _TAG = "inertial"

    def __init__(self, mass: Any = None, inertia: Any = None, origin: Any = None) -> None:
        self._mass = mass
        self._inertia = inertia
        self._origin = origin

    def __repr__(self) -> str:  # pragma nocover
        return f"Inertial(mass={self.mass}, inertia={self.inertia}, origin={self.origin})"
    
    @property
    def mass(self) -> Any:
        """float : The mass of the link in kilograms."""
        return self._mass

    @mass.setter
    def mass(self, value: Any) -> None:
        self._mass = float(value)

    @property
    def inertia(self) -> Any:
        """(3,3) float : The 3x3 symmetric rotational inertia matrix."""
        return self._inertia

    @inertia.setter
    def inertia(self, value: Any) -> None:
        value = np.asanyarray(value).astype(np.float64)
        if not np.allclose(value, value.T):  # pragma nocover
            raise ValueError("Inertia must be a symmetric matrix")
        self._inertia = value

    @property
    def origin(self) -> Any:
        """(4,4) float : The pose of the inertials relative to the link frame."""
        return self._origin

    @origin.setter
    def origin(self, value: Any) -> None:
        self._origin = configure_origin(value)

    @classmethod
    def _from_xml(cls, node: Any, path: Any) -> Any:
        origin, _ = parse_origin(node)
        mass_node = node.find("mass")
        if mass_node is None or "value" not in mass_node.attrib:
            raise URDFError(
                '<inertial> needs a <mass value="..."/> element',
                stage="urdf",
                element=_describe(node),
            )
        mass = float(mass_node.attrib["value"])
        n = node.find("inertia")
        if n is None:
            raise URDFError(
                "<inertial> needs an <inertia .../> element",
                stage="urdf",
                element=_describe(node),
            )
        keys = ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")
        missing = [k for k in keys if k not in n.attrib]
        if missing:
            raise URDFError(
                f"<inertia> is missing attribute(s) {', '.join(missing)}",
                stage="urdf",
                element=_describe(node),
            )
        xx = float(n.attrib["ixx"])
        xy = float(n.attrib["ixy"])
        xz = float(n.attrib["ixz"])
        yy = float(n.attrib["iyy"])
        yz = float(n.attrib["iyz"])
        zz = float(n.attrib["izz"])
        inertia = np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]], dtype=np.float64)
        return Inertial(mass=mass, inertia=inertia, origin=origin)


###############################################################################
# Joint types
###############################################################################


class JointCalibration(URDFType):  # pragma nocover
    """The reference positions of the joint.
    Parameters
    ----------
    rising : float, optional
        When the joint moves in a positive direction, this position will
        trigger a rising edge.
    falling :
        When the joint moves in a positive direction, this position will
        trigger a falling edge.
    """

    _ATTRIBS = {"rising": (float, False), "falling": (float, False)}
    _TAG = "calibration"

    def __init__(self, rising: Any = None, falling: Any = None) -> None:
        self.rising = rising
        self.falling = falling

    def __repr__(self) -> str:  # pragma nocover
        return f"JointCalibration(rising={self.rising}, falling={self.falling})"

    @property
    def rising(self) -> Any:
        """float : description."""
        return self._rising

    @rising.setter
    def rising(self, value: Any) -> None:
        if value is not None:
            value = float(value)
        self._rising = value

    @property
    def falling(self) -> Any:
        """float : description."""
        return self._falling

    @falling.setter
    def falling(self, value: Any) -> None:
        if value is not None:
            value = float(value)
        self._falling = value

    def copy(self, prefix: str = "", scale: Any = None) -> Any:
        """Create a deep copy of the visual with the prefix applied to all names.
        Parameters
        ----------
        prefix : str
            A prefix to apply to all joint and link names.
        Returns
        -------
        :class:`.JointCalibration`
            A deep copy of the visual.
        """
        return JointCalibration(
            rising=self.rising,
            falling=self.falling,
        )


class JointDynamics(URDFType):
    """The dynamic properties of the joint.
    Parameters
    ----------
    damping : float
        The damping value of the joint (Ns/m for prismatic joints,
        Nms/rad for revolute).
    friction : float
        The static friction value of the joint (N for prismatic joints,
        Nm for revolute).
    """

    _ATTRIBS = {
        "damping": (float, False),
        "friction": (float, False),
    }
    _TAG = "dynamics"

    def __init__(self, damping: Any, friction: Any) -> None:
        self.damping = damping
        self.friction = friction

    def __repr__(self) -> str:  # pragma nocover
        return f"JointDynamics(damping={self.damping}, friction={self.friction})"
    
    @property
    def damping(self) -> Any:  # pragma nocover
        """float : The damping value of the joint."""
        return self._damping

    @damping.setter
    def damping(self, value: Any) -> None:
        if value is not None:
            value = float(value)
        self._damping = value

    @property
    def friction(self) -> Any:
        """float : The static friction value of the joint."""
        return self._friction

    @friction.setter
    def friction(self, value: Any) -> None:
        if value is not None:
            value = float(value)
        self._friction = value


class JointLimit(URDFType):
    """The limits of the joint.
    Parameters
    ----------
    effort : float
        The maximum joint effort (N for prismatic joints, Nm for revolute).
    velocity : float
        The maximum joint velocity (m/s for prismatic joints, rad/s for
        revolute).
    lower : float, optional
        The lower joint limit (m for prismatic joints, rad for revolute).
    upper : float, optional
        The upper joint limit (m for prismatic joints, rad for revolute).
    """

    _ATTRIBS = {
        "effort": (float, True),
        "velocity": (float, True),
        "lower": (float, False),
        "upper": (float, False),
    }
    _TAG = "limit"

    def __init__(
        self, effort: Any, velocity: Any, lower: Any = None, upper: Any = None
    ) -> None:
        self.effort = effort
        self.velocity = velocity
        self.lower = lower
        self.upper = upper

    def __repr__(self) -> str:  # pragma nocover
        return f"JointLimit(effort={self.effort}, velocity={self.velocity}, lower={self.lower}, upper={self.upper})"

    @property
    def effort(self) -> Any:
        """float : The maximum joint effort."""
        return self._effort

    @effort.setter
    def effort(self, value: Any) -> None:
        self._effort = float(value)

    @property
    def velocity(self) -> Any:
        """float : The maximum joint velocity."""
        return self._velocity

    @velocity.setter
    def velocity(self, value: Any) -> None:
        self._velocity = float(value)

    @property
    def lower(self) -> Any:
        """float : The lower joint limit."""
        return self._lower

    @lower.setter
    def lower(self, value: Any) -> None:
        if value is not None:
            value = float(value)
        self._lower = value

    @property
    def upper(self) -> Any:
        """float : The upper joint limit."""
        return self._upper

    @upper.setter
    def upper(self, value: Any) -> None:
        if value is not None:
            value = float(value)
        self._upper = value


class JointMimic(URDFType):  # pragma nocover
    """A mimicry tag for a joint, which forces its configuration to
    mimic another joint's.
    This joint's configuration value is set equal to
    ``multiplier * other_joint_cfg + offset``.
    Parameters
    ----------
    joint : str
        The name of the joint to mimic.
    multiplier : float
        The joint configuration multiplier. Defaults to 1.0.
    offset : float, optional
        The joint configuration offset. Defaults to 0.0.
    """

    _ATTRIBS = {
        "joint": (str, True),
        "multiplier": (float, False),
        "offset": (float, False),
    }
    _TAG = "mimic"

    def __init__(self, joint: Any, multiplier: Any = None, offset: Any = None) -> None:
        self.joint = joint
        self.multiplier = multiplier
        self.offset = offset

    def __repr__(self) -> str:  # pragma nocover
        return f"JointMimic(joint={self.joint}, multiplier={self.multiplier}, offset={self.offset})"

    @property
    def joint(self) -> Any:
        """float : The name of the joint to mimic."""
        return self._joint

    @joint.setter
    def joint(self, value: Any) -> None:
        self._joint = str(value)

    @property
    def multiplier(self) -> Any:
        """float : The multiplier for the joint configuration."""
        return self._multiplier

    @multiplier.setter
    def multiplier(self, value: Any) -> None:
        if value is not None:
            value = float(value)
        else:
            value = 1.0
        self._multiplier = value

    @property
    def offset(self) -> Any:
        """float : The offset for the joint configuration"""
        return self._offset

    @offset.setter
    def offset(self, value: Any) -> None:
        if value is not None:
            value = float(value)
        else:
            value = 0.0
        self._offset = value


class SafetyController(URDFType):  # pragma nocover
    """A controller for joint movement safety.
    Parameters
    ----------
    k_velocity : float
        An attribute specifying the relation between the effort and velocity
        limits.
    k_position : float, optional
        An attribute specifying the relation between the position and velocity
        limits. Defaults to 0.0.
    soft_lower_limit : float, optional
        The lower joint boundary where the safety controller kicks in.
        Defaults to 0.0.
    soft_upper_limit : float, optional
        The upper joint boundary where the safety controller kicks in.
        Defaults to 0.0.
    """

    _ATTRIBS = {
        "k_velocity": (float, True),
        "k_position": (float, False),
        "soft_lower_limit": (float, False),
        "soft_upper_limit": (float, False),
    }
    _TAG = "safety_controller"

    def __init__(
        self,
        k_velocity: Any,
        k_position: Any = None,
        soft_lower_limit: Any = None,
        soft_upper_limit: Any = None,
    ) -> None:
        self.k_velocity = k_velocity
        self.k_position = k_position
        self.soft_lower_limit = soft_lower_limit
        self.soft_upper_limit = soft_upper_limit

    def __repr__(self) -> str:  # pragma nocover
        return f"SafetyController(k_velocity={self.k_velocity}, k_position={self.k_position}, soft_lower_limit={self.soft_lower_limit}, soft_upper_limit={self.soft_upper_limit})"

    @property
    def soft_lower_limit(self) -> Any:
        """float : The soft lower limit where the safety controller kicks in."""
        return self._soft_lower_limit

    @soft_lower_limit.setter
    def soft_lower_limit(self, value: Any) -> None:
        if value is not None:
            value = float(value)
        else:
            value = 0.0
        self._soft_lower_limit = value

    @property
    def soft_upper_limit(self) -> Any:
        """float : The soft upper limit where the safety controller kicks in."""
        return self._soft_upper_limit

    @soft_upper_limit.setter
    def soft_upper_limit(self, value: Any) -> None:
        if value is not None:
            value = float(value)
        else:
            value = 0.0
        self._soft_upper_limit = value

    @property
    def k_position(self) -> Any:
        """float : A relation between the position and velocity limits."""
        return self._k_position

    @k_position.setter
    def k_position(self, value: Any) -> None:
        if value is not None:
            value = float(value)
        else:
            value = 0.0
        self._k_position = value

    @property
    def k_velocity(self) -> Any:
        """float : A relation between the effort and velocity limits."""
        return self._k_velocity

    @k_velocity.setter
    def k_velocity(self, value: Any) -> None:
        self._k_velocity = float(value)


###############################################################################
# Transmission types
###############################################################################


class Actuator(URDFType):
    """An actuator.
    Parameters
    ----------
    name : str
        The name of this actuator.
    mechanicalReduction : str, optional
        A specifier for the mechanical reduction at the joint/actuator
        transmission.
    hardwareInterfaces : list of str, optional
        The supported hardware interfaces to the actuator.
    """

    _ATTRIBS = {
        "name": (str, True),
    }
    _TAG = "actuator"

    def __init__(
        self, name: Any, mechanicalReduction: Any = None, hardwareInterfaces: Any = None
    ) -> None:
        self.name = name
        self.mechanicalReduction = mechanicalReduction
        self.hardwareInterfaces = hardwareInterfaces


    def __repr__(self) -> str:  # pragma nocover
        return f"Actuator(name={self.name}, mechanicalReduction={self.mechanicalReduction}, hardwareInterfaces={self.hardwareInterfaces})"

    @property
    def name(self) -> Any:  # pragma nocover
        """str : The name of this actuator."""
        return self._name

    @name.setter
    def name(self, value: Any) -> None:
        self._name = str(value)

    @property
    def mechanicalReduction(self) -> Any:  # pragma nocover
        """str : A specifier for the type of mechanical reduction."""
        return self._mechanicalReduction

    @mechanicalReduction.setter
    def mechanicalReduction(self, value: Any) -> None:
        if value is not None:
            value = str(value)
        self._mechanicalReduction = value

    @property
    def hardwareInterfaces(self) -> Any:  # pragma nocover
        """list of str : The supported hardware interfaces."""
        return self._hardwareInterfaces

    @hardwareInterfaces.setter
    def hardwareInterfaces(self, value: Any) -> None:  # pragma nocover
        if value is None:
            value = []
        else:
            value = list(value)
            for i, v in enumerate(value):
                value[i] = str(v)
        self._hardwareInterfaces = value

    @classmethod
    def _from_xml(cls, node: Any, path: Any) -> Any:  # pragma nocover
        kwargs = cls._parse(node, path)
        mr = node.find("mechanicalReduction")
        if mr is not None:
            mr = float(mr.text)
        kwargs["mechanicalReduction"] = mr
        hi = node.findall("hardwareInterface")
        if len(hi) > 0:
            hi = [h.text for h in hi]
        kwargs["hardwareInterfaces"] = hi
        return Actuator(**kwargs)


class TransmissionJoint(URDFType):
    """A transmission joint specification.
    Parameters
    ----------
    name : str
        The name of this actuator.
    hardwareInterfaces : list of str, optional
        The supported hardware interfaces to the actuator.
    """

    _ATTRIBS = {
        "name": (str, True),
    }
    _TAG = "joint"

    def __init__(self, name: Any, hardwareInterfaces: Any) -> None:
        self.name = name
        self.hardwareInterfaces = hardwareInterfaces

    def __repr__(self) -> str:  # pragma nocover
        return f"TransmissionJoint(name={self.name}, hardwareInterfaces={self.hardwareInterfaces})"
    
    @property
    def name(self) -> Any:  # pragma nocover
        """str : The name of this transmission joint."""
        return self._name

    @name.setter
    def name(self, value: Any) -> None:
        self._name = str(value)

    @property
    def hardwareInterfaces(self) -> Any:  # pragma nocover
        """list of str : The supported hardware interfaces."""
        return self._hardwareInterfaces

    @hardwareInterfaces.setter
    def hardwareInterfaces(self, value: Any) -> None:  # pragma nocover
        if value is None:
            value = []
        else:
            value = list(value)
            for i, v in enumerate(value):
                value[i] = str(v)
        self._hardwareInterfaces = value

    @classmethod
    def _from_xml(cls, node: Any, path: Any) -> Any:
        kwargs = cls._parse(node, path)
        hi = node.findall("hardwareInterface")
        if len(hi) > 0:
            hi = [h.text for h in hi]
        kwargs["hardwareInterfaces"] = hi
        return TransmissionJoint(**kwargs)


###############################################################################
# Top-level types
###############################################################################


class Transmission(URDFType):
    """An element that describes the relationship between an actuator and a
    joint.
    Parameters
    ----------
    name : str
        The name of this transmission.
    trans_type : str
        The type of this transmission.
    joints : list of :class:`.TransmissionJoint`
        The joints connected to this transmission.
    actuators : list of :class:`.Actuator`
        The actuators connected to this transmission.
    """

    _ATTRIBS = {
        "name": (str, True),
    }
    _ELEMENTS = {
        "joints": (TransmissionJoint, False, True),
        "actuators": (Actuator, False, True),
    }
    _TAG = "transmission"

    def __init__(
        self, name: Any, trans_type: Any, joints: Any = None, actuators: Any = None
    ) -> None:
        self.name = name
        self.trans_type = trans_type
        self.joints = joints
        self.actuators = actuators

    def __repr__(self) -> str:  # pragma nocover
        return f"Transmission(name={self.name}, trans_type={self.trans_type}, joints={self.joints}, actuators={self.actuators})"

    @property
    def name(self) -> Any:
        """str : The name of this transmission."""
        return self._name

    @name.setter
    def name(self, value: Any) -> None:
        self._name = str(value)

    @property
    def trans_type(self) -> Any:  # pragma nocover
        """str : The type of this transmission."""
        return self._trans_type

    @trans_type.setter
    def trans_type(self, value: Any) -> None:
        self._trans_type = str(value)

    @property
    def joints(self) -> Any:  # pragma nocover
        """:class:`.TransmissionJoint` : The joints the transmission is
        connected to.
        """
        return self._joints

    @joints.setter
    def joints(self, value: Any) -> None:  # pragma nocover
        if value is None:
            value = []
        else:
            value = list(value)
            for v in value:
                if not isinstance(v, TransmissionJoint):
                    raise TypeError("Joints expects a list of TransmissionJoint")
        self._joints = value

    @property
    def actuators(self) -> Any:  # pragma nocover
        """:class:`.Actuator` : The actuators the transmission is connected to."""
        return self._actuators

    @actuators.setter
    def actuators(self, value: Any) -> None:  # pragma nocover
        if value is None:
            value = []
        else:
            value = list(value)
            for v in value:
                if not isinstance(v, Actuator):
                    raise TypeError("Actuators expects a list of Actuator")
        self._actuators = value

    @classmethod
    def _from_xml(cls, node: Any, path: Any) -> Any:
        kwargs = cls._parse(node, path)
        if node.find("type") is not None:
            kwargs["trans_type"] = node.find("type").text
        else:
            kwargs["trans_type"] = " "  # pragma nocover

        return Transmission(**kwargs)


class Joint(URDFType):
    """A connection between two links.
    There are several types of joints, including:
    - ``fixed`` - a joint that cannot move.
    - ``prismatic`` - a joint that slides along the joint axis.
    - ``revolute`` - a hinge joint that rotates about the axis with a limited
      range of motion.
    - ``continuous`` - a hinge joint that rotates about the axis with an
      unlimited range of motion.
    - ``planar`` - a joint that moves in the plane orthogonal to the axis.
    - ``floating`` - a joint that can move in 6DoF.
    Parameters
    ----------
    name : str
        The name of this joint.
    parent : str
        The name of the parent link of this joint.
    child : str
        The name of the child link of this joint.
    joint_type : str
        The type of the joint. Must be one of :obj:`.Joint.TYPES`.
    axis : (3,) float, optional
        The axis of the joint specified in joint frame. Defaults to
        ``[1,0,0]``.
    origin : (4,4) float, optional
        The pose of the child link with respect to the parent link's frame.
        The joint frame is defined to be coincident with the child link's
        frame, so this is also the pose of the joint frame with respect to
        the parent link's frame.
    limit : :class:`.JointLimit`, optional
        Limit for the joint. Only required for revolute and prismatic
        joints.
    dynamics : :class:`.JointDynamics`, optional
        Dynamics for the joint.
    safety_controller : :class`.SafetyController`, optional
        The safety controller for this joint.
    calibration : :class:`.JointCalibration`, optional
        Calibration information for the joint.
    mimic : :class:`JointMimic`, optional
        Joint mimicry information.
    """

    TYPES = ["fixed", "prismatic", "revolute", "continuous", "floating", "planar"]
    _ATTRIBS = {
        "name": (str, True),
    }
    _ELEMENTS = {
        "dynamics": (JointDynamics, False, False),
        "limit": (JointLimit, False, False),
        "mimic": (JointMimic, False, False),
        "safety_controller": (SafetyController, False, False),
        "calibration": (JointCalibration, False, False),
    }
    _TAG = "joint"

    def __init__(
        self,
        name: Any,
        joint_type: Any,
        parent: Any,
        child: Any,
        axis: Any = None,
        origin: Any = None,
        limit: Any = None,
        dynamics: Any = None,
        safety_controller: Any = None,
        calibration: Any = None,
        mimic: Any = None,
        rpy: Any = None,
    ) -> None:
        self.name = name
        self.parent = parent
        self.child = child
        self.joint_type = joint_type
        self.axis = axis
        self.origin = origin
        self.rpy = rpy
        self.limit = limit
        self.dynamics = dynamics
        self.safety_controller = safety_controller
        self.calibration = calibration
        self.mimic = mimic

    def __repr__(self) -> str:
        return f"Joint(name={self.name}, joint_type={self.joint_type}, parent={self.parent}, child={self.child})"

    @property
    def name(self) -> Any:
        """str : Name for this joint."""
        return self._name

    @name.setter
    def name(self, value: Any) -> None:
        self._name = str(value)

    @property
    def joint_type(self) -> Any:
        """str : The type of this joint."""
        return self._joint_type

    @joint_type.setter
    def joint_type(self, value: Any) -> None:
        value = str(value)
        if value not in Joint.TYPES:  # pragma nocover
            raise ValueError(f"Unsupported joint type {value}")
        self._joint_type = value

    @property
    def parent(self) -> Any:
        """str : The name of the parent link."""
        return self._parent

    @parent.setter
    def parent(self, value: Any) -> None:
        self._parent = str(value)

    @property
    def child(self) -> Any:
        """str : The name of the child link."""
        return self._child

    @child.setter
    def child(self, value: Any) -> None:
        self._child = str(value)

    @property
    def axis(self) -> Any:
        """(3,) float : The joint axis in the joint frame."""
        return self._axis

    @axis.setter
    def axis(self, value: Any) -> None:
        if value is None:
            value = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            value = np.asanyarray(value, dtype=np.float64)
            if value.shape != (3,):  # pragma nocover
                raise ValueError("Invalid shape for axis, should be (3,)")
            norm = np.linalg.norm(value)

            if norm != 0:
                value = value / norm
        self._axis = value

    @property
    def origin(self) -> Any:
        """(4,4) float : The pose of child and joint frames relative to the
        parent link's frame.
        """
        return self._origin

    @origin.setter
    def origin(self, value: Any) -> None:
        self._origin = configure_origin(value)

    @property
    def rpy(self) -> Any:
        return self._rpy

    @rpy.setter
    def rpy(self, value: Any) -> None:
        self._rpy = value

    @property
    def limit(self) -> Any:
        """:class:`.JointLimit` : The limits for this joint."""
        return self._limit

    @limit.setter
    def limit(self, value: Any) -> None:
        if value is None:
            if self.joint_type in ["prismatic", "revolute"]:  # pragma nocover
                raise ValueError(
                    "Require joint limit for prismatic and revolute joints"
                )
        elif not isinstance(value, JointLimit):  # pragma nocover
            raise TypeError("Expected JointLimit type")
        self._limit = value

    @property
    def dynamics(self) -> Any:
        """:class:`.JointDynamics` : The dynamics for this joint."""
        return self._dynamics

    @dynamics.setter
    def dynamics(self, value: Any) -> None:
        if value is not None:
            if not isinstance(value, JointDynamics):  # pragma nocover
                raise TypeError("Expected JointDynamics type")
        self._dynamics = value

    @property
    def safety_controller(self) -> Any:  # pragma nocover
        """:class:`.SafetyController` : The safety controller for this joint."""
        return self._safety_controller

    @safety_controller.setter
    def safety_controller(self, value: Any) -> None:
        if value is not None:
            if not isinstance(value, SafetyController):  # pragma nocover
                raise TypeError("Expected SafetyController type")
        self._safety_controller = value

    @property
    def calibration(self) -> Any:  # pragma nocover
        """:class:`.JointCalibration` : The calibration for this joint."""
        return self._calibration

    @calibration.setter
    def calibration(self, value: Any) -> None:
        if value is not None:
            if not isinstance(value, JointCalibration):  # pragma nocover
                raise TypeError("Expected JointCalibration type")
        self._calibration = value

    @property
    def mimic(self) -> Any:  # pragma nocover
        """:class:`.JointMimic` : The mimic for this joint."""
        return self._mimic

    @mimic.setter
    def mimic(self, value: Any) -> None:
        if value is not None:
            if not isinstance(value, JointMimic):  # pragma nocover
                raise TypeError("Expected JointMimic type")
        self._mimic = value

    def is_valid(self, cfg: Any) -> Any:  # pragma nocover
        """Check if the provided configuration value is valid for this joint.
        Parameters
        ----------
        cfg : float, (2,) float, (6,) float, or (4,4) float
            The configuration of the joint.
        Returns
        -------
        is_valid : bool
            True if the configuration is valid, and False otherwise.
        """
        if self.joint_type not in ["fixed", "revolute"]:
            return True
        if self.joint_limit is None:  # type: ignore[attr-defined]
            return True
        cfg = float(cfg)
        lower = -np.inf
        upper = np.inf
        if self.limit.lower is not None:
            lower = self.limit.lower
        if self.limit.upper is not None:
            upper = self.limit.upper
        return cfg >= lower and cfg <= upper

    @classmethod
    def _from_xml(cls, node: Any, path: Any) -> Any:
        kwargs = cls._parse(node, path)
        if "type" not in node.attrib:
            raise URDFError(
                'joint is missing the required "type" attribute',
                stage="urdf",
                element=_describe(node),
            )
        kwargs["joint_type"] = str(node.attrib["type"])
        for role in ("parent", "child"):
            ref = node.find(role)
            if ref is None or "link" not in ref.attrib:
                raise URDFError(
                    f'joint is missing a <{role} link="..."/> element',
                    stage="urdf",
                    element=_describe(node),
                )
        kwargs["parent"] = node.find("parent").attrib["link"]
        kwargs["child"] = node.find("child").attrib["link"]
        axis = node.find("axis")
        if axis is not None:
            axis = np.fromstring(axis.attrib["xyz"], sep=" ")
        kwargs["axis"] = axis
        kwargs["origin"], kwargs["rpy"] = parse_origin(node)
        return Joint(**kwargs)


class Link(URDFType):
    """A link of a rigid object.
    Parameters
    ----------
    name : str
        The name of the link.
    inertial : :class:`.Inertial`, optional
        The inertial properties of the link.
    visuals : list of :class:`.Visual`, optional
        The visual properties of the link.
    collsions : list of :class:`.Collision`, optional
        The collision properties of the link.
    """

    _ATTRIBS = {
        "name": (str, True),
    }
    _ELEMENTS = {
        "inertial": (Inertial, False, False),
        "visuals": (Visual, False, True),
        "collisions": (Collision, False, True),
    }
    _TAG = "link"

    def __init__(self, name: Any, inertial: Any, visuals: Any, collisions: Any) -> None:
        self.name = name
        self.inertial = inertial
        self.visuals = visuals
        self.collisions = collisions

    def __repr__(self) -> str:  # pragma nocover
        return f"Link(name={self.name}, inertial={self.inertial}, visuals={self.visuals}, collisions={self.collisions})"
    
    @property
    def name(self) -> Any:
        """str : The name of this link."""
        return self._name

    @name.setter
    def name(self, value: Any) -> None:
        self._name = str(value)

    @property
    def inertial(self) -> Any:
        """:class:`.Inertial` : Inertial properties of the link."""
        return self._inertial

    @inertial.setter
    def inertial(self, value: Any) -> None:
        if value is not None and not isinstance(value, Inertial):
            raise TypeError("Expected Inertial object")  # pragma nocover
        # Set default inertial
        if value is None:
            value = Inertial()
        self._inertial = value

    @property
    def visuals(self) -> Any:
        """list of :class:`.Visual` : The visual properties of this link."""
        return self._visuals

    @visuals.setter
    def visuals(self, value: Any) -> None:
        if value is None:  # pragma nocover
            value = []
        else:
            value = list(value)
            for v in value:
                if not isinstance(v, Visual):  # pragma nocover
                    raise ValueError("Expected list of Visual objects")
        self._visuals = value

    @property
    def collisions(self) -> Any:
        """list of :class:`.Collision` : The collision properties of this link."""
        return self._collisions

    @collisions.setter
    def collisions(self, value: Any) -> None:
        if value is None:  # pragma nocover
            value = []
        else:
            value = list(value)
            for v in value:
                if not isinstance(v, Collision):  # pragma nocover
                    raise ValueError("Expected list of Collision objects")
        self._collisions = value


class URDF(URDFType):
    """The top-level URDF specification.
    The URDF encapsulates an articulated object, such as a robot or a gripper.
    It is made of links and joints that tie them together and define their
    relative motions.
    Parameters
    ----------
    name : str
        The name of the URDF.
    links : list of :class:`.Link`
        The links of the URDF.
    joints : list of :class:`.Joint`, optional
        The joints of the URDF.
    transmissions : list of :class:`.Transmission`, optional
        The transmissions of the URDF.
    materials : list of :class:`.Material`, optional
        The materials for the URDF.
    other_xml : str, optional
        A string containing any extra XML for extensions.
    """

    _ATTRIBS = {
        "name": (str, True),
    }
    _ELEMENTS = {
        "links": (Link, True, True),
        "joints": (Joint, False, True),
        "transmissions": (Transmission, False, True),
        "materials": (Material, False, True),
    }
    _TAG = "robot"

    def __init__(
        self,
        name: Any,
        links: Any,
        joints: Any = None,
        transmissions: Any = None,
        materials: Any = None,
        other_xml: Any = None,
    ) -> None:
        if joints is None:  # pragma nocover
            joints = []
        if transmissions is None:  # pragma nocover
            transmissions = []
        if materials is None:
            materials = []

        self.name = name
        self.other_xml = other_xml

        # No setters for these
        self._links = list(links)
        self._joints = list(joints)
        self._transmissions = list(transmissions)
        self._materials = list(materials)
        self._material_map: dict[str, Any] = {}

        for x in self._materials:
            if x.name in self._material_map:
                raise ValueError(f"Two materials with name {x.name} found")
            self._material_map[x.name] = x

        # check for duplicate names
        for kind, items in (
            ("link", self._links),
            ("joint", self._joints),
            ("transmission", self._transmissions),
        ):
            names = [x.name for x in items]
            duplicates = sorted({n for n in names if names.count(n) > 1})
            if duplicates:
                raise URDFError(
                    f"duplicate {kind} name(s): {', '.join(duplicates)}",
                    stage="urdf",
                )

        self._link_map = {link.name: link for link in self._links}
        self._joint_map = {joint.name: joint for joint in self._joints}
        self._transmission_map = {
            transmission.name: transmission for transmission in self._transmissions
        }
        self._validate_transmissions()

    @property
    def name(self) -> Any:
        """str : The name of the URDF."""
        return self._name

    @name.setter
    def name(self, value: Any) -> None:
        self._name = str(value)

    @property
    def links(self) -> Any:
        """list of :class:`.Link` : The links of the URDF.
        This returns a copy of the links array which cannot be edited
        directly. If you want to add or remove links, use
        the appropriate functions.
        """
        return copy.copy(self._links)

    @property
    def link_map(self) -> Any:  # pragma nocover
        """dict : Map from link names to the links themselves.
        This returns a copy of the link map which cannot be edited
        directly. If you want to add or remove links, use
        the appropriate functions.
        """
        return copy.copy(self._link_map)

    @property
    def joints(self) -> Any:
        """list of :class:`.Joint` : The links of the URDF.
        This returns a copy of the joints array which cannot be edited
        directly. If you want to add or remove joints, use
        the appropriate functions.
        """
        return copy.copy(self._joints)

    @property
    def joint_map(self) -> Any:  # pragma nocover
        """dict : Map from joint names to the joints themselves.
        This returns a copy of the joint map which cannot be edited
        directly. If you want to add or remove joints, use
        the appropriate functions.
        """
        return copy.copy(self._joint_map)

    @property
    def transmissions(self) -> Any:
        """list of :class:`.Transmission` : The transmissions of the URDF.
        This returns a copy of the transmissions array which cannot be edited
        directly. If you want to add or remove transmissions, use
        the appropriate functions.
        """
        return copy.copy(self._transmissions)

    @property
    def transmission_map(self) -> Any:  # pragma nocover
        """dict : Map from transmission names to the transmissions themselves.
        This returns a copy of the transmission map which cannot be edited
        directly. If you want to add or remove transmissions, use
        the appropriate functions.
        """
        return copy.copy(self._transmission_map)

    @property
    def other_xml(self) -> Any:  # pragma nocover
        """str : Any extra XML that belongs with the URDF."""
        return self._other_xml

    @other_xml.setter
    def other_xml(self, value: Any) -> None:
        self._other_xml = value

    @property
    def actuated_joints(self) -> Any:  # pragma nocover
        """list of :class:`.Joint` : The joints that are independently
        actuated.
        This excludes mimic joints and fixed joints. The joints are listed
        in topological order, starting from the base-most joint.
        """
        return self._actuated_joints  # type: ignore[attr-defined]

    def _merge_materials(self) -> None:
        """Merge the top-level material set with the link materials."""
        for link in self.links:
            for v in link.visuals:
                if v.material is None:
                    continue
                if v.material.name in self.material_map:  # type: ignore[attr-defined]
                    v.material = self._material_map[v.material.name]
                    v.geometry.ob.color = v.material.color
                else:
                    self._materials.append(v.material)
                    self._material_map[v.material.name] = v.material

    @staticmethod
    def load(file_obj: Any) -> Any:  # pragma nocover
        """Load a URDF from a file.
        Parameters
        ----------
        file_obj : str or file-like object
            The file to load the URDF from. Should be the path to the
            ``.urdf`` XML file. Any paths in the URDF should be specified
            as relative paths to the ``.urdf`` file instead of as ROS
            resources.
        Returns
        -------
        urdf : :class:`.URDF`
            The parsed URDF.
        """
        if isinstance(file_obj, str):
            if os.path.isfile(file_obj):
                parser = ETT.XMLParser()
                tree = ETT.parse(file_obj, parser=parser)
                path, _ = os.path.split(file_obj)
            else:
                raise ValueError(f"{file_obj} is not a file")
        else:
            parser = ETT.XMLParser()
            tree = ETT.parse(file_obj, parser=parser)
            path, _ = os.path.split(file_obj.name)

        node = tree.getroot()
        return URDF._from_xml(node, path)

    @staticmethod
    def loadstr(str_obj: Any, file_obj: Any, base_path: Any = None) -> Any:
        """Load a URDF from a file.
        Parameters
        ----------
        file_obj : str or file-like object
            The file to load the URDF from. Should be the path to the
            ``.urdf`` XML file. Any paths in the URDF should be specified
            as relative paths to the ``.urdf`` file instead of as ROS
            resources.
        Returns
        -------
        urdf : :class:`.URDF`
            The parsed URDF.
        """
        global _base_path

        if base_path is not None:
            _base_path = base_path

        if isinstance(str_obj, str):
            source = BytesIO(bytes(str_obj, "utf-8"))
        else:  # pragma nocover
            source = file_obj

        try:
            tree = ETT.parse(source, parser=ETT.XMLParser())
        except ETT.ParseError as e:
            line, column = getattr(e, "position", (None, None))
            raise URDFError(
                f"not well-formed XML: {e}", stage="xml", line=line, column=column
            ) from e

        return _parse_child(URDF, tree.getroot(), None)

    def _validate_transmissions(self) -> None:
        """Raise an exception if any transmissions are invalidly specified.
        Checks for the following:
        - Transmission joints have valid joint names.
        """
        for t in self.transmissions:  # pragma nocover
            for joint in t.joints:
                if joint.name not in self._joint_map:
                    raise ValueError(
                        f"Transmission {t.name} has invalid joint name {joint.name}"
                    )

    @classmethod
    def _from_xml(cls, node: Any, path: Any) -> Any:
        valid_tags = set(["joint", "link", "transmission", "material"])
        kwargs = cls._parse(node, path)

        extra_xml_node = ETT.Element("extra")
        for child in node:
            if child.tag not in valid_tags:
                extra_xml_node.append(child)

        # data = ETT.ostring(extra_xml_node)
        # kwargs['other_xml'] = data
        return URDF(**kwargs)
