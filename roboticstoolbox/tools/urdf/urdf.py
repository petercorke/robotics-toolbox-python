#!/usr/bin/env python
"""
@author (Original) Matthew Matl, Github: mmatl
@author (Adapted by) Jesse Haviland
"""

import numpy as np
import roboticstoolbox as rtb
import copy
import os
import xml.etree.ElementTree as ET
import spatialmath as sm
from io import BytesIO
from pathlib import Path

from .utils import (parse_origin, configure_origin)

abspath = Path(rtb.__file__).parent / 'models' / 'URDF' / 'xacro'


class URDFType(object):
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
    _ATTRIBS = {}   # Map from attrib name to (type, required)
    _ELEMENTS = {}  # Map from element name to (type, required, multiple)
    _TAG = ''       # XML tag for this element

    def __init__(self):  # pragma nocover
        pass

    @classmethod
    def _parse_attrib(cls, val_type, val):
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
            val = np.fromstring(val, sep=' ')
        else:
            val = val_type(val)
        return val

    @classmethod
    def _parse_simple_attribs(cls, node):
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
                try:
                    v = cls._parse_attrib(t, node.attrib[a])
                except Exception:   # pragma nocover
                    raise ValueError(
                        'Missing required attribute {} when parsing an object '
                        'of type {}'.format(a, cls.__name__)
                    )
            else:
                v = None
                if a in node.attrib:
                    v = cls._parse_attrib(t, node.attrib[a])
            kwargs[a] = v
        return kwargs

    @classmethod
    def _parse_simple_elements(cls, node, path):
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
                if r or v is not None:
                    v = t._from_xml(v, path)
            else:
                vs = node.findall(t._TAG)
                if len(vs) == 0 and r:   # pragma nocover
                    raise ValueError(
                        'Missing required subelement(s) of type {} when '
                        'parsing an object of type {}'.format(
                            t.__name__, cls.__name__
                        )
                    )
                v = [t._from_xml(n, path) for n in vs]
            kwargs[a] = v
        return kwargs

    @classmethod
    def _parse(cls, node, path):
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
    def _from_xml(cls, node, path):
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

    _ATTRIBS = {
        'size': (np.ndarray, True)
    }
    _TAG = 'box'

    def __init__(self, size):
        self.size = size

    @property
    def size(self):
        """(3,) float : The length, width, and height of the box in meters.
        """
        return self._size

    @size.setter
    def size(self, value):
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
        'radius': (float, True),
        'length': (float, True),
    }
    _TAG = 'cylinder'

    def __init__(self, radius, length):
        self.radius = radius
        self.length = length

    @property
    def radius(self):
        """float : The radius of the cylinder in meters.
        """
        return self._radius

    @radius.setter
    def radius(self, value):
        self._radius = float(value)

    @property
    def length(self):
        """float : The length of the cylinder in meters.
        """
        return self._length

    @length.setter
    def length(self, value):
        self._length = float(value)


class Sphere(URDFType):
    """A sphere whose center is at the local origin.
    Parameters
    ----------
    radius : float
        The radius of the sphere in meters.
    """
    _ATTRIBS = {
        'radius': (float, True),
    }
    _TAG = 'sphere'

    def __init__(self, radius):
        self.radius = radius

    @property
    def radius(self):
        """float : The radius of the sphere in meters.
        """
        return self._radius

    @radius.setter
    def radius(self, value):
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
    _ATTRIBS = {
        'filename': (str, True),
        'scale': (np.ndarray, False)
    }
    _TAG = 'mesh'

    def __init__(self, filename, scale=None):
        self.filename = filename
        self.scale = scale

    @property
    def filename(self):
        """str : The path to the mesh file for this object.
        """
        return self._filename

    @filename.setter
    def filename(self, value):

        if value.startswith('package://'):
            value = value.replace('package://', '')

        value = str(abspath / value)

        # print(value)

        self._filename = value

    @property
    def scale(self):
        """(3,) float : A scaling for the mesh along its local XYZ axes.
        """
        return self._scale

    @scale.setter
    def scale(self, value):
        if value is not None:
            value = np.asanyarray(value).astype(np.float64)
        self._scale = value

    @classmethod
    def _from_xml(cls, node, path):
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
    _ATTRIBS = {
        'name': (str, True)
    }
    _ELEMENTS = {}
    _TAG = 'material'

    def __init__(self, name, color=None, texture=None):

        # if color is None:
        #     color = name

        self.name = name
        self.color = color
        self.texture = texture

    @classmethod
    def _from_xml(cls, node, path):  # pragma nocover
        kwargs = cls._parse(node, path)

        # Extract the color -- it's weirdly an attribute of a subelement
        color = node.find('color')
        if color is not None:
            color = np.fromstring(
                color.attrib['rgba'], sep=' ', dtype=np.float64)
        kwargs['color'] = color

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
        'box': (Box, False, False),
        'cylinder': (Cylinder, False, False),
        'sphere': (Sphere, False, False),
        'mesh': (Mesh, False, False),
    }
    _TAG = 'geometry'

    def __init__(self, box=None, cylinder=None, sphere=None, mesh=None):
        if (box is None and cylinder is None and
                sphere is None and mesh is None):   # pragma nocover
            raise ValueError('At least one geometry element must be set')

        if box is not None:
            self.box = box
            self.ob = rtb.Box(box.size)

        if cylinder is not None:
            self.cylinder = cylinder
            self.ob = rtb.Cylinder(cylinder.radius, cylinder.length)

        if sphere is not None:
            self.sphere = sphere
            self.ob = rtb.Sphere(sphere.radius)

        if mesh is not None:
            self.mesh = mesh
            self.ob = rtb.Mesh(mesh.filename, scale=mesh.scale)

    @property
    def box(self):
        """:class:`.Box` : Box geometry.
        """
        return self._box

    @box.setter
    def box(self, value):
        if value is not None and not isinstance(value, Box):   # pragma nocover
            raise TypeError('Expected Box type')
        self._box = value

    @property
    def cylinder(self):
        """:class:`.Cylinder` : Cylinder geometry.
        """
        return self._cylinder

    @cylinder.setter
    def cylinder(self, value):
        if value is not None and not isinstance(value, Cylinder):
            raise TypeError('Expected Cylinder type')   # pragma nocover
        self._cylinder = value

    @property
    def sphere(self):
        """:class:`.Sphere` : Spherical geometry.
        """
        return self._sphere

    @sphere.setter
    def sphere(self, value):
        if value is not None and not isinstance(value, Sphere):
            raise TypeError('Expected Sphere type')   # pragma nocover
        self._sphere = value

    @property
    def mesh(self):
        """:class:`.Mesh` : Mesh geometry.
        """
        return self._mesh

    @mesh.setter
    def mesh(self, value):
        if value is not None and not isinstance(value, Mesh):
            raise TypeError('Expected Mesh type')   # pragma nocover
        self._mesh = value

    @property
    def geometry(self):   # pragma nocover
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

    _ATTRIBS = {
        'name': (str, False)
    }
    _ELEMENTS = {
        'geometry': (Geometry, True, False),
    }
    _TAG = 'collision'

    def __init__(self, name, origin, geometry):
        self.geometry = geometry
        self.name = name
        self.origin = origin
        self.geometry.ob.base = origin

    @property
    def geometry(self):
        """:class:`.Geometry` : The geometry of this element.
        """
        return self._geometry

    @geometry.setter
    def geometry(self, value):
        if not isinstance(value, Geometry):   # pragma nocover
            raise TypeError('Must set geometry with Geometry object')
        self._geometry = value

    @property
    def name(self):
        """str : The name of this collision element.
        """
        return self._name

    @name.setter
    def name(self, value):
        if value is not None:
            value = str(value)
        self._name = value

    @property
    def origin(self):
        """(4,4) float : The pose of this element relative to the link frame.
        """
        return self._origin

    @origin.setter
    def origin(self, value):
        self._origin = configure_origin(value)

    @classmethod
    def _from_xml(cls, node, path):
        kwargs = cls._parse(node, path)
        kwargs['origin'], _ = parse_origin(node)
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
    _ATTRIBS = {
        'name': (str, False)
    }
    _ELEMENTS = {
        'geometry': (Geometry, True, False),
        'material': (Material, False, False)
    }
    _TAG = 'visual'

    def __init__(self, geometry, name=None, origin=None, material=None):
        self.geometry = geometry
        geometry.ob.base = origin
        self.name = name
        self.origin = origin
        self.material = material

        # Do not set material color yet. The top level URDF colors have not
        # been parsed/defined yet so we do not know what 'Grey' or 'Blue2'
        # mean yet. We set colors after these top level definitions come in

        # Do set it if the color was defined in line by the URDF
        if material is not None and material.color is not None:
            self.geometry.ob.color = material.color

    @property
    def geometry(self):
        """:class:`.Geometry` : The geometry of this element.
        """
        return self._geometry

    @geometry.setter
    def geometry(self, value):
        if not isinstance(value, Geometry):   # pragma nocover
            raise TypeError('Must set geometry with Geometry object')
        self._geometry = value

    @property
    def name(self):
        """str : The name of this visual element.
        """
        return self._name

    @name.setter
    def name(self, value):
        if value is not None:
            value = str(value)
        self._name = value

    @property
    def origin(self):
        """(4,4) float : The pose of this element relative to the link frame.
        """
        return self._origin

    @origin.setter
    def origin(self, value):
        self._origin = configure_origin(value)

    @classmethod
    def _from_xml(cls, node, path):
        kwargs = cls._parse(node, path)
        kwargs['origin'], _ = parse_origin(node)
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
    _TAG = 'inertial'

    def __init__(self, mass, inertia, origin=None):
        self.mass = mass
        self.inertia = inertia
        self.origin = origin

    @property
    def mass(self):
        """float : The mass of the link in kilograms.
        """
        return self._mass

    @mass.setter
    def mass(self, value):
        self._mass = float(value)

    @property
    def inertia(self):
        """(3,3) float : The 3x3 symmetric rotational inertia matrix.
        """
        return self._inertia

    @inertia.setter
    def inertia(self, value):
        value = np.asanyarray(value).astype(np.float64)
        if not np.allclose(value, value.T):  # pragma nocover
            raise ValueError('Inertia must be a symmetric matrix')
        self._inertia = value

    @property
    def origin(self):
        """(4,4) float : The pose of the inertials relative to the link frame.
        """
        return self._origin

    @origin.setter
    def origin(self, value):
        self._origin = configure_origin(value)

    @classmethod
    def _from_xml(cls, node, path):
        origin, _ = parse_origin(node)
        mass = float(node.find('mass').attrib['value'])
        n = node.find('inertia')
        xx = float(n.attrib['ixx'])
        xy = float(n.attrib['ixy'])
        xz = float(n.attrib['ixz'])
        yy = float(n.attrib['iyy'])
        yz = float(n.attrib['iyz'])
        zz = float(n.attrib['izz'])
        inertia = np.array([
            [xx, xy, xz],
            [xy, yy, yz],
            [xz, yz, zz]
        ], dtype=np.float64)
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
    _ATTRIBS = {
        'rising': (float, False),
        'falling': (float, False)
    }
    _TAG = 'calibration'

    def __init__(self, rising=None, falling=None):
        self.rising = rising
        self.falling = falling

    @property
    def rising(self):
        """float : description.
        """
        return self._rising

    @rising.setter
    def rising(self, value):
        if value is not None:
            value = float(value)
        self._rising = value

    @property
    def falling(self):
        """float : description.
        """
        return self._falling

    @falling.setter
    def falling(self, value):
        if value is not None:
            value = float(value)
        self._falling = value

    def copy(self, prefix='', scale=None):
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
        'damping': (float, False),
        'friction': (float, False),
    }
    _TAG = 'dynamics'

    def __init__(self, damping, friction):
        self.damping = damping
        self.friction = friction

    @property
    def damping(self):    # pragma nocover
        """float : The damping value of the joint.
        """
        return self._damping

    @damping.setter
    def damping(self, value):
        if value is not None:
            value = float(value)
        self._damping = value

    @property
    def friction(self):
        """float : The static friction value of the joint.
        """
        return self._friction

    @friction.setter
    def friction(self, value):
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
        'effort': (float, True),
        'velocity': (float, True),
        'lower': (float, False),
        'upper': (float, False),
    }
    _TAG = 'limit'

    def __init__(self, effort, velocity, lower=None, upper=None):
        self.effort = effort
        self.velocity = velocity
        self.lower = lower
        self.upper = upper

    @property
    def effort(self):
        """float : The maximum joint effort.
        """
        return self._effort

    @effort.setter
    def effort(self, value):
        self._effort = float(value)

    @property
    def velocity(self):
        """float : The maximum joint velocity.
        """
        return self._velocity

    @velocity.setter
    def velocity(self, value):
        self._velocity = float(value)

    @property
    def lower(self):
        """float : The lower joint limit.
        """
        return self._lower

    @lower.setter
    def lower(self, value):
        if value is not None:
            value = float(value)
        self._lower = value

    @property
    def upper(self):
        """float : The upper joint limit.
        """
        return self._upper

    @upper.setter
    def upper(self, value):
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
        'joint': (str, True),
        'multiplier': (float, False),
        'offset': (float, False),
    }
    _TAG = 'mimic'

    def __init__(self, joint, multiplier=None, offset=None):
        self.joint = joint
        self.multiplier = multiplier
        self.offset = offset

    @property
    def joint(self):
        """float : The name of the joint to mimic.
        """
        return self._joint

    @joint.setter
    def joint(self, value):
        self._joint = str(value)

    @property
    def multiplier(self):
        """float : The multiplier for the joint configuration.
        """
        return self._multiplier

    @multiplier.setter
    def multiplier(self, value):
        if value is not None:
            value = float(value)
        else:
            value = 1.0
        self._multiplier = value

    @property
    def offset(self):
        """float : The offset for the joint configuration
        """
        return self._offset

    @offset.setter
    def offset(self, value):
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
        'k_velocity': (float, True),
        'k_position': (float, False),
        'soft_lower_limit': (float, False),
        'soft_upper_limit': (float, False),
    }
    _TAG = 'safety_controller'

    def __init__(self, k_velocity, k_position=None, soft_lower_limit=None,
                 soft_upper_limit=None):
        self.k_velocity = k_velocity
        self.k_position = k_position
        self.soft_lower_limit = soft_lower_limit
        self.soft_upper_limit = soft_upper_limit

    @property
    def soft_lower_limit(self):
        """float : The soft lower limit where the safety controller kicks in.
        """
        return self._soft_lower_limit

    @soft_lower_limit.setter
    def soft_lower_limit(self, value):
        if value is not None:
            value = float(value)
        else:
            value = 0.0
        self._soft_lower_limit = value

    @property
    def soft_upper_limit(self):
        """float : The soft upper limit where the safety controller kicks in.
        """
        return self._soft_upper_limit

    @soft_upper_limit.setter
    def soft_upper_limit(self, value):
        if value is not None:
            value = float(value)
        else:
            value = 0.0
        self._soft_upper_limit = value

    @property
    def k_position(self):
        """float : A relation between the position and velocity limits.
        """
        return self._k_position

    @k_position.setter
    def k_position(self, value):
        if value is not None:
            value = float(value)
        else:
            value = 0.0
        self._k_position = value

    @property
    def k_velocity(self):
        """float : A relation between the effort and velocity limits.
        """
        return self._k_velocity

    @k_velocity.setter
    def k_velocity(self, value):
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
        'name': (str, True),
    }
    _TAG = 'actuator'

    def __init__(self, name, mechanicalReduction=None,
                 hardwareInterfaces=None):
        self.name = name
        self.mechanicalReduction = mechanicalReduction
        self.hardwareInterfaces = hardwareInterfaces

    @property
    def name(self):  # pragma nocover
        """str : The name of this actuator.
        """
        return self._name

    @name.setter
    def name(self, value):
        self._name = str(value)

    @property
    def mechanicalReduction(self):     # pragma nocover
        """str : A specifier for the type of mechanical reduction.
        """
        return self._mechanicalReduction

    @mechanicalReduction.setter
    def mechanicalReduction(self, value):
        if value is not None:
            value = str(value)
        self._mechanicalReduction = value

    @property
    def hardwareInterfaces(self):   # pragma nocover
        """list of str : The supported hardware interfaces.
        """
        return self._hardwareInterfaces

    @hardwareInterfaces.setter
    def hardwareInterfaces(self, value):   # pragma nocover
        if value is None:
            value = []
        else:
            value = list(value)
            for i, v in enumerate(value):
                value[i] = str(v)
        self._hardwareInterfaces = value

    @classmethod
    def _from_xml(cls, node, path):   # pragma nocover
        kwargs = cls._parse(node, path)
        mr = node.find('mechanicalReduction')
        if mr is not None:
            mr = float(mr.text)
        kwargs['mechanicalReduction'] = mr
        hi = node.findall('hardwareInterface')
        if len(hi) > 0:
            hi = [h.text for h in hi]
        kwargs['hardwareInterfaces'] = hi
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
        'name': (str, True),
    }
    _TAG = 'joint'

    def __init__(self, name, hardwareInterfaces):
        self.name = name
        self.hardwareInterfaces = hardwareInterfaces

    @property
    def name(self):     # pragma nocover
        """str : The name of this transmission joint.
        """
        return self._name

    @name.setter
    def name(self, value):
        self._name = str(value)

    @property
    def hardwareInterfaces(self):   # pragma nocover
        """list of str : The supported hardware interfaces.
        """
        return self._hardwareInterfaces

    @hardwareInterfaces.setter
    def hardwareInterfaces(self, value):   # pragma nocover
        if value is None:
            value = []
        else:
            value = list(value)
            for i, v in enumerate(value):
                value[i] = str(v)
        self._hardwareInterfaces = value

    @classmethod
    def _from_xml(cls, node, path):
        kwargs = cls._parse(node, path)
        hi = node.findall('hardwareInterface')
        if len(hi) > 0:
            hi = [h.text for h in hi]
        kwargs['hardwareInterfaces'] = hi
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
        'name': (str, True),
    }
    _ELEMENTS = {
        'joints': (TransmissionJoint, True, True),
        'actuators': (Actuator, True, True),
    }
    _TAG = 'transmission'

    def __init__(self, name, trans_type, joints=None, actuators=None):
        self.name = name
        self.trans_type = trans_type
        self.joints = joints
        self.actuators = actuators

    @property
    def name(self):
        """str : The name of this transmission.
        """
        return self._name

    @name.setter
    def name(self, value):
        self._name = str(value)

    @property
    def trans_type(self):   # pragma nocover
        """str : The type of this transmission.
        """
        return self._trans_type

    @trans_type.setter
    def trans_type(self, value):
        self._trans_type = str(value)

    @property
    def joints(self):     # pragma nocover
        """:class:`.TransmissionJoint` : The joints the transmission is
        connected to.
        """
        return self._joints

    @joints.setter
    def joints(self, value):   # pragma nocover
        if value is None:
            value = []
        else:
            value = list(value)
            for v in value:
                if not isinstance(v, TransmissionJoint):
                    raise TypeError(
                        'Joints expects a list of TransmissionJoint'
                    )
        self._joints = value

    @property
    def actuators(self):     # pragma nocover
        """:class:`.Actuator` : The actuators the transmission is connected to.
        """
        return self._actuators

    @actuators.setter
    def actuators(self, value):   # pragma nocover
        if value is None:
            value = []
        else:
            value = list(value)
            for v in value:
                if not isinstance(v, Actuator):
                    raise TypeError(
                        'Actuators expects a list of Actuator'
                    )
        self._actuators = value

    @classmethod
    def _from_xml(cls, node, path):
        kwargs = cls._parse(node, path)
        if node.find('type') is not None:
            kwargs['trans_type'] = node.find('type').text
        else:
            kwargs['trans_type'] = ' '    # pragma nocover

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
    TYPES = ['fixed', 'prismatic', 'revolute',
             'continuous', 'floating', 'planar']
    _ATTRIBS = {
        'name': (str, True),
    }
    _ELEMENTS = {
        'dynamics': (JointDynamics, False, False),
        'limit': (JointLimit, False, False),
        'mimic': (JointMimic, False, False),
        'safety_controller': (SafetyController, False, False),
        'calibration': (JointCalibration, False, False),
    }
    _TAG = 'joint'

    def __init__(self, name, joint_type, parent, child, axis=None, origin=None,
                 limit=None, dynamics=None, safety_controller=None,
                 calibration=None, mimic=None, rpy=None):
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

    @property
    def name(self):
        """str : Name for this joint.
        """
        return self._name

    @name.setter
    def name(self, value):
        self._name = str(value)

    @property
    def joint_type(self):
        """str : The type of this joint.
        """
        return self._joint_type

    @joint_type.setter
    def joint_type(self, value):
        value = str(value)
        if value not in Joint.TYPES:   # pragma nocover
            raise ValueError('Unsupported joint type {}'.format(value))
        self._joint_type = value

    @property
    def parent(self):
        """str : The name of the parent link.
        """
        return self._parent

    @parent.setter
    def parent(self, value):
        self._parent = str(value)

    @property
    def child(self):
        """str : The name of the child link.
        """
        return self._child

    @child.setter
    def child(self, value):
        self._child = str(value)

    @property
    def axis(self):
        """(3,) float : The joint axis in the joint frame.
        """
        return self._axis

    @axis.setter
    def axis(self, value):
        if value is None:
            value = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            value = np.asanyarray(value, dtype=np.float64)
            if value.shape != (3,):    # pragma nocover
                raise ValueError('Invalid shape for axis, should be (3,)')
            norm = np.linalg.norm(value)

            if norm != 0:
                value = value / norm
        self._axis = value

    @property
    def origin(self):
        """(4,4) float : The pose of child and joint frames relative to the
        parent link's frame.
        """
        return self._origin

    @origin.setter
    def origin(self, value):
        self._origin = configure_origin(value)

    @property
    def rpy(self):
        return self._rpy

    @rpy.setter
    def rpy(self, value):
        self._rpy = value

    @property
    def limit(self):
        """:class:`.JointLimit` : The limits for this joint.
        """
        return self._limit

    @limit.setter
    def limit(self, value):
        if value is None:
            if self.joint_type in ['prismatic', 'revolute']:   # pragma nocover
                raise ValueError('Require joint limit for prismatic and '
                                 'revolute joints')
        elif not isinstance(value, JointLimit):   # pragma nocover
            raise TypeError('Expected JointLimit type')
        self._limit = value

    @property
    def dynamics(self):
        """:class:`.JointDynamics` : The dynamics for this joint.
        """
        return self._dynamics

    @dynamics.setter
    def dynamics(self, value):
        if value is not None:
            if not isinstance(value, JointDynamics):   # pragma nocover
                raise TypeError('Expected JointDynamics type')
        self._dynamics = value

    @property
    def safety_controller(self):   # pragma nocover
        """:class:`.SafetyController` : The safety controller for this joint.
        """
        return self._safety_controller

    @safety_controller.setter
    def safety_controller(self, value):
        if value is not None:
            if not isinstance(value, SafetyController):   # pragma nocover
                raise TypeError('Expected SafetyController type')
        self._safety_controller = value

    @property
    def calibration(self):   # pragma nocover
        """:class:`.JointCalibration` : The calibration for this joint.
        """
        return self._calibration

    @calibration.setter
    def calibration(self, value):
        if value is not None:
            if not isinstance(value, JointCalibration):   # pragma nocover
                raise TypeError('Expected JointCalibration type')
        self._calibration = value

    @property
    def mimic(self):   # pragma nocover
        """:class:`.JointMimic` : The mimic for this joint.
        """
        return self._mimic

    @mimic.setter
    def mimic(self, value):
        if value is not None:
            if not isinstance(value, JointMimic):   # pragma nocover
                raise TypeError('Expected JointMimic type')
        self._mimic = value

    def is_valid(self, cfg):    # pragma nocover
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
        if self.joint_type not in ['fixed', 'revolute']:
            return True
        if self.joint_limit is None:
            return True
        cfg = float(cfg)
        lower = -np.infty
        upper = np.infty
        if self.limit.lower is not None:
            lower = self.limit.lower
        if self.limit.upper is not None:
            upper = self.limit.upper
        return (cfg >= lower and cfg <= upper)

    @classmethod
    def _from_xml(cls, node, path):
        kwargs = cls._parse(node, path)
        kwargs['joint_type'] = str(node.attrib['type'])
        kwargs['parent'] = node.find('parent').attrib['link']
        kwargs['child'] = node.find('child').attrib['link']
        axis = node.find('axis')
        if axis is not None:
            axis = np.fromstring(axis.attrib['xyz'], sep=' ')
        kwargs['axis'] = axis
        kwargs['origin'], kwargs['rpy'] = parse_origin(node)
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
        'name': (str, True),
    }
    _ELEMENTS = {
        'inertial': (Inertial, False, False),
        'visuals': (Visual, False, True),
        'collisions': (Collision, False, True),
    }
    _TAG = 'link'

    def __init__(self, name, inertial, visuals, collisions):
        self.name = name
        self.inertial = inertial
        self.visuals = visuals
        self.collisions = collisions

    @property
    def name(self):
        """str : The name of this link.
        """
        return self._name

    @name.setter
    def name(self, value):
        self._name = str(value)

    @property
    def inertial(self):
        """:class:`.Inertial` : Inertial properties of the link.
        """
        return self._inertial

    @inertial.setter
    def inertial(self, value):
        if value is not None and not isinstance(value, Inertial):
            raise TypeError('Expected Inertial object')   # pragma nocover
        # Set default inertial
        if value is None:
            value = Inertial(mass=0.0, inertia=np.eye(3))
        self._inertial = value

    @property
    def visuals(self):
        """list of :class:`.Visual` : The visual properties of this link.
        """
        return self._visuals

    @visuals.setter
    def visuals(self, value):
        if value is None:   # pragma nocover
            value = []
        else:
            value = list(value)
            for v in value:
                if not isinstance(v, Visual):    # pragma nocover
                    raise ValueError('Expected list of Visual objects')
        self._visuals = value

    @property
    def collisions(self):
        """list of :class:`.Collision` : The collision properties of this link.
        """
        return self._collisions

    @collisions.setter
    def collisions(self, value):
        if value is None:    # pragma nocover
            value = []
        else:
            value = list(value)
            for v in value:
                if not isinstance(v, Collision):    # pragma nocover
                    raise ValueError('Expected list of Collision objects')
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
        'name': (str, True),
    }
    _ELEMENTS = {
        'links': (Link, True, True),
        'joints': (Joint, False, True),
        'transmissions': (Transmission, False, True),
        'materials': (Material, False, True),
    }
    _TAG = 'robot'

    def __init__(self, name, links, joints=None,
                 transmissions=None, materials=None,
                 other_xml=None):
        if joints is None:   # pragma nocover
            joints = []
        if transmissions is None:   # pragma nocover
            transmissions = []
        if materials is None:
            materials = []

        # TODO, what does this next line do?
        # why arent the other things validated
        try:
            self._validate_transmissions()
        except Exception:
            pass

        self.name = name
        self.other_xml = other_xml

        # No setters for these
        self._links = list(links)
        self._joints = list(joints)
        self._transmissions = list(transmissions)
        self._materials = list(materials)
        self._material_map = {}

        for x in self._materials:
            if x.name in self._material_map:
                raise ValueError('Two materials with name {} '
                                 'found'.format(x.name))
            self._material_map[x.name] = x

        # check for duplicate names
        if len(self._links) > len(set([x.name for x in self._links])):     # pragma nocover  # noqa
            raise ValueError('Duplicate link names')
        if len(self._joints) > len(set([x.name for x in self._joints])):     # pragma nocover  # noqa
            raise ValueError('Duplicate joint names')
        if len(self._transmissions) > len(
                set([x.name for x in self._transmissions])):     # pragma nocover  # noqa
            raise ValueError('Duplicate transmission names')

        elinks = []
        elinkdict = {}
        # jointdict = {}

        # build the list of links in URDF file order
        for link in self._links:
            elink = rtb.ELink(name=link.name)
            elinks.append(elink)
            elinkdict[link.name] = elink

            # add the inertial parameters
            elink.r = link.inertial.origin[:3, 3]
            elink.m = link.inertial.mass
            elink.inertia = link.inertial.inertia

            # add the visuals to visual list
            try:
                elink.geometry = [v.geometry.ob for v in link.visuals]
            except AttributeError:   # pragma nocover
                pass

            #  add collision objects to collision object list
            try:
                elink.collision = [col.geometry.ob for col in link.collisions]
            except AttributeError:   # pragma nocover
                pass

        # connect the links using joint info
        for joint in self._joints:
            # get references to joint's parent and child
            childlink = elinkdict[joint.child]
            parentlink = elinkdict[joint.parent]

            childlink._parent = parentlink  # connect child link to parent
            childlink._joint_name = joint.name

            # constant part of link transform
            trans = sm.SE3(joint.origin).t
            # TODO, find where reverse is used and change
            # it to [::-1] or do that here
            rot = joint.rpy
            childlink._ets = rtb.ETS.SE3(trans, rot)
            childlink._init_Ts()

            # variable part of link transform
            if joint.joint_type in ('revolute', 'continuous'):   # pragma nocover # noqa
                if joint.axis[0] == 1:
                    var = rtb.ETS.rx()
                elif joint.axis[0] == -1:
                    var = rtb.ETS.rx(flip=True)
                elif joint.axis[1] == 1:
                    var = rtb.ETS.ry()
                elif joint.axis[1] == -1:
                    var = rtb.ETS.ry(flip=True)
                elif joint.axis[2] == 1:
                    var = rtb.ETS.rz()
                elif joint.axis[2] == -1:
                    var = rtb.ETS.rz(flip=True)
            elif joint.joint_type == 'prismatic':   # pragma nocover
                if joint.axis[0] == 1:
                    var = rtb.ETS.tx()
                elif joint.axis[0] == -1:
                    var = rtb.ETS.tx(flip=True)
                elif joint.axis[1] == 1:
                    var = rtb.ETS.ty()
                elif joint.axis[1] == -1:
                    var = rtb.ETS.ty(flip=True)
                elif joint.axis[2] == 1:
                    var = rtb.ETS.tz()
                elif joint.axis[2] == -1:
                    var = rtb.ETS.tz(flip=True)
            elif joint.joint_type == 'fixed':
                var = None

            childlink._v = var

            # joint limit
            try:
                joint.qlim = [joint.limit.lower, joint.limit.upper]
            except AttributeError:
                # no joint limits provided
                pass

            # joint friction
            try:
                if joint.dynamics.friction is not None:
                    childlink.B = joint.dynamics.friction

                # TODO Add damping
                # joint.dynamics.damping
            except AttributeError:
                pass

            # joint gear ratio
            # TODO, not sure if t.joint.name is a thing
            for t in self.transmissions:     # pragma nocover
                if t.name == joint.name:
                    childlink.G = t.actuators[0].mechanicalReduction

            self.elinks = elinks

            # TODO, why did you put the base_link on the end?
            # easy to do it here

    @property
    def name(self):
        """str : The name of the URDF.
        """
        return self._name

    @name.setter
    def name(self, value):
        self._name = str(value)

    @property
    def links(self):
        """list of :class:`.Link` : The links of the URDF.
        This returns a copy of the links array which cannot be edited
        directly. If you want to add or remove links, use
        the appropriate functions.
        """
        return copy.copy(self._links)

    @property
    def link_map(self):   # pragma nocover
        """dict : Map from link names to the links themselves.
        This returns a copy of the link map which cannot be edited
        directly. If you want to add or remove links, use
        the appropriate functions.
        """
        return copy.copy(self._link_map)

    @property
    def joints(self):
        """list of :class:`.Joint` : The links of the URDF.
        This returns a copy of the joints array which cannot be edited
        directly. If you want to add or remove joints, use
        the appropriate functions.
        """
        return copy.copy(self._joints)

    @property
    def joint_map(self):     # pragma nocover
        """dict : Map from joint names to the joints themselves.
        This returns a copy of the joint map which cannot be edited
        directly. If you want to add or remove joints, use
        the appropriate functions.
        """
        return copy.copy(self._joint_map)

    @property
    def transmissions(self):
        """list of :class:`.Transmission` : The transmissions of the URDF.
        This returns a copy of the transmissions array which cannot be edited
        directly. If you want to add or remove transmissions, use
        the appropriate functions.
        """
        return copy.copy(self._transmissions)

    @property
    def transmission_map(self):   # pragma nocover
        """dict : Map from transmission names to the transmissions themselves.
        This returns a copy of the transmission map which cannot be edited
        directly. If you want to add or remove transmissions, use
        the appropriate functions.
        """
        return copy.copy(self._transmission_map)

    @property
    def other_xml(self):   # pragma nocover
        """str : Any extra XML that belongs with the URDF.
        """
        return self._other_xml

    @other_xml.setter
    def other_xml(self, value):
        self._other_xml = value

    @property
    def actuated_joints(self):   # pragma nocover
        """list of :class:`.Joint` : The joints that are independently
        actuated.
        This excludes mimic joints and fixed joints. The joints are listed
        in topological order, starting from the base-most joint.
        """
        return self._actuated_joints

    def _merge_materials(self):
        """Merge the top-level material set with the link materials.
        """
        for link in self.links:
            for v in link.visuals:
                if v.material is None:
                    continue
                if v.material.name in self.material_map:
                    v.material = self._material_map[v.material.name]
                    v.geometry.ob.color = v.material.color
                else:
                    self._materials.append(v.material)
                    self._material_map[v.material.name] = v.material

    @staticmethod
    def load(file_obj):     # pragma nocover
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
                parser = ET.XMLParser()
                tree = ET.parse(file_obj, parser=parser)
                path, _ = os.path.split(file_obj)
            else:
                raise ValueError('{} is not a file'.format(file_obj))
        else:
            parser = ET.XMLParser()
            tree = ET.parse(file_obj, parser=parser)
            path, _ = os.path.split(file_obj.name)

        node = tree.getroot()
        return URDF._from_xml(node, path)

    @staticmethod
    def loadstr(str_obj, file_obj):
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
        if isinstance(str_obj, str):
            if os.path.isfile(file_obj):
                parser = ET.XMLParser()
                bytes_obj = BytesIO(bytes(str_obj, 'utf-8'))
                tree = ET.parse(bytes_obj, parser=parser)
                path, _ = os.path.split(file_obj)

        else:   # pragma nocover
            parser = ET.XMLParser()
            tree = ET.parse(file_obj, parser=parser)
            path, _ = os.path.split(file_obj.name)

        node = tree.getroot()
        return URDF._from_xml(node, path)

    def _validate_transmissions(self):
        """Raise an exception of any transmissions are invalidly specified.
        Checks for the following:
        - Transmission joints have valid joint names.
        """
        for t in self.transmissions:   # pragma nocover
            for joint in t.joints:
                if joint.name not in self._joint_map:
                    raise ValueError('Transmission {} has invalid joint name '
                                     '{}'.format(t.name, joint.name))

    @classmethod
    def _from_xml(cls, node, path):
        valid_tags = set([
            'joint', 'link', 'transmission',
            'material'
            ])
        kwargs = cls._parse(node, path)

        extra_xml_node = ET.Element('extra')
        for child in node:
            if child.tag not in valid_tags:
                extra_xml_node.append(child)

        # data = ET.ostring(extra_xml_node)
        # kwargs['other_xml'] = data
        return URDF(**kwargs)
