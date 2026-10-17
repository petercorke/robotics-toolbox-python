Elementary transform sequences
==============================

Elementary transforms are canonic rotations or translations about, or along,
the x-, y- or z-axes.  The amount of rotation or translation can be a constant
or a variable.  A variable amount corresponds to a joint.  A sequence of them
is an *elementary transform sequence* (ETS), and describes the kinematics of a
serial-link robot arm, or of one link of a branched manipulator.

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Page
     - Sequence
     - Element
   * - :doc:`3D <ets_3d>`
     - :class:`~roboticstoolbox.ets.ETS.ETS`
     - :class:`~roboticstoolbox.ets.ET.ET`, an :math:`\SE{3}` transform
   * - :doc:`2D <ets_2d>`
     - :class:`~roboticstoolbox.ets.ETS2.ETS2`
     - :class:`~roboticstoolbox.ets.ET2.ET2`, an :math:`\SE{2}` transform

.. toctree::
   :maxdepth: 2

   ets_3d
   ets_2d

Consider the example:

.. runblock:: pycon
   :linenos:

   >>> from roboticstoolbox import ET
   >>> ET.Rx(45, 'deg')
   >>> ET.tz(0.75)
   >>> e = ET.Rx(0.3) * ET.tz(0.75)
   >>> print(e)
   >>> e.fkine([])

Lines 2 and 3 each define one elementary transform: a rotation of 45° about the
x-axis, and a translation of 0.75m along the z-axis.  Multiplying two
elementary transforms, as in line 4, gives an elementary transform sequence (an
:class:`~roboticstoolbox.ets.ETS.ETS`) of length two.  An ETS can be
arbitrarily long.

Line 6 *evaluates* the forward kinematics of the sequence, substituting in
values, and the result is an :class:`~spatialmath.pose3d.SE3` object.

The real power comes from having variable arguments to the elementary transforms
as shown in this example where we define a simple two link planar manipulator.

.. runblock:: pycon
   :linenos:

   >>> from roboticstoolbox import ET
   >>> e = ET.Rz() * ET.tx(1) * ET.Rz() * ET.tx(1)
   >>> print(e)
   >>> len(e)
   >>> e[1:3]
   >>> e.fkine([0, 0])
   >>> e.fkine([1.57, -1.57])

Line 2 succinctly describes the kinematics in terms of elementary transforms: a
rotation around the z-axis by the first joint angle, then a translation in the
x-direction, then a rotation around the z-axis by the second joint angle, and
finally a translation in the x-direction.  A transform created without an
argument, like ``ET.Rz()``, is *variable*: it is a joint.

``e`` is a single object that encapsulates a list of elementary transforms, and
it has list-like methods such as ``len`` as well as indexing and slicing, as
shown in lines 4 and 5.  Lines 6 and 7 *evaluate* the sequence, substituting
elements from the passed array as the joint variables.

This approach is general enough to describe any serial-link robot manipulator.
For a branched manipulator we can use an ETS to describe the connection between
every parent and child link pair.

A joint is a variable transform, and the number of joints in the sequence is its
``n``.  Joint values are passed to the sequence as a vector, and a joint can
also be *flipped* (its sign reversed) and have *joint limits*.  The ETS joint
index rules, including the descriptor strings such as ``"q2"`` or ``"-theta3"``,
are described with :ref:`ets-descriptors`.

A sequence converts to a robot with :class:`~roboticstoolbox.robot.Robot.Robot`,
and a robot returns the sequence between two of its links with its ``ets``
method.  The inverse kinematics of a sequence are described on the
:ref:`IK <IK>` page.

**Reference:**

   - `A simple and systematic approach to assigning Denavit-Hartenberg parameters <https://petercorke.com/robotics/a-simple-and-systematic-approach-to-assigning-denavit-hartenberg-parameters>`_.
     Peter I. Corke, IEEE Transactions on Robotics, 23(3), pp 590-594, June 2007.

Transform chains
----------------

The :func:`~roboticstoolbox.tools.trchain.trchain` and
:func:`~roboticstoolbox.tools.trchain.trchain2` functions provide the compact
string notation used by the MATLAB Toolbox for one-off SE(3) and SE(2)
transform chains.

Three-dimensional chains contain rotations and translations about or along
any Cartesian axis.  Joint variables are numbered from one, and named
constants are passed explicitly in ``variables``.

.. runblock:: pycon

   >>> import roboticstoolbox as rtb
   >>> q = [0.2, -0.3, 0.4]
   >>> rtb.trchain("Rx(q1) Ry(q2) Rz(q3) Tx(x) Ty(y) Tz(z)", q,
   ...             variables={"x": 0.5, "y": 0.2, "z": 0.1})

Arguments can use arithmetic, the names ``pi``, ``sin``, ``cos``, ``tan`` and
``sqrt``, or functions supplied in ``variables``.  The joint prefix and angle
unit can also be changed.

.. runblock:: pycon

   >>> import roboticstoolbox as rtb
   >>> rtb.trchain("Rz(pi / 2) Tx(sqrt(4)) Ty(sin(pi / 2) + cos(0) + tan(0))")
   >>> double = lambda value: 2 * value
   >>> rtb.trchain("Rz(theta1 - 45) Tx(double(a))", [90], unit="deg",
   ...             qvar="theta", variables={"a": 0.5, "double": double})

For planar motion, :func:`~roboticstoolbox.tools.trchain.trchain2` accepts
``R`` (or ``Rz``), ``Tx`` and ``Ty`` transforms.

.. runblock:: pycon

   >>> import numpy as np
   >>> import roboticstoolbox as rtb
   >>> rtb.trchain2("R(q1) Tx(L1) Rz(q2) Tx(L2) Ty(y)",
   ...              [np.pi / 2, -np.pi / 2],
   ...              variables={"L1": 1, "L2": 0.5, "y": 0.25})

Parsed tokens can also be returned and reused when evaluating the same chain
for different joint values.

.. runblock:: pycon

   >>> import roboticstoolbox as rtb
   >>> _, tokens = rtb.trchain("Rz(q1) Tx(a)", [0], variables={"a": 1},
   ...                         return_tokens=True)
   >>> tokens
   >>> rtb.trchain(tokens, [0.5], variables={"a": 1})

Joint values and named constants can be symbolic.

.. runblock:: pycon

   >>> import roboticstoolbox as rtb
   >>> import sympy
   >>> q, a = sympy.symbols("q a", real=True)
   >>> T = rtb.trchain("Rz(q1) Tx(a)", [q], variables={"a": a})
   >>> T[0, 0], T[0, 3], T[1, 3]

.. autofunction:: roboticstoolbox.tools.trchain.trchain

.. autofunction:: roboticstoolbox.tools.trchain.trchain2
