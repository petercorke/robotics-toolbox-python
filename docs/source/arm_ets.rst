Elementary transform sequence (ETS) models
==========================================

.. codeauthor:: Jesse Haviland

Elementary transforms are canonic rotations or translations about, or along,
the x-, y- or z-axes.  The amount of rotation or translation can be a constant
or a variable.  A variable amount corresponds to a joint.

Consider the example:

.. runblock:: pycon
   :linenos:

   >>> from roboticstoolbox import ETS
   >>> ETS.rx(45, 'deg')
   >>> ETS.tz(0.75)
   >>> e = ETS.rx(0.3) * ETS.tz(0.75)
   >>> print(e)
   >>> e.eval()

In lines 2-5 we defined two elementary transforms.  Line 2 defines a rotation
of 45° about the x-axis, and line 4 defines a translation of 0.75m along the z-axis.
Compounding them in line 6, the result is the two elementary transforms in a
sequence - an elementary transform sequence (ETS).  An ETS can be arbitrarily
long.

Line 8 *evaluates* the sequence, substituting in values, and the result is an
SE(3) matrix encapsulated in an ``SE3`` object.

The real power comes from having variable arguments to the elementary transforms
as shown in this example where we define a simple two link planar manipulator.


.. runblock:: pycon
   :linenos:

   >>> from roboticstoolbox import ETS
   >>> e = ETS.rz() * ETS.tx(1) * ETS.rz() * ETS.tx(1)
   >>> print(e)
   >>> len(e)
   >>> e[1:3]
   >>> e.eval([0, 0])
   >>> e.eval([90, -90], 'deg')

Line 2 succinctly describes the kinematics in terms of elementary transforms: a rotation around the z-axis by the
first joint angle, then a translation in the x-direction, then a rotation around
the z-axis by the second joint angle, and finally a translation in the
x-direction.

Line 3 creates the elementary transform sequence, with variable transforms.
``e`` is a single object that encapsulates a list of elementary transforms, and list like 
methods such as ``len`` as well as indexing and slicing as shown in lines 5-8.

Lines 9-18 *evaluate* the sequence, and substitutes elements from the passed
arrays as the joint variables.

This approach is general enough to be able to describe any serial-link robot
manipulator.  For a branched manipulator we can use ETS to describe the
connections between every parent and child link pair.

The ETS inherits list-like properties and has methods like ``reverse`` and ``pop``.

**Reference:**

   - `A simple and systematic approach to assigning Denavit-Hartenberg parameters <https://petercorke.com/robotics/a-simple-and-systematic-approach-to-assigning-denavit-hartenberg-parameters>`_.
     Peter I. Corke, IEEE Transactions on Robotics, 23(3), pp 590-594, June 2007.

ETS - 3D
--------

.. autoclass:: roboticstoolbox.robot.ETS.ETS
   :members: rx, ry, rz, tx, ty, tz, eta, n, joints, isjoint, isconstant, isrevolute, isprismatic, axis, config, __getitem__, pop, __mul__, __repr__, __str__, T, eval, compile, SE3, jacob0, hessian0
   :undoc-members:
   :show-inheritance:

ETS - 2D
--------

.. autoclass:: roboticstoolbox.robot.ETS.ETS2
   :members: r, tx, ty, eta, n, joints, isjoint, isconstant, isrevolute, isprismatic, axis, config, __getitem__, pop, __mul__, __repr__, __str__, T, eval, compile
   :undoc-members:
   :show-inheritance:

