Elementary transform sequences
==============================

Elementary transforms are canonic rotations or translations about, or along,
the x-, y- or z-axes.  The angle of a rotation, or the distance of a translation,
can be a constant or a variable.  Variable rotation or translation corresponds to a joint.  A sequence of them
is an *elementary transform sequence* (ETS), and describes the kinematics of a
serial-link robot arm, or of one link of a branched manipulator.

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Work space
     - Sequence
     - Element
   * - :doc:`3D <ets_3d>`
     - :class:`~roboticstoolbox.ETS`
     - :class:`~roboticstoolbox.ET`, an :math:`\SE{3}` transform
   * - :doc:`2D <ets_2d>`
     - :class:`~roboticstoolbox.ETS2`
     - :class:`~roboticstoolbox.ET2`, an :math:`\SE{2}` transform


Introduction
------------

Consider the example:

.. runblock:: pycon
   :linenos:
   :session: eg1

   >>> from roboticstoolbox import ET
   >>> e1 = ET.Rx(45, 'deg')
   >>> print(e1)
   >>> e2 = ET.tz(0.75)
   >>> print(e2)

Lines 2 and 5 each define one elementary transform: a rotation of 45° about the
x-axis, and a translation of 0.75m along the z-axis.

Multiplying elementary transforms

.. runblock:: pycon
   :linenos:
   :session: eg1

   >>> e = e1 * e2 * ET.tx(0.5) * ET.Ry(90, 'deg')
   >>> print(e)


gives an elementary transform sequence (an
:class:`~roboticstoolbox.ETS`).

Now we can *evaluate* the forward kinematics of the sequence

.. runblock:: pycon
   :linenos:
   :session: eg1

   >>> e.fkine()

and the result is an :class:`~spatialmath.pose3d.SE3` object.

An ETS is a list-like object that can be indexed, sliced and iterated over

.. runblock:: pycon
   :linenos:
   :session: eg1

   >>> print(len(e))
   >>> print(e[1])
   >>> print(e[1:3])
   >>> type(e[1]), type(e[1:3])
   >>> print(e2, e[1])
   >>> for x in e:
   ...     print(x)

If the slice is a single element then it is an :class:`~roboticstoolbox.ET`, otherwise it is an :class:`~roboticstoolbox.ETS`. An :class:`~roboticstoolbox.ETS` can be arbitrarily long.
We can multiply an :class:`~roboticstoolbox.ETS` by an elementary transform, or by another :class:`~roboticstoolbox.ETS`, and the result is
always an :class:`~roboticstoolbox.ETS`.

.. runblock:: pycon
   :linenos:
   :session: eg1

   >>> type(e1), type(e2)
   >>> e12 = e1 * e2
   >>> type(e12)
   >>> type(e12 * e1), type(e1 * e12), type(e12 * e12)


The real power comes from having variable transforms -- elementary transforms
with variable arguments. In this next example, we use variable transforms
to define a simple two link planar manipulator.

.. runblock:: pycon
   :linenos:
   :session: eg2

   >>> from roboticstoolbox import ET
   >>> e = ET.Rz() * ET.tx(1) * ET.Rz() * ET.tx(1)
   >>> print(e)

Line 2 succinctly describes the kinematics in terms of elementary transforms: a
rotation around the z-axis by the first joint angle, then a translation in the
x-direction, then a rotation around the z-axis by the second joint angle, and
finally a translation in the x-direction.  A transform created without an
argument, like ``ET.Rz()``, is *variable*: it is a joint.

This approach is general enough to describe any serial-link robot manipulator.
Compared to Denavit-Hartenberg parameters, this is a more natural way to describe the
kinematics of a manipulator, and allows joint motion about, or along, any Cartesian axis.
For a branched manipulator we can use an ETS to describe the connection between
every parent and child link pair.

.. runblock:: pycon
   :linenos:
   :session: eg2

   >>> e.fkine([0, 0])
   >>> from math import pi
   >>> e.fkine([pi/2, -pi/2])

The sequence can be drawn, here for the second set of joint angles above.  The
``plot`` method wraps the sequence in a robot, and plots that.

.. plot::
   :include-source:

   from math import pi
   from roboticstoolbox import ET

   e = ET.Rz() * ET.tx(1) * ET.Rz() * ET.tx(1)
   e.plot([pi/2, -pi/2], block=False)

A joint is a variable transform, and the number of joints in the sequence is its
``n``.  Joint values are passed to the sequence as a vector, and a joint can
also be *flipped* (its sign reversed) and have *joint limits*.  The ETS joint
index rules, including the descriptor strings such as ``"q2"`` or ``"-theta3"``,
are described with :ref:`ets-descriptors`.

A sequence converts to a robot with :class:`~roboticstoolbox.robot.Robot.Robot`,
and a robot returns the sequence between two of its links with its ``ets``
method.  The inverse kinematics of a sequence are described on the
:ref:`IK <IK>` page.

Inverse kinematics
-------------------

The other very common kinematics operation is inverse kinematics (IK) and there
are several options.

Using C++ solvers, fast but not extensible.  They need the compiled extension, so they do not work in the browser (Pyodide/JupyterLite), and they cannot be used for a sequence with SymPy parameters or joint offsets, see :ref:`ik-cpp-limits`

.. runblock:: pycon
   :linenos:
   :session: eg2

   >>> T = e.fkine([0.2, 0.4])
   >>> sol = e.ik_LM(T)
   >>> print(sol)

The return value is an :class:`~roboticstoolbox.robot.IK.IKSolution` object, which contains the solution, the number of iterations, and a success flag.  The solution can be evaluated with the ``fkine`` method to check that it is correct.

Using Python solvers, based on NumPy, slower but extensible, and they work everywhere, including the browser

.. runblock:: pycon
   :linenos:
   :session: eg2

   >>> sol = e.ikine_LM(T)
   >>> print(sol)

or directly with a solver object, which takes the ETS as an argument.

.. runblock:: pycon
   :linenos:
   :session: eg2

   >>> from roboticstoolbox import IK_LM
   >>> solver = IK_LM()
   >>> sol = solver.solve(e, T)
   >>> print(sol)

The solver object can be configured with various options and then used, without setup overhead,
to solve many different problems. Line 20 is equivalent to lines 24-25.

Note that for all numerical solvers, the solution is randomly seeded and could converge to any of the
valid kinematic solutions. In general, IK has no unique solution, and there may be no solution
if the pose is out of reach.  There are many different IK solvers available, and only the QP solver
can handle joint limits.  The initial guess for the solution,
can be set with the ``q0`` argument, and can be used to bias the solution towards a particular kinematic configuration.

**Reference:**

   - `A simple and systematic approach to assigning Denavit-Hartenberg parameters <https://petercorke.com/robotics/a-simple-and-systematic-approach-to-assigning-denavit-hartenberg-parameters>`_.
     Peter I. Corke, IEEE Transactions on Robotics, 23(3), pp 590-594, June 2007.

In 2D
-------

Our examples have been in 3D, even though the robots are planar.  The :class:`~roboticstoolbox.ET` subclasses can express all elementary
translations and rotation in 3D space, and the result of the forward kinematics is a 3D pose. The 3D ET and ETS are described on the :doc:`ets_3d` page.

A similar approach works in 2D.  There are fewer elementary transforms, just translation along the x- andy y-axes and rotation about
the z-axis.  The result of the forward kinematics is a 2D pose. The 2D ET and ETS are described on the :doc:`ets_2d` page.

Advanced usage
---------------

In the example above the joint variables are internally indexed starting from 0

.. runblock:: pycon
   :linenos:
   :session: eg2

   >>> print(e)
   >>> print(e.joints())
   >>> print(e[0].jindex, e[2].jindex)

In the forward kinematics the elements of the joint vector are assigned to the variable transforms according to the ``jindex``, ie.
``q[0]`` is assigned to the variable transform with ``jindex`` of 0, ``q[1]`` is assigned to the variable transform with ``jindex`` of 1, and so on.

The joint index can be set explicitly, and this is useful for branched manipulators where the joint
indices for a particular kinematic chain may not be consecutive.  

For example, for the two link planar manipulator above, we can set the joint indices to
5 and 3, and then the joint vector passed to ``fkine`` must have the first joint angle at index 5,
and the second element at index 3.

.. runblock:: pycon
   :linenos:

   >>> from roboticstoolbox import ET
   >>> from math import pi
   >>> e = ET.Rz(jindex=5) * ET.tx(1) * ET.Rz(jindex=3) * ET.tx(1)
   >>> print(e)
   >>> print(e.joints())
   >>> print(e[0].jindex, e[2].jindex)
   >>> print(e.jindices)
   >>> print(e.fkine([0, 0, 0, -pi/2, 0, pi/2]))

Note that by default, the string representation of an ETS, uses the "q" as the joint variable prefix, 
followed by the joint index (``jindex``).


However, the joint variable can be described by a string such as "θ5" or "q3"

.. runblock:: pycon
   :linenos:

   >>> from roboticstoolbox import ET
   >>> from math import pi
   >>> e = ET.Rz("θ5") * ET.tx(1) * ET.Rz("θ3") * ET.tx(1)
   >>> print(e)
   >>> print(e.joints())

which is the same ETS as the example above.

If we were describing a robot using Kuka joint variable names, which are A1, A2,
A3, A4, A5, A6 -- they are one-based.  We append an underscore to the name to indicate that it is one-based
indexing which should be decremented to zero-based indexing for the internal representation.

.. runblock:: pycon
   :linenos:

   >>> from roboticstoolbox import ET
   >>> e = ET.Rz("A1_") * ET.tx(1) * ET.Rz("A2_") * ET.tx(1)
   >>> print(e)
   >>> print(e[0].jindex, e[2].jindex)
   >>> e.fkine([0, 0])

Sometimes we want the joint variable to represented rotation or translation in the
opposite sense, for example if the joint is physically reversed.  In this case, we can
set the ``flip`` argument to ``True`` when creating the elementary transform, or we can use a negative sign in the joint variable string,
for example "-q2" or "-θ3". 

The ETS package fully supports symbolic computation using SymPy. 
Here we define the two joint angles and the two link lengths as symbolic variables

.. runblock:: pycon
   :linenos:
   :session: eg4

   >>> from roboticstoolbox import ET
   >>> import sympy
   >>> q0, q1, L1, L2 = sympy.symbols("q0 q1 L1 L2", real=True)
   >>> e = ET.Rz() * ET.tx(L1) * ET.Rz() * ET.tx(L2)
   >>> print(e)

which shows the ETS with symbolic variables for the displacements and the place holders for the joint variables.  
Now, we can evaluate the forward kinematics with symbolic joint angles

.. runblock:: pycon
   :linenos:
   :session: eg4

   >>> print(e.fkine([q0, q1]))

Such expressions can be then be further manipulated using SymPy, for example symbolically differentiation
to obtain symbolic Jacobians, or LaTeX expressions for documentation,
or code generation in a wide range of programming languages.


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


Full API
-------------

.. toctree::
   :maxdepth: 2

   ets_3d
   ets_2d