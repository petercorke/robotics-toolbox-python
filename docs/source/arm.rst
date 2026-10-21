**************************
Robot manipulators (arms)
**************************

A robot manipulator is an object of a subclass of :py:class:`~roboticstoolbox.robot.Robot.Robot`, which provides the methods in the tables below.  The subclass depends on how the robot is defined: from Denavit-Hartenberg parameters, from an elementary transform sequence, or from a URDF file.  The robots that the Toolbox provides ready made are described under :doc:`arm_models`.

.. toctree::
   :maxdepth: 4
   :hidden:

   arm_models
   IK/ik
   arm_trajectory
   arm_backend

.. rubric:: Construction

A robot is made in one of three ways, and each has its own class.

.. autosummary::
    :nosignatures:

    ~roboticstoolbox.robot.Robot.Robot
    ~roboticstoolbox.models.URDF.URDFRobot.URDFRobot
    ~roboticstoolbox.robot.DHRobot.DHRobot

A ``Robot`` is made from an :doc:`elementary transform sequence <ets>`, and can be a tree.  (``ERobot`` is a subclass with the same behaviour, kept for compatibility.)  A ``URDFRobot`` is made from a URDF or xacro file.  A ``DHRobot`` is made from a list of links, each with its Denavit-Hartenberg parameters, standard or modified, and a prismatic or revolute joint, using the link classes below.

.. autosummary::
    :nosignatures:

    ~roboticstoolbox.robot.DHLink.RevoluteDH
    ~roboticstoolbox.robot.DHLink.PrismaticDH
    ~roboticstoolbox.robot.DHLink.RevoluteMDH
    ~roboticstoolbox.robot.DHLink.PrismaticMDH

.. runblock:: pycon

    >>> import roboticstoolbox as rtb
    >>> from roboticstoolbox import DHRobot, Robot, RevoluteDH, ET
    >>> dh = DHRobot([RevoluteDH(a=1), RevoluteDH(a=1)], name="two link")  # from DH parameters
    >>> e = ET.Rz() * ET.tx(1) * ET.Rz() * ET.tx(1)
    >>> ets = Robot(e, name="two link")  # from an elementary transform sequence
    >>> dh.n, ets.n

.. autosummary::
    :nosignatures:

    ~roboticstoolbox.robot.Robot.Robot.copy
    ~roboticstoolbox.robot.Robot.Robot.addconfiguration

.. rubric:: Information

.. autosummary::
    :nosignatures:

    ~roboticstoolbox.robot.Robot.Robot.name
    ~roboticstoolbox.robot.Robot.Robot.manufacturer
    ~roboticstoolbox.robot.Robot.Robot.comment
    ~roboticstoolbox.robot.Robot.Robot.keywords
    ~roboticstoolbox.robot.Robot.Robot.n
    ~roboticstoolbox.robot.Robot.Robot.nlinks
    ~roboticstoolbox.robot.Robot.Robot.nbranches
    ~roboticstoolbox.robot.Robot.Robot.links
    ~roboticstoolbox.robot.Robot.Robot.structure
    ~roboticstoolbox.robot.Robot.Robot.isrevolute
    ~roboticstoolbox.robot.Robot.Robot.isprismatic
    ~roboticstoolbox.robot.Robot.Robot.revolutejoints
    ~roboticstoolbox.robot.Robot.Robot.prismaticjoints
    ~roboticstoolbox.robot.Robot.Robot.q
    ~roboticstoolbox.robot.Robot.Robot.qd
    ~roboticstoolbox.robot.Robot.Robot.qdd
    ~roboticstoolbox.robot.Robot.Robot.qlim
    ~roboticstoolbox.robot.Robot.Robot.random_q
    ~roboticstoolbox.robot.Robot.Robot.configs
    ~roboticstoolbox.robot.Robot.Robot.base
    ~roboticstoolbox.robot.Robot.Robot.tool
    ~roboticstoolbox.robot.Robot.Robot.gravity
    ~roboticstoolbox.robot.Robot.Robot.reach
    ~roboticstoolbox.robot.Robot.Robot.hasgeometry
    ~roboticstoolbox.robot.Robot.Robot.hascollision
    ~roboticstoolbox.robot.Robot.Robot.hasdynamics

.. rubric:: Visualization

.. autosummary::
    :nosignatures:

    ~roboticstoolbox.robot.Robot.Robot.plot
    ~roboticstoolbox.robot.Robot.Robot.teach
    ~roboticstoolbox.robot.Robot.Robot.plot_vellipse
    ~roboticstoolbox.robot.Robot.Robot.plot_fellipse
    ~roboticstoolbox.robot.Robot.Robot.vellipse
    ~roboticstoolbox.robot.Robot.Robot.fellipse
    ~roboticstoolbox.robot.Robot.Robot.showgraph
    ~roboticstoolbox.robot.Robot.Robot.default_backend
    ~roboticstoolbox.robot.Robot.Robot.set_default_backend

The graphical :doc:`backends <arm_backend>` do the drawing.

.. rubric:: Kinematics

.. autosummary::
    :nosignatures:

    ~roboticstoolbox.robot.Robot.Robot.fkine
    ~roboticstoolbox.robot.Robot.Robot.fkine_all
    ~roboticstoolbox.robot.Robot.Robot.jacob0
    ~roboticstoolbox.robot.Robot.Robot.jacobe
    ~roboticstoolbox.robot.Robot.Robot.jacob0_analytical
    ~roboticstoolbox.robot.Robot.Robot.jacob0_dot
    ~roboticstoolbox.robot.Robot.Robot.hessian0
    ~roboticstoolbox.robot.Robot.Robot.hessiane
    ~roboticstoolbox.robot.Robot.Robot.manipulability
    ~roboticstoolbox.robot.Robot.Robot.jacobm
    ~roboticstoolbox.robot.Robot.Robot.jtraj
    ~roboticstoolbox.robot.Robot.Robot.ikine_LM
    ~roboticstoolbox.robot.Robot.Robot.ik_LM

``ikine_LM`` and ``ik_LM`` are two examples: the first is written in Python, the second in C++ and is faster but has :ref:`limits <ik-cpp-limits>`.  All the inverse kinematics methods, ``ikine_XX`` and ``ik_XX``, are described under :doc:`Inverse Kinematics <IK/ik>`.

.. rubric:: Dynamics

.. autosummary::
    :nosignatures:

    ~roboticstoolbox.robot.Robot.Robot.rne
    ~roboticstoolbox.robot.Robot.Robot.gravload
    ~roboticstoolbox.robot.Robot.Robot.inertia
    ~roboticstoolbox.robot.Robot.Robot.coriolis
    ~roboticstoolbox.robot.Robot.Robot.accel
    ~roboticstoolbox.robot.Robot.Robot.fdyn
    ~roboticstoolbox.robot.Robot.Robot.itorque
    ~roboticstoolbox.robot.Robot.Robot.friction
    ~roboticstoolbox.robot.Robot.Robot.nofriction
    ~roboticstoolbox.robot.Robot.Robot.payload
    ~roboticstoolbox.robot.Robot.Robot.perturb

.. rubric:: Collision checking

.. autosummary::
    :nosignatures:

    ~roboticstoolbox.robot.Robot.Robot.iscollided
    ~roboticstoolbox.robot.Robot.Robot.closest_point

The ``Shape`` that these methods take is a class of the `spatialgeometry <https://github.com/jhavl/spatialgeometry>`_ package, such as ``Sphere``, ``Cuboid`` or ``Mesh``.  In the signatures it links to the documentation of that package.
