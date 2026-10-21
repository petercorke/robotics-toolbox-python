Models
======

The Toolbox includes models of many commercial and research robots: classic arms such as the Puma 560, modern ones such as the Panda, UR and Kinova families, and a few mobile manipulators and humanoids.  Every model is a robot object, an instance of a subclass of :py:class:`~roboticstoolbox.robot.Robot.Robot`, so the same methods work for all of them, whichever way the model is defined.

The model of a robot can be defined in three ways, and the Toolbox has models of each kind:

.. list-table::
    :header-rows: 1
    :widths: 18 30 22 30

    * - Category
      - Defined by
      - Class
      - Good for
    * - ``DH``
      - the standard or modified Denavit-Hartenberg parameters of each link
      - :py:class:`~roboticstoolbox.robot.DHRobot.DHRobot`
      - serial-link arms: compact, fast, and the notation of many textbooks and papers
    * - ``ETS``
      - an elementary transform sequence, see :doc:`ets`
      - :py:class:`~roboticstoolbox.robot.Robot.Robot`
      - any structure, including branched robots, from a short expression
    * - ``URDF``
      - a URDF or xacro file
      - :py:class:`~roboticstoolbox.models.URDF.URDFRobot.URDFRobot`
      - the robot as its manufacturer describes it, with meshes for display and collision checking

Some robots are available in more than one category, for example the Panda:

.. runblock:: pycon

    >>> import roboticstoolbox as rtb
    >>> for kind in (rtb.models.DH, rtb.models.ETS, rtb.models.URDF):
    ...     robot = kind.Panda()
    ...     print(kind.__name__.split(".")[-1], type(robot).__name__, robot.n)

.. toctree::
   :maxdepth: 2
   :hidden:

   arm_dh
   arm_erobot
   arm_superclass

The classes and the models of each category are described on the pages below.  Models are in a sub-package named for the category, so ``rtb.models.DH.Puma560()`` and ``rtb.models.URDF.Panda()`` make a Puma 560 and a Panda.  The URDF models are also in the ``rtb.models`` namespace itself, so ``rtb.models.Panda`` is ``rtb.models.URDF.Panda``.

.. note::

    The dynamics methods, such as ``gravload``, ``inertia`` and ``rne``, need link masses and inertias.  The ``DH`` models define them, but the ``URDF`` and ``ETS`` models do not, and for these the methods raise a ``ValueError``.  ``robot.hasdynamics`` tells you whether a model has them.

Listing available models
------------------------

.. automodule:: roboticstoolbox.models.catalog
   :members:
