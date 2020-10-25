Arm Type Robots - DH
====================

.. codeauthor:: Jesse Haviland

A number of models are defined in terms of Denavit-Hartenberg parameters, either
standard or modified.  They can be listed by:

.. runblock:: pycon

   >>> import roboticstoolbox as rtb 
   >>> rtb.models.list(mtype="DH")

DHRobot
-------

.. inheritance-diagram:: roboticstoolbox.DHRobot
   :top-classes: roboticstoolbox.Robot
   :parts: 2

The various :ref:`DH Models` all subclass this class.

.. automodule:: roboticstoolbox.robot.DHRobot
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   
DHLink
------

The ``DHRobot`` is defined by a list of ``DHLink`` subclass objects.

.. inheritance-diagram:: roboticstoolbox.RevoluteDH roboticstoolbox.PrismaticDH roboticstoolbox.RevoluteMDH roboticstoolbox.PrismaticMDH
   :top-classes: roboticstoolbox.robot.Link
   :parts: 2


.. automodule:: roboticstoolbox.robot.DHLink
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

Revolute - standard DH
^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: roboticstoolbox.robot.RevoluteDH
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

Prismatic - standard DH
^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: roboticstoolbox.robot.PrismaticDH
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   

Revolute - modified DH
^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: roboticstoolbox.robot.RevoluteMDH
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   
Prismatic - modified DH
^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: roboticstoolbox.robot.PrismaticMDH
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:


.. _DH Models:

Models
------

.. automodule:: roboticstoolbox.models.DH
   :members:
   :undoc-members:
   :show-inheritance: