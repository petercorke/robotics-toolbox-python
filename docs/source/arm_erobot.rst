ERobot models
=============

.. codeauthor:: Jesse Haviland


ERobot
------

.. inheritance-diagram:: roboticstoolbox.ERobot
   :top-classes: roboticstoolbox.Robot
   :parts: 2

The various models :ref:`E Models` all subclass this class.

.. automodule:: roboticstoolbox.robot.ERobot
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   
ELink
-------

The ``ERobot`` is defined by a tree of ``ELink`` subclass objects.

.. inheritance-diagram:: roboticstoolbox.ELink
   :top-classes: roboticstoolbox.robot.Link
   :parts: 2

.. automodule:: roboticstoolbox.robot.ELink
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:


.. _E Models:

ERobot models
-------------

Defined using ETS
^^^^^^^^^^^^^^^^^

A number of models are defined in terms of elementary transform sequences.  
They can be listed by:

.. runblock:: pycon

   >>> import roboticstoolbox as rtb 
   >>> rtb.models.list(mtype="ETS")

.. automodule:: roboticstoolbox.models.ETS
   :members:
   :undoc-members:
   :show-inheritance:

Defined from URDF
^^^^^^^^^^^^^^^^^

A number of models are defined in terms of Denavit-Hartenberg parameters, either
standard or modified.  They can be listed by:

.. runblock:: pycon

   >>> import roboticstoolbox as rtb 
   >>> rtb.models.list(mtype="URDF")

.. automodule:: roboticstoolbox.models.URDF
   :members:
   :undoc-members:
   :show-inheritance: