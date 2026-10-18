ERobot models
=============


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
   >>> rtb.models.catalog(mtype="ETS")

.. automodule:: roboticstoolbox.models.ETS
   :members:
   :undoc-members:
   :show-inheritance:

Defined from URDF
^^^^^^^^^^^^^^^^^

Any URDF or xacro file can be loaded with the class below, which is also the
superclass of the models.

.. autoclass:: roboticstoolbox.models.URDF.URDFRobot.URDFRobot
   :members:
   :undoc-members:
   :show-inheritance:

A number of models are defined by a URDF or xacro file.  They can be listed by:

.. runblock:: pycon

   >>> import roboticstoolbox as rtb 
   >>> rtb.models.catalog(mtype="URDF")

.. automodule:: roboticstoolbox.models.URDF
   :members:
   :undoc-members:
   :show-inheritance: