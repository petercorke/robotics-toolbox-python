Denavit-Hartenberg models
=========================

.. codeauthor:: Jesse Haviland

This class is used to model robots defined in terms of standard or modified
Denavit-Hartenberg notation.

.. note:: These classes provide similar functionality and notation to MATLAB Toolbox ``SerialLink`` and
   ``Link`` classes.

DHRobot
-------

.. inheritance-diagram:: roboticstoolbox.DHRobot
   :top-classes: roboticstoolbox.Robot
   :parts: 2

The various :ref:`DH Models` all subclass this class.

.. autoclass:: roboticstoolbox.robot.DHRobot.DHRobot
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


.. autoclass:: roboticstoolbox.robot.DHLink
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

Revolute - standard DH
^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: roboticstoolbox.robot.DHLink.RevoluteDH
   :show-inheritance:

Prismatic - standard DH
^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: roboticstoolbox.robot.DHLink.PrismaticDH
   :show-inheritance:
   

Revolute - modified DH
^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: roboticstoolbox.robot.DHLink.RevoluteMDH
   :show-inheritance:
   
Prismatic - modified DH
^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: roboticstoolbox.robot.DHLink.PrismaticMDH
   :show-inheritance:


.. _DH Models:

Models
------

A number of models are defined in terms of Denavit-Hartenberg parameters, either
standard or modified.  They can be listed by:

.. runblock:: pycon

   >>> import roboticstoolbox as rtb 
   >>> rtb.models.list(mtype="DH")

.. automodule:: roboticstoolbox.models.DH
   :members:
   :undoc-members:
   :show-inheritance: