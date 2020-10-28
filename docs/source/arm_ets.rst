Elementary transform sequence (ETS) models
==========================================

.. codeauthor:: Jesse Haviland

A number of models are defined in terms of elementary transform sequences.  
They can be listed by:

.. runblock:: pycon

   >>> import roboticstoolbox as rtb 
   >>> rtb.models.list(mtype="ETS")

:references:

   - https://petercorke.com/robotics/a-simple-and-systematic-approach-to-assigning-denavit-hartenberg-parameters/

ETS - 3D
--------
.. automodule:: roboticstoolbox.robot.ETS
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :special-members:
   :exclude-members: count, index, sort, remove, __dict__, __weakref__, __add__, __init__, __repr__, __str__, __module__

ETS - 2D
--------
.. automodule:: roboticstoolbox.robot.ETS2
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :special-members:
   :exclude-members: count, index, sort, remove, __dict__, __weakref__, __add__, __init__, __repr__, __str__, __module__

ET
------------
.. automodule:: roboticstoolbox.robot.ET
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :special-members:
   :exclude-members: count, index, sort, remove, __dict__, __weakref__, __add__, __init__, __repr__, __str__, __module__
