There are three types of model supported by the Toolbox, and all are subclasses of the abstract `Robot` superclass.

* [Denavit-Hartenberg (DH) models](roboticstoolbox/models/DH). These are defined using standard or modified DH parameters, with optional 3D meshes for visualisation and optional dynamic parameters.
* [ETS models](roboticstoolbox/models/ETS). There are defined using a sequence of elementary transformations (rotation and translations), and is a quick and intuitive way to describe a robot, see [this article](https://petercorke.com/robotics/a-simple-and-systematic-approach-to-assigning-denavit-hartenberg-parameters/).
* [URDF models](roboticstoolbox/models/URDF). These models are defined by a Unified Robot Description Format file, an XML format file. Models exist for the classic Puma560 robot as well as the Franka-Emika Panda, the Universal robotics range and all the Interbotix hobby-class robots.

In each folder you will find a README describing how to create your own model.

If you think your model might be interesting to others consider submitting a pull request.