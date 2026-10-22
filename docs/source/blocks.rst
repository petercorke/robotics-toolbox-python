bdsim blocks
============

.. image:: https://raw.githubusercontent.com/petercorke/bdsim/main/docs/figs/bdsim_logo.png
    :width: 400

A set of block definitions that add robotic capability to the `bdsim <https://github.com/petercorke/bdsim>`_
block diagram simulation environment.

.. toctree::
   :maxdepth: 1
   :hidden:

   blocks-spatial
   blocks-arm
   blocks-mobile
   blocks-uav

.. rubric:: :doc:`Spatial maths blocks <blocks-spatial>`

.. autosummary::
    :nosignatures:

    ~roboticstoolbox.blocks.spatial.Tr2Delta
    ~roboticstoolbox.blocks.spatial.Delta2Tr
    ~roboticstoolbox.blocks.spatial.Point2Tr
    ~roboticstoolbox.blocks.spatial.TR2T

.. rubric:: :doc:`Robot manipulator blocks <blocks-arm>`

.. autosummary::
    :nosignatures:

    ~roboticstoolbox.blocks.arm.FKine
    ~roboticstoolbox.blocks.arm.IKine
    ~roboticstoolbox.blocks.arm.Jacobian
    ~roboticstoolbox.blocks.arm.ArmPlot
    ~roboticstoolbox.blocks.arm.JTraj
    ~roboticstoolbox.blocks.arm.CTraj
    ~roboticstoolbox.blocks.arm.CirclePath
    ~roboticstoolbox.blocks.arm.Trapezoidal
    ~roboticstoolbox.blocks.arm.Traj
    ~roboticstoolbox.blocks.arm.IDyn
    ~roboticstoolbox.blocks.arm.Gravload
    ~roboticstoolbox.blocks.arm.Gravload_X
    ~roboticstoolbox.blocks.arm.Inertia
    ~roboticstoolbox.blocks.arm.Inertia_X
    ~roboticstoolbox.blocks.arm.FDyn
    ~roboticstoolbox.blocks.arm.FDyn_X

.. rubric:: :doc:`Mobile robot blocks <blocks-mobile>`

.. autosummary::
    :nosignatures:

    ~roboticstoolbox.blocks.mobile.Bicycle
    ~roboticstoolbox.blocks.mobile.Unicycle
    ~roboticstoolbox.blocks.mobile.DiffSteer
    ~roboticstoolbox.blocks.mobile.VehiclePlot

.. rubric:: :doc:`UAV blocks <blocks-uav>`

.. autosummary::
    :nosignatures:

    ~roboticstoolbox.blocks.uav.MultiRotor
    ~roboticstoolbox.blocks.uav.MultiRotorMixer
    ~roboticstoolbox.blocks.uav.MultiRotorPlot
