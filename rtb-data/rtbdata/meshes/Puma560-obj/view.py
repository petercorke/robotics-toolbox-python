import pyvista as pv
import numpy as np

plotter = pv.Plotter(
    shape=(2, 3), border=False, polygon_smoothing=True, window_size=(2000, 1000)
)
plotter.set_background("white")
plotter.enable_parallel_projection()

# link = pv.read('pieza3.obj')
# link = pv.read('../../xacro/franka_description/meshes/visual/link0.dae')
link = pv.read("../puma_560/link0.stl")
plotter.add_mesh(link)

plotter.show()
