# Created by: Aditya Dua
# 18 August 2017
import pkg_resources
import vtk
import math
import numpy as np


class VtkPipeline:
    def __init__(self, background=(0.15, 0.15, 0.15), total_time_steps=None, timer_rate=60, gif_file=None):
        self.ren = vtk.vtkRenderer()
        self.ren.SetBackground(background[0], background[1], background[2])
        self.ren_win = vtk.vtkRenderWindow()
        self.ren_win.AddRenderer(self.ren)
        self.iren = vtk.vtkRenderWindowInteractor()
        self.iren.SetRenderWindow(self.ren_win)
        self.iren.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
        self.actor_list = []
        self.mapper_list = []
        self.source_list = []
        self.screenshot_count = 0
        self.timer_rate = timer_rate
        self.gif_data = []
        if gif_file is not None:
            try:
                assert type(gif_file) is str
            except AssertionError:
                gif_file = str(gif_file)
            self.gif_file = gif_file
        else:
            self.gif_file = None

        if total_time_steps is not None:
            assert type(total_time_steps) is int
            self.timer_count = 0
            self.total_time_steps = total_time_steps

    def render(self, ui=True):
        for each in self.actor_list:
            self.ren.AddActor(each)
        self.ren.ResetCamera()
        self.ren_win.Render()
        if ui:
            self.iren.Initialize()
            self.iren.Start()

    def add_actor(self, actor):
        self.actor_list.append(actor)

    def set_camera(self):
        cam = self.ren.GetActiveCamera()
        cam.Roll(-90)
        cam.Elevation(-90)
        cam.Zoom(0.6)

    def animate(self):
        self.ren.ResetCamera()
        self.ren_win.Render()
        self.iren.Initialize()
        self.iren.CreateRepeatingTimer(math.floor(1000 / self.timer_rate))  # Timer creates 60 fps
        self.render()

    def screenshot(self, filename=None):
        w2if = vtk.vtkWindowToImageFilter()
        w2if.SetInput(self.ren_win)
        w2if.Update()
        if filename is None:
            filename = 'screenshot'
        filename = filename + '%d.png' % self.screenshot_count
        writer = vtk.vtkPNGWriter()
        writer.SetFileName(filename)
        self.screenshot_count += 1
        writer.SetInputData(w2if.GetOutput())
        writer.Write()

    def timer_tick(self):
        import imageio
        self.timer_count += 1

        if self.timer_count >= self.total_time_steps:
            self.iren.DestroyTimer()
            if self.gif_file is not None:
                assert len(self.gif_data) > 0
                imageio.mimsave(self.gif_file + '.gif', self.gif_data)
                import os
                for i in range(self.screenshot_count):
                    os.remove(self.gif_file + '%d.png' % i)
                return

        if self.gif_file is not None:
            if (self.timer_count % 60) == 0:
                self.screenshot(self.gif_file)
                path = self.gif_file + '%d.png' % (self.screenshot_count - 1)
                self.gif_data.append(imageio.imread(path))


def axesUniversal():
    axes_uni = vtk.vtkAxesActor()
    axes_uni.SetXAxisLabelText("x'")
    axes_uni.SetYAxisLabelText("y'")
    axes_uni.SetZAxisLabelText("z'")
    axes_uni.SetTipTypeToSphere()
    axes_uni.SetShaftTypeToCylinder()
    axes_uni.SetTotalLength(2, 2, 2)
    axes_uni.SetCylinderRadius(0.02)
    axes_uni.SetAxisLabels(0)

    return axes_uni


def axesCube(ren, x_bound=np.matrix([[-1.5, 1.5]]), y_bound=np.matrix([[-1.5, 1.5]]), z_bound=np.matrix([[-1.5, 1.5]])):
    cube_axes_actor = vtk.vtkCubeAxesActor()
    cube_axes_actor.SetBounds(x_bound[0, 0], x_bound[0, 1], y_bound[0, 0], y_bound[0, 1], z_bound[0, 0], z_bound[0, 1])
    cube_axes_actor.SetCamera(ren.GetActiveCamera())
    cube_axes_actor.GetTitleTextProperty(0).SetColor(1.0, 0.0, 0.0)
    cube_axes_actor.GetLabelTextProperty(0).SetColor(1.0, 0.0, 0.0)

    cube_axes_actor.GetTitleTextProperty(1).SetColor(0.0, 1.0, 0.0)
    cube_axes_actor.GetLabelTextProperty(1).SetColor(0.0, 1.0, 0.0)

    cube_axes_actor.GetTitleTextProperty(2).SetColor(0.0, 0.0, 1.0)
    cube_axes_actor.GetLabelTextProperty(2).SetColor(0.0, 0.0, 1.0)

    cube_axes_actor.XAxisMinorTickVisibilityOff()
    cube_axes_actor.YAxisMinorTickVisibilityOff()
    cube_axes_actor.ZAxisMinorTickVisibilityOff()

    cube_axes_actor.SetFlyModeToStaticTriad()

    return cube_axes_actor


def axes_x_y(ren):
    axis_x_y = axesCube(ren)
    axis_x_y.SetUse2DMode(1)
    axis_x_y.ZAxisLabelVisibilityOff()
    axis_x_y.SetAxisOrigin(-3, -3, 0)
    axis_x_y.SetUseAxisOrigin(1)

    return axis_x_y


def axesActor2d():
    axes = vtk.vtkAxesActor()
    axes.SetTotalLength(1, 1, 0)
    axes.SetZAxisLabelText("")

    return axes


def vtk_named_colors(colors):
    """
    Returns a list of vtk colors
    :param colors: List of color names supported by vtk
    :return: A list of vtk colors
    """
    if type(colors) is not list:
        colors = [colors]
    colors_rgb = [0] * len(colors)
    for i in range(len(colors)):
        colors_rgb[i] = list(vtk.vtkNamedColors().GetColor3d(colors[i]))
    return colors_rgb


def floor():
    plane = vtk.vtkPlaneSource()
    reader = vtk.vtkJPEGReader()
    reader.SetFileName(pkg_resources.resource_filename("robopy", "media/imgs/floor.jpg"))
    texture = vtk.vtkTexture()
    texture.SetInputConnection(reader.GetOutputPort())
    map_to_plane = vtk.vtkTextureMapToPlane()
    map_to_plane.SetInputConnection(plane.GetOutputPort())
    mapper = vtk.vtkPolyDataMapper()

    mapper.SetInputConnection(map_to_plane.GetOutputPort())
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.SetTexture(texture)
    return actor


def axesCubeFloor(ren, x_bound=np.matrix([[-1.5, 1.5]]),
                  y_bound=np.matrix([[-1.5, 1.5]]),
                  z_bound=np.matrix([[-1.5, 1.5]]),
                  position=np.matrix([[0, -1.5, 0]])):
    axes = axesCube(ren, x_bound=x_bound, y_bound=y_bound, z_bound=z_bound)
    flr = floor()
    flr.RotateX(90)
    flr.SetPosition(position[0, 0], position[0, 1], position[0, 2])
    flr.SetScale(3)
    assembly = vtk.vtkAssembly()
    assembly.AddPart(flr)
    assembly.AddPart(axes)
    return assembly
