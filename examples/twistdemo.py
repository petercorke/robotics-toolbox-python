#!/usr/bin/env python
"""
@author Peter Corke
@author Jesse Haviland
"""
import sys
import swift
from math import pi
import roboticstoolbox as rtb
from spatialgeometry import Mesh, Cylinder
from spatialmath import SO3, SE3, Twist3
import numpy as np


# TODO
#  rotate the rings according to the rotation axis, so that the axles
#  point the right way

# Launch the simulator Swift
env = swift.Swift()
env.launch()

path = rtb.rtb_path_to_datafile("data")

plane = Mesh(
    filename=str(path / "spitfire_assy-gear_up.stl"),
    scale=(1.0 / (180 * 3),) * 3,
    color=[240, 103, 103],
)
print("Supermarine Spitfire Mk VIII by Ed Morley @GRABCAD")

screw = Cylinder(radius=0.015, length=4, color='blue', collision=False)

BASE = SE3(0.15, 0, 0.1) * SE3.Rz(pi/2) * SE3.Rx(np.pi/2)

env.add(plane)
env.add(screw)


twist_params = {'x':0, 'y': 0, 'R': 0, 'P':0, 'p': 0.1}
twist = Twist3()

def update_twist_param(value, key):
    global twist_params

    twist_params[key] = value

    update_twist()

def update_twist():
    global twist

    axis = SO3.Rx(twist_params['R']) @ SO3.Ry(twist_params['P'])
    point = [twist_params['x'], twist_params['y'], 0]
    twist = Twist3.UnitRevolute(axis.A[:, 2], point, twist_params['p'])

    update_screw_axis(axis, point)
    update_plane()

# button to reset twist
def reset_twist(e):
    global twist_params

    twist_params['x'] = 0
    twist_params['y'] = 0
    twist_params['R'] = 0
    twist_params['P'] = 0
    twist_params['p'] = 0
    update_all_sliders()

    update_twist()
    update_plane()
    # env.step(0)

def update_plane(*args):

    T = twist.SE3(twist_theta.value) * BASE
    plane.T = T

def update_screw_axis(axis, point):
    screw.T = SE3.Rt(axis, point)

def update_all_sliders():
    twist_x.value = twist_params['x']
    twist_y.value = twist_params['y']
    twist_rollangle.value = twist_params['R']
    twist_pitchangle.value = twist_params['P']
    twist_pitch.value = twist_params['p']


twist_x = swift.Slider(
    lambda x: update_twist_param(x, 'x'), min=-1, max=1, step=0.01, value=0, desc="x position"
)

twist_y = swift.Slider(
    lambda y: update_twist_param(y, 'y'), min=-1, max=1, step=0.01, value=0, desc="y position"
)

twist_rollangle = swift.Slider(
    lambda r: update_twist_param(np.deg2rad(r), 'R'), min=-180, max=180, step=1, value=0, desc="roll angle (rx)", unit="&#176;"
)

twist_pitchangle = swift.Slider(
    lambda p: update_twist_param(np.deg2rad(p), 'P'), min=-180, max=180, step=1, value=0, desc="pitch angle (ry)", unit="&#176;"
)

twist_pitch = swift.Slider(
    lambda p: update_twist_param(p, 'p'), min=0, max=1, step=0.02, value=0, desc="screw pitch"
)

twist_theta = swift.Slider(
    update_plane, min=0, max=10, step=0.02, value=0, desc="Twist rotation")

button = swift.Button(reset_twist, desc="Set to Zero")
quit = swift.Button(lambda x: sys.exit(0), desc="Quit")

label1 = swift.Label(desc="Twist parameters")
label2 = swift.Label(desc="Twist application")

env.add(label1)
env.add(twist_x)
env.add(twist_y)
env.add(twist_rollangle)
env.add(twist_pitchangle)
env.add(twist_pitch)
env.add(button)

env.add(label2)

env.add(twist_theta)

env.add(quit)


update_all_sliders()
while True:
    env.step(0)
