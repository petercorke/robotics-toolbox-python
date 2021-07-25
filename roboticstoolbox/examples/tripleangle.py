#!/usr/bin/env python
"""
@author Peter Corke
@author Jesse Haviland
"""

import swift
from math import pi
import roboticstoolbox as rtb
from spatialgeometry import Mesh
from spatialmath import SO3, SE3
import numpy as np


# TODO
#  rotate the rings according to the rotation axis, so that the axles
#  point the right way

# Launch the simulator Swift
env = swift.Swift()
env.launch()

path = rtb.rtb_path_to_datafile("data")


g1 = Mesh(
    filename=str(path / "gimbal-ring1.stl"), color=[34, 143, 201], scale=(1.0 / 3,) * 3
)

g2 = Mesh(
    filename=str(path / "gimbal-ring2.stl"), color=[31, 184, 72], scale=(1.1 / 3,) * 3
)

g3 = Mesh(
    filename=str(path / "gimbal-ring3.stl"),
    color=[240, 103, 103],
    scale=(1.1 ** 2 / 3,) * 3,
)

plane = Mesh(
    filename=str(path / "spitfire_assy-gear_up.stl"),
    scale=(1.0 / (180 * 3),) * 3,
    color=[240, 103, 103],
)
print(path / "spitfire_assy-gear_up.stl")
env.add(g1)
env.add(g2)
env.add(g3)
env.add(plane)

print("Supermarine Spitfire Mk VIII by Ed Morley @GRABCAD")
print("Gimbal models by Peter Corke using OpenSCAD")

# compute the three rotation matrices
BASE = SE3(0, 0, 0.5)
R1 = SO3()
R2 = SO3()
R3 = SO3()

# rotation angle sequence
sequence = "ZYX"


def update_gimbals(theta, ring):
    global R1, R2, R3

    # update the relevant transform, depending on which ring's slider moved
    def Rxyz(theta, which):
        theta = np.radians(theta)
        if which == "X":
            return SO3.Rx(theta)
        elif which == "Y":
            return SO3.Ry(theta)
        elif which == "Z":
            return SO3.Rz(theta)

    if ring == 1:
        R1 = Rxyz(theta, sequence[ring - 1])
    elif ring == 2:
        R2 = Rxyz(theta, sequence[ring - 1])
    elif ring == 3:
        R3 = Rxyz(theta, sequence[ring - 1])

    # figure the transforms for each gimbal and the plane, and update their
    # pose
    def convert(R):
        return BASE * SE3.Rt(R, t=[0, 0, 0])

    g3.base = convert(R1 * SO3.Ry(pi / 2))
    g2.base = convert(R1 * R2 * SO3.Rz(pi / 2))
    g1.base = convert(R1 * R2 * R3 * SO3.Rx(pi / 2))
    plane.base = convert(R1 * R2 * R3 * SO3.Ry(pi / 2) * SO3.Rz(pi / 2))


# slider call backs, invoke the central handler
def set_one(x):
    update_gimbals(float(x), 1)


def set_two(x):
    update_gimbals(float(x), 2)


def set_three(x):
    update_gimbals(float(x), 3)


r_one = swift.Slider(
    set_one, min=-180, max=180, step=1, value=0, desc="Outer gimbal", unit="&#176;"
)


r_two = swift.Slider(
    set_two, min=-180, max=180, step=1, value=0, desc="Middle gimbal", unit="&#176;"
)


r_three = swift.Slider(
    set_three, min=-180, max=180, step=1, value=0, desc="Inner gimbal", unit="&#176;"
)


# buttons to set a 3-angle sequence
ZYX_button = swift.Button(
    lambda x: change_sequence("ZYX"), desc="ZYX (roll-pitch-yaw angles)"
)

XYZ_button = swift.Button(
    lambda x: change_sequence("XYZ"), desc="XYZ (roll-pitch-yaw angles)"
)

ZYZ_button = swift.Button(lambda x: change_sequence("ZYZ"), desc="ZYZ (Euler angles)")

button = swift.Button(lambda x: set("ZYX"), desc="Set to Zero")


# button to reset joint angles
def reset(e):
    r_one.value = 0
    r_two.value = 0
    r_three.value = 0
    # env.step(0)


zero_button = swift.Button(reset, desc="Set to Zero")


def update_all_sliders():
    update_gimbals(float(r_one.value), 1)
    update_gimbals(float(r_two.value), 2)
    update_gimbals(float(r_three.value), 3)


def change_sequence(new):
    global sequence

    xyz = "XYZ"

    # update the state of the ring_axis dropdowns
    ring1_axis.checked = xyz.find(new[0])
    ring2_axis.checked = xyz.find(new[1])
    ring3_axis.checked = xyz.find(new[2])

    sequence = new
    update_all_sliders()


# handle radio button on angle slider
def angle(index, ring):
    global sequence

    # print('angle', index, ring)
    xyz = "XYZ"
    s = list(sequence)
    s[ring] = xyz[int(index)]
    sequence = "".join(s)
    update_all_sliders()


ring1_axis = swift.Radio(lambda x: angle(x, 0), options=["X", "Y", "Z"], checked=2)

ring2_axis = swift.Radio(lambda x: angle(x, 1), options=["X", "Y", "Z"], checked=1)

ring3_axis = swift.Radio(lambda x: angle(x, 2), options=["X", "Y", "Z"], checked=0)


label = swift.Label(desc="Triple angle")


def chekked(e, el):
    nlabel = "s: "

    if e[0]:
        nlabel += "a"
        r_one.value = 0

    if e[1]:
        nlabel += "b"
        r_two.value = 0

    if e[2]:
        nlabel += "c"
        r_three.value = 0

    if e[3]:
        el.value = 1

    label.desc = nlabel


env.add(label)
env.add(r_one)
env.add(ring1_axis)

env.add(r_two)
env.add(ring2_axis)

env.add(r_three)
env.add(ring3_axis)

env.add(ZYX_button)
env.add(XYZ_button)
env.add(ZYZ_button)
env.add(zero_button)

update_gimbals(0, 1)
update_gimbals(0, 2)
update_gimbals(0, 3)

while True:
    env.step(0)
