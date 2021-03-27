# #!/usr/bin/env python
# """
# @author Jesse Haviland
# """

from math import pi
import roboticstoolbox as rtb
from spatialmath import SO3, SE3
import numpy as np
import pathlib
import os

path = os.path.realpath('.')


# TODO
#  rotate the rings according to the rotation axis, so that the axles 
#  point the right way

# Launch the simulator Swift
from roboticstoolbox.backends import Swift
env = Swift.Swift()
env.launch()

path = rtb.path_to_datafile('data')


g1 = rtb.Mesh(
    filename=str(path / 'gimbal-ring1.stl'),
    color=[34, 143, 201],
    scale=(1./3,) * 3
)
# g1.v = [0, 0, 0, 0.4, 0, 0]

g2 = rtb.Mesh(
    filename=str(path / 'gimbal-ring2.stl'),
    color=[31, 184, 72],
    scale=(1.1/3,) * 3

)
# g2.v = [0, 0, 0, 0.4, 0.0, 0]

g3 = rtb.Mesh(
    filename=str(path / 'gimbal-ring3.stl'),
    color=[240, 103, 103],
    scale=(1.1**2/3,) * 3
)
# g3.v = [0, 0, 0, 0.4, 0, 0]

plane = rtb.Mesh(
    filename=str(path / 'spitfire_assy-gear_up.stl'),
    scale=(1./(180*3),) * 3,
    color=[240, 103, 103]
)
print(path / 'spitfire_assy-gear_up.stl')
env.add(g1)
env.add(g2)
env.add(g3)
env.add(plane)

print('Supermarine Spitfire Mk VIII by Ed Morley @GRABCAD')
print('Gimbal models by Peter Corke using OpenSCAD')

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
        if which == 'X':
            return SO3.Rx(theta)
        elif which == 'Y':
            return SO3.Ry(theta)
        elif which == 'Z':
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
        return BASE * SE3.SO3(R)

    g3.base = convert(R1 * SO3.Ry(pi/2))
    g2.base = convert(R1 * R2 * SO3.Rz(pi/2))
    g1.base = convert(R1 * R2 * R3 * SO3.Rx(pi/2))
    plane.base = convert(R1 * R2 * R3 * SO3.Ry(pi/2) * SO3.Rz(pi/2))

# slider call backs, invoke the central handler
def set_one(x):
    update_gimbals(float(x), 1)

def set_two(x):
    update_gimbals(float(x), 2)

def set_three(x):
    update_gimbals(float(x), 3)


r_one = Swift.Slider(
    set_one,
    min=-180, max=180,
    step=1, value=0,
    desc='Outer gimbal', unit='&#176;')


r_two = Swift.Slider(
    set_two,
    min=-180, max=180,
    step=1, value=0,
    desc='Middle gimbal', unit='&#176;')


r_three = Swift.Slider(
    set_three,
    min=-180, max=180,
    step=1, value=0,
    desc='Inner gimbal', unit='&#176;')


# buttons to set a 3-angle sequence
ZYX_button = Swift.Button(
    lambda x: change_sequence('ZYX'),
    desc='ZYX (roll-pitch-yaw angles)'
)

XYZ_button = Swift.Button(
    lambda x: change_sequence('XYZ'),
    desc='XYZ (roll-pitch-yaw angles)'
)

ZYZ_button = Swift.Button(
    lambda x: change_sequence('ZYZ'),
    desc='ZYZ (Euler angles)'
)

button = Swift.Button(
    lambda x: set('ZYX'),
    desc='Set to Zero'
)

# button to reset joint angles
def reset(e):
    r_one.value = 0
    r_two.value = 0
    r_three.value = 0

zero_button = Swift.Button(
    reset,
    desc='Set to Zero'
)

def change_sequence(new):
    global sequence

    xyz = 'XYZ'

    # update the state of the ring_axis dropdowns
    ring1_axis.checked = xyz.find(new[0])
    ring2_axis.checked = xyz.find(new[1])
    ring3_axis.checked = xyz.find(new[2])

    sequence = new
    # print(sequence)

# handle radio button on angle slider
def angle(index, ring):
    global sequence

    # print('angle', index, ring)
    xyz = 'XYZ'
    s = list(sequence)
    s[ring] = xyz[int(index)]
    sequence = ''.join(s)

ring1_axis = Swift.Radio(
    lambda x: angle(x, 0),
    options=[
        'X',
        'Y',
        'Z'
    ],
    checked=2
)

ring2_axis = Swift.Radio(
    lambda x: angle(x, 1),
    options=[
        'X',
        'Y',
        'Z'
    ],
    checked=1
)

ring3_axis = Swift.Radio(
    lambda x: angle(x, 2),
    options=[
        'X',
        'Y',
        'Z'
    ],
    checked=0
)


def check_fn(indices):
    if indices[1]:
        print("YOU ARE WRONG")
    elif indices[0] and indices[2]:
        print("You are correct :)")
    else:
        print('Half marks')

check = Swift.Checkbox(
    check_fn,
    desc='Describe Jesse',
    options=[
        'Amazing',
        'Bad',
        'The Greatest'
    ],
    checked=[True, False, True]
)


def radio_fn(idx):
    if idx == 0:
        print("YOU ARE WRONG")
    else:
        print("You are correct :)")

radio = Swift.Radio(
    radio_fn,
    desc='Gimbal axis',
    options=[
        'X',
        'Y',
        'Z'
    ],
    checked=1
)

label = Swift.Label(
    desc='Triple angle'
)

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


# env.add(check)
# env.add(radio)

update_gimbals(0, 1)
update_gimbals(0, 2)
update_gimbals(0, 3)

while(True):
    # env.process_events()
    env.step(0)

# ring1_axis = Swift.Select(
#     lambda x: angle(x, 0),
#     desc='Outer gimbal axis',
#     options=[
#         'X',
#         'Y',
#         'Z'
#     ],
#     value=2
# )


# ring2_axis = Swift.Select(
#     lambda x: angle(x, 1),
#     desc='Middle gimbal axis',
#     options=[
#         'X',
#         'Y',
#         'Z'
#     ],
#     value=1
# )

# ring3_axis = Swift.Select(
#     lambda x: angle(x, 2),
#     desc='Inner gimbal axis',
#     options=[
#         'X',
#         'Y',
#         'Z'
#     ],
#     value=0
# )