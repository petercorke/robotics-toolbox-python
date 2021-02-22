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

# Launch the simulator Swift
env = rtb.backends.Swift()
env.launch()

path = pathlib.Path(path) / 'roboticstoolbox' / 'data'


g1 = rtb.Mesh(
    filename=str(path / 'gimbal-ring1.stl'),
    color=[34, 143, 201],
    scale=(1./3,) * 3
)
# g1.v = [0, 0, 0, 0.4, 0, 0]

g2 = rtb.Mesh(
    filename=str(path / 'gimbal-ring2.stl'),
    color=[31, 184, 72],
    scale=(1./3,) * 3
)
# g2.v = [0, 0, 0, 0.4, 0.0, 0]

g3 = rtb.Mesh(
    filename=str(path / 'gimbal-ring3.stl'),
    color=[240, 103, 103],
    scale=(1./3,) * 3
)
# g3.v = [0, 0, 0, 0.4, 0, 0]

plane = rtb.Mesh(
    filename=str(path / 'spitfire_assy-gear_up.stl'),
    scale=(1./(180*3),) * 3,
    color=[240, 103, 103]
)

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
sequence = "zyx"


def update_gimbals(theta, ring):
    global R1, R2, R3

    # update the relevant transform, depending on which ring's slider moved

    def Rxyz(theta, which):
        theta = np.radians(theta)
        if which == 'x':
            return SO3.Rx(theta)
        elif which == 'y':
            return SO3.Ry(theta)
        elif which == 'z':
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


r_one = rtb.backends.Slider(
    set_one,
    min=-180, max=180,
    step=1, value=0,
    desc='Ring One', unit='&#176;')


r_two = rtb.backends.Slider(
    set_two,
    min=-180, max=180,
    step=1, value=0,
    desc='Ring Two', unit='&#176;')


r_three = rtb.backends.Slider(
    set_three,
    min=-180, max=180,
    step=1, value=0,
    desc='Ring Three', unit='&#176;')


def reset(e):
    r_one.value = 0
    r_two.value = 0
    r_three.value = 0

button = rtb.backends.Button(
    reset,
    desc='Set to Zero'
)


def angle(index):
    print('Selection Box Index selected: ' + str(index))

rot_seq = rtb.backends.Select(
    angle,
    desc='Rotation Sequence',
    options=[
        'zyx',
        'zyz',
        'I dont know'
    ],
    value=0
)


def check_fn(indices):
    if indices[1]:
        print("YOU ARE WRONG")
    elif indices[0] and indices[2]:
        print("You are correct :)")
    else:
        print('Half marks')

check = rtb.backends.Checkbox(
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

radio = rtb.backends.Radio(
    radio_fn,
    desc='Jesse is:',
    options=[
        'Bad',
        'Good',
    ],
    checked=1
)

label = rtb.backends.Label(
    desc='Teach Panel'
)

env.add(label)
env.add(r_one)
env.add(r_two)
env.add(r_three)
env.add(button)
env.add(rot_seq)
env.add(check)
env.add(radio)

update_gimbals(0, 1)
update_gimbals(0, 2)
update_gimbals(0, 3)

while(True):
    # env.process_events()
    env.step(0)

