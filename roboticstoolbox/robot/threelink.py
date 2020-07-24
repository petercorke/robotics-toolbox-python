"""
Defines the object 'tl' in the current workspace

Also define the vector qz = [0 0 0] which corresponds to the zero joint
angle configuration.

@author: Luis Fernando Lara Tobar and Peter Corke

Edited June 2020 by Samuel Drew
"""

from roboticstoolbox.robot.serial_link import *

L = []

L.append(Link(a=1, jointtype='R'))
L.append(Link(a=1, jointtype='R'))
L.append(Link(a=1, jointtype='R'))


qz = [0.1,0.1,0.1]

tl = SerialLink(L, name='Simple three link')

