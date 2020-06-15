"""
Defines the object 'robot' in the current workspace

Also define the vector q = [0 0 0] which corresponds to the zero joint
angle configuration.

@author: Luis Fernando Lara Tobar and Peter Corke

Edited June 2020 by Samuel Drew
"""

from roboticstoolbox.robot.serial_link import *

L = []

L.append(Link('a', 1, 'type', 'revolute'))
L.append(Link('a', 1, 'type', 'revolute'))
L.append(Link('a', 1, 'type', 'revolute'))


qz = [0,0,0]

tl = SerialLink(L, name='Simple three link')

