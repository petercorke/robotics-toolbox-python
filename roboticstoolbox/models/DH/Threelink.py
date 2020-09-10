"""
@author: Luis Fernando Lara Tobar
@author: Peter Corke
@author: Samuel Drew
"""

from roboticstoolbox import DHRobot, RevoluteDH
import numpy as np


class Threelink(DHRobot):
    """
    Defines the object 'tl' in the current workspace

    Also define the vector qz = [0 0 0] which corresponds to the zero joint
    angle configuration.
    """

    def __init__(self):

        L = [RevoluteDH(a=1, jointtype='R'),
             RevoluteDH(a=1, jointtype='R'),
             RevoluteDH(a=1, jointtype='R')]

        self._qz = [np.pi/4, 0.1, 0.1]

        super(Threelink, self).__init__(L, name='Simple three link')

    @property
    def qz(self):
        return self._qz
