# """
# Defines the object 'arm' in the current workspace

# Also define the vector qz = [pi/4,0,-pi/3]
# """

# from roboticstoolbox.robot.serial_link import *


# class Uprighttl(SerialLink):

#     def __init__(self):
#         L = [Link(a=0.1, alpha=pi / 2, jointtype='R'),
#              Link(a=1.0, jointtype='R'),
#              Link(a=1.0, jointtype='R'),
#              Link(a=0.5, jointtype='R')]

#         self._qz = [0, 0, 0, 0]

#         super(Uprighttl, self).__init__(L, name='Upright Arm')

#     @property
#     def qz(self):
#         return self._qz
