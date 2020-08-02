# #!/usr/bin/env python

# import numpy as np
# from ropy.robot.Revolute import Revolute
# from ropy.robot.SerialLink import SerialLink

# class Mico:

#     def __init__(self):

#         deg = np.pi/180

#         # robot length values (metres)  page 4
#         self.D1 = 0.2755
#         self.D2 = 0.2900
#         self.D3 = 0.1233
#         self.D4 = 0.0741
#         self.D5 = 0.0741
#         self.D6 = 0.1600
#         self.e2 = 0.0070
    
#         # alternate parameters
#         self.aa = 30 * deg
#         self.ca = np.cos(self.aa)
#         self.sa = np.sin(self.aa)
#         self.c2a = np.cos(2*self.aa)
#         self.s2a = np.sin(2*self.aa)
#         self.d4b = self.D3 + self.sa/self.s2a*self.D4
#         self.d5b = self.sa/self.s2a*self.D4 + self.sa/self.s2a*self.D5
#         self.d6b = self.sa/self.s2a*self.D5 + self.D6

#         L = [  Revolute(alpha = np.pi/2,     a = 0,       d =  self.D1,   flip = True),
#                Revolute(alpha = np.pi,       a = self.D2, d =  0,         offset = -np.pi/2),
#                Revolute(alpha = np.pi/2,     a = 0,       d = -self.e2,   offset = np.pi/2),
#                Revolute(alpha = 2*self.aa,   a = 0,       d = -self.d4b),
#                Revolute(alpha = 2*self.aa,   a = 0,       d = -self.d5b,  offset = -np.pi),
#                Revolute(alpha = np.pi,       a = 0,       d = -self.d6b,  offset = np.pi/2)  ]

#         self.r = SerialLink( L, name = 'Mico', manufacturer = 'Kinova')
