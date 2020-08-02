# #!/usr/bin/env python

# import numpy as np
# from ropy.robot.Revolute import Revolute
# from ropy.robot.SerialLink import SerialLink
# from ropy.tools.transform import transl, xyzrpy_to_trans

# class Sawyer(SerialLink):
#     """
#     A class representing the Rethink Robotics Sawyer robot arm.
    
#     Attributes:
#     --------
#         name : string
#             Name of the robot
#         manufacturer : string
#             Manufacturer of the robot
#         links : List[n]
#             Series of links which define the robot
#         base : float np.ndarray(4,4)
#             Locaation of the base
#         tool : float np.ndarray(4,4)
#             Location of the tool
#         mdh : int
#             1: Pnada is modified D&H
#         n : int
#             Number of joints in the robot

#     Examples
#     --------
#     >>> sawyer = Sawyer()

#     See Also
#     --------
#     ropy.robot.SerialLink : A superclass for arm type robots
#     """

#     def __init__(self):

#         deg = np.pi/180
#         mm = 1e-3
        
#         d1 = (81)*mm

#         lim1 = 30 * deg
        
#         L1 = Revolute(a = 317*mm,    d =  d1, alpha = -np.pi/2, qlim = np.array([-lim1, lim1]))
#         L2 = Revolute(a = 192.5*mm,  d = 0.0, alpha = -np.pi/2, qlim = np.array([-lim1, lim1]))
#         L3 = Revolute(a = 400*mm,    d = 0.0, alpha = -np.pi/2, qlim = np.array([-lim1, lim1]))
#         L4 = Revolute(a = 168.5*mm,  d = 0.0, alpha = -np.pi/2, qlim = np.array([-lim1, lim1]))
#         L5 = Revolute(a = 400*mm,    d = 0.0, alpha = -np.pi/2, qlim = np.array([-lim1, lim1]))
#         L6 = Revolute(a = 136.3*mm,  d = 0.0, alpha = -np.pi/2, qlim = np.array([-lim1, lim1]))
#         L7 = Revolute(a = 133.75*mm, d = 0.0, alpha =        0, qlim = np.array([-lim1, lim1]))

#         L = [L1, L2, L3, L4, L5, L6, L7]

#         super(Sawyer, self).__init__(L, name = 'Sawyer', manufacturer = 'Rethink Robotics', tool = xyzrpy_to_trans(0, 0, 0, 0, 0, 0))

#         self.qz = np.array([0, 0, 0, 0, 0, 0, 0])
#         self.qr = np.array([0, -90, -90, 90, 0, -90, 90]) * deg

