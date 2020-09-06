# #!/usr/bin/env python

# import numpy as np
# from ropy.robot.Revolute import Revolute
# from ropy.robot.SerialLink import SerialLink
# from ropy.tools.transform import transl, xyzrpy_to_trans

# class LBR7(SerialLink):
#     """
#     A class representing the Kuka LBR iiwa 7 robot arm.
    
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
#     >>> kuka = Iiwa7()

#     See Also
#     --------
#     ropy.robot.SerialLink : A superclass for arm type robots
#     """

#     def __init__(self):

#         deg = np.pi/180
#         mm = 1e-3
        
#         d3 = (400)*mm
#         d5 = (400)*mm

#         lim1 = 170 * deg
#         lim2 = 175 * deg
        
#         L1 = Revolute(a = 0.0, d = 0.0, alpha =      0.0, qlim = np.array([-lim1, lim1]), mdh = 1)
#         L2 = Revolute(a = 0.0, d = 0.0, alpha =  np.pi/2, qlim = np.array([-lim1, lim1]), mdh = 1)
#         L3 = Revolute(a = 0.0, d =  d3, alpha = -np.pi/2, qlim = np.array([-lim1, lim1]), mdh = 1)
#         L4 = Revolute(a = 0.0, d = 0.0, alpha = -np.pi/2, qlim = np.array([-lim1, lim1]), mdh = 1)
#         L5 = Revolute(a = 0.0, d =  d5, alpha =  np.pi/2, qlim = np.array([-lim1, lim1]), mdh = 1)
#         L6 = Revolute(a = 0.0, d = 0.0, alpha =  np.pi/2, qlim = np.array([-lim1, lim1]), mdh = 1)
#         L7 = Revolute(a = 0.0, d = 0.0, alpha = -np.pi/2, qlim = np.array([-lim2, lim2]), mdh = 1)

#         L = [L1, L2, L3, L4, L5, L6, L7]

#         super(LBR7, self).__init__(L, name = 'LBR-7', manufacturer = 'Kuka', tool = xyzrpy_to_trans(0, 0, 0, 0, 0, 0))

#         self.qz = np.array([0, 0, 0, 0, 0, 0, 0])
#         self.qr = np.array([0, -90, -90, 90, 0, -90, 90]) * deg

