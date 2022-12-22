import numpy as np
import roboticstoolbox as rtb
import spatialmath as sm


r = rtb.models.DH.Panda()

print(isinstance(r, rtb.DHRobot))
