import numpy as np
import roboticstoolbox as rtb
from spatialmath import *
from math import pi

p560 = rtb.models.DH.Puma560()
print(p560)

T = p560.fkine(p560.qz)
print(T)

T = p560.fkine_all(p560.qz, old=False)
print(T)
