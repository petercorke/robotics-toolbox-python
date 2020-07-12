#!/usr/bin/env python3

import roboticstoolbox as rtb

# create a Puma560 model
puma = rtb.models.Puma560()
print(puma)

# do forward kinematics
T = puma.fkine( puma.config('qr') )
print(T)