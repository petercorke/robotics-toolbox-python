#!/usr/bin/env python3

import roboticstoolbox as rtb

# create a Puma560 model
puma = rtb.models.Puma560()
print(puma)

# do forward kinematics
T = puma.fkine( puma.qr )
print(T)

# we should be able to execute a line like the following
#   - right now the browser window and axes appear, but no robot
puma.plot( puma.qr )
