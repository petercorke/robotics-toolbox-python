#!/usr/bin/env python3 -i

# a simple Robotics Toolbox "shell", runs Python3 and loads in NumPy, RTB, SMTB
# 
# Run it from the shell
#  % rtb.py
#
# or setup an alias
#
#  alias rtb=PATH/rtb.py   # sh/bash
#  alias rtb PATH/rtb.py   # csh/tcsh
#
# % rtb

# import stuff
from math import pi              # lgtm [py/unused-import]
import numpy as np
import matplotlib as plt         # lgtm [py/unused-import]
import roboticstoolbox as rtb    # lgtm [py/unused-import]
from spatialmath import *        # lgtm [py/polluting-import]
from spatialmath.base import *   # lgtm [py/polluting-import]

# setup defaults
np.set_printoptions(linewidth=120, formatter={'float': lambda x: f"{x:8.4g}" if abs(x) > 1e-10 else f"{0:8.4g}"})
SE3._ansimatrix = True

# print the banner
# https://patorjk.com/software/taag/#p=display&f=Cybermedium&t=Robotics%20Toolbox%0A
print(r"""____ ____ ___  ____ ___ _ ____ ____    ___ ____ ____ _    ___  ____ _  _
|__/ |  | |__] |  |  |  | |    [__      |  |  | |  | |    |__] |  |  \/
|  \ |__| |__] |__|  |  | |___ ___]     |  |__| |__| |___ |__] |__| _/\_

for Python

""")
