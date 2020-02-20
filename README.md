# ropy
A robotics library for Python

## Installing

Requires Python ≥ 3.2

```shell script
git clone git@github.com:jhavl/ropy.git
cd ropy
pip3 install -e .
```

## Usage

### Arm-Type Robots

```python
import ropy as rp
import numpy as np

# Initialise a Franka-Emika Panda robot
panda = rp.Panda()

# Set the joint angles of the robot
q0 = np.array([0,0,-1.2,0,0,-2,0])
panda.q = q0

# Calculate the Kinematic Jacobian (in the world frame) at q0
panda.J0
# or
panda.jacob0(q0)
```
