# ropy
A robotics library for Python

**Used in**

J. Haviland and P. Corke, "Maximising  manipulability  during  resolved-rate  motion control," _arXiv preprint arXiv:2002.11901_, 2020.
[[arxiv](https://arxiv.org/abs/2002.11901)] [[Project Website](https://jhavl.github.io/mmc)]


## Installing

Requires Python ≥ 3.2

```shell script
git clone https://github.com/jhavl/ropy.git
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

# Calculate the Kinematic Jacobian (in the world frame) at joint angles q0
panda.J0
# or
panda.jacob0(q0)

# Calculate the forward kinematics of the robot at joint angles q0
panda.T
# or
panda.fkine(q0)

# Calculate the manipulability of the robot at joint angles q0
panda.m
# or
panda.manip(q0)

# Calculate the Kinematic Hessian (in the world frame) at joint angles q0
panda.H0
# or
panda.hessian0(q0)

# Print the Elementary Transform Sequence (ETS) of the robot
print(panda.ets)

```

### Manipulability Motion Control Example
This example implements Manipulability Motion Control from [this paper](https://arxiv.org/abs/2002.11901) within a position-based servoing scheme. We use the library [qpsolvers](https://pypi.org/project/qpsolvers/) to solve the optimisation function. However, you can use whichever solver you wish.

```python
import ropy as rp
import numpy as np
import qpsolvers as qp

# Initialise a Franka-Emika Panda Robot
panda = rp.Panda()

# The current joint angles of the Panda
# You need to obtain these from however you interfave with your robot
# eg. ROS messages, PyRep etc.
panda.q = np.array([0, -3, 0, -2.3, 0, 2, 0])

# The current pose of the robot
wTe = panda.T

# The desired pose of the robot
# = Current pose offset 20cm in the x-axis
wTep = np.copy(wTe)
wTep[0,0] += 0.2

# Gain term (lambda) for control minimisation
Y = 0.005

# Quadratic component of objective function
Q = Y * np.eye(7)

arrived = False
while not arrived:

    # The current joint angles of the Panda
    # You need to obtain these from however you interfave with your robot
    # eg. ROS messages, PyRep etc.
    panda.q = np.array([0, -3, 0, -2.3, 0, 2, 0])

    # The desired end-effecor spatial velocity
    v, arrived = rp.p_servo(wTe, wTep)

    # Form the equality constraints
    # The kinematic Jacobian in the end-effecor frame
    Aeq = panda.Je
    beq = v.reshape((6,))

    # Linear component of objective function: the manipulability Jacobian
    c = -panda.Jm.reshape((7,))

    # Solve for the joint velocities dq
    dq = qp.solve_qp(Q, c, None, None, Aeq, beq)

    # Send the joint velocities to the robot
    # eg. ROS messages, PyRep etc.
```

### Resolved-Rate Motion Control Example
This example implements resolved-rate motion control within a position-based servoing scheme

```python
import ropy as rp
import numpy as np

# Initialise a Franka-Emika Panda Robot
panda = rp.Panda()

# The current joint angles of the Panda
# You need to obtain these from however you interfave with your robot
# eg. ROS messages, PyRep etc.
panda.q = np.array([0, -3, 0, -2.3, 0, 2, 0])

# The current pose of the robot
wTe = panda.T

# The desired pose of the robot
# = Current pose offset 20cm in the x-axis
wTep = np.copy(wTe)
wTep[0,0] += 0.2

arrived = False
while not arrived:

    # The current joint angles of the Panda
    # You need to obtain these from however you interfave with your robot
    # eg. ROS messages, PyRep etc.
    panda.q = np.array([0, -3, 0, -2.3, 0, 2, 0])

    # The desired end-effecor spatial velocity
    v, arrived = rp.p_servo(wTe, wTep)
    
    # Solve for the joint velocities dq
    # Perfrom the pseudoinverse of the manipulator Jacobian in the end-effector frame
    dq = np.linalg.pinv(panda.Je) @ v

    # Send the joint velocities to the robot
    # eg. ROS messages, PyRep etc.
```
