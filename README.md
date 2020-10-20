![stability-wip](https://img.shields.io/badge/stability-work_in_progress-lightgrey.svg) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/petercorke/robotics-toolbox-python/master?filepath=notebooks)
[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)
<!---
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/roboticstoolbox-python.svg)
--->
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/petercorke/robotics-toolbox-python.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/petercorke/robotics-toolbox-python/context:python)
[![Build Status](https://github.com/petercorke/robotics-toolbox-python/workflows/build/badge.svg?branch=master)](https://github.com/petercorke/robotics-toolbox-python/actions?query=workflow%3Abuild)
[![Coverage](https://codecov.io/gh/petercorke/robotics-toolbox-python/branch/master/graph/badge.svg)](https://codecov.io/gh/petercorke/robotics-toolbox-python)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


# Robotics Toolbox for Python

![RTB-P logo](https://github.com/petercorke/robotics-toolbox-python/raw/master/docs/figs/RobToolBox_RoundLogoB.png)

This is a Python implementation of the [Robotics Toolbox for MATLAB<sup>&reg;</sup>](https://github.com/petercorke/robotics-toolbox-matlab).

* GitHub repository [https://github.com/petercorke/robotics-toolbox-python](https://github.com/petercorke/robotics-toolbox-python)
* Examples and details: [https://github.com/petercorke/robotics-toolbox-python/wiki](https://github.com/petercorke/robotics-toolbox-python/wiki)    
* Documentation [https://petercorke.github.io/robotics-toolbox-python](https://petercorke.github.io/robotics-toolbox-python)



## Synopsis

This toolbox brings robotics specific functionality to Python, and leverages the advantages of Python such as portability, ubiquity and support, and the capability of the Python ecosystem for linear algebra (numpy, scipy),  graphics (matplotlib, three.js), interactive development (jupyter, jupyterlab), documentation (sphinx).

The Toolbox provides tools for representing the kinematics and dynamics of serial-link manipulators  - you can create your own in Denavit-Hartenberg form, import a URDF file, or use supplied models for well known robots from Franka-Emika, Kinova, Universal Robotics, Rethink as well as classical robots such as the Puma 560 and the Stanford arm.

The toolbox also supports mobile robots with functions for robot motion models (unicycle, bicycle), path planning algorithms (bug, distance transform, D*, PRM), kinodynamic planning (lattice, RRT), localization (EKF, particle filter), map building (EKF) and simultaneous localization and mapping (EKF), and a Simulink model a of non-holonomic vehicle.  The Toolbox also including a detailed Simulink model for a quadrotor flying robot.

Advantages of the Toolbox are that:

  * the code is mature and provides a point of comparison for other implementations of the same algorithms;
  * the routines are generally written in a straightforward manner which allows for easy understanding, perhaps at the expense of computational efficiency. If you feel strongly about computational efficiency then you can always rewrite the function to be more efficient, compile the M-file using the MATLAB compiler, or create a MEX version;
  * since source code is available there is a benefit for understanding and teaching.
  
The MATLAB version of this Toolbox dates back to 1993.

## Code Example

```python
>>> import roboticstoolbox as rtb
>>> p560 = rtb.models.DH.Puma560()
>>> print(p560)

┏━━━┳━━━━━━━━━┳━━━━━━━━┳━━━━━━┓
┃θⱼ ┃   dⱼ    ┃   aⱼ   ┃  ⍺ⱼ  ┃
┣━━━╋━━━━━━━━━╋━━━━━━━━╋━━━━━━┫
┃q1 ┃   0.672 ┃      0 ┃ None ┃
┃q2 ┃       0 ┃ 0.4318 ┃ None ┃
┃q3 ┃ 0.15005 ┃ 0.0203 ┃ None ┃
┃q4 ┃  0.4318 ┃      0 ┃ None ┃
┃q5 ┃       0 ┃      0 ┃ None ┃
┃q6 ┃       0 ┃      0 ┃ None ┃
┗━━━┻━━━━━━━━━┻━━━━━━━━┻━━━━━━┛

┌───┬────────────────────────────┐
│qz │ 0°, 0°, 0°, 0°, 0°, 0°     │
│qr │ 0°, 90°, -90°, 0°, 0°, 0°  │
│qs │ 0°, 0°, -90°, 0°, 0°, 0°   │
│qn │ 0°, 45°, 180°, 0°, 45°, 0° │
└───┴────────────────────────────┘

 
>>> p560.fkine([0, 0, 0, 0, 0, 0])  # forward kinematics
   1           0           0           0.4521       
   0           1           0          -0.15005      
   0           0           1           0.4318       
   0           0           0           1            
```

We can animate a path

```python
qt = rtb.tools.trajectory.jtraj(p560.qz, p560.qr, 50)
p560.plot(qt.q)
```

![Puma robot animation](https://github.com/petercorke/robotics-toolbox-python/raw/master/docs/figs/puma_sitting.gif)

which uses the default matplotlib backend.  We can instantiate our robot inside
the 3d simulation environment

```python
env = rtb.backend.Sim()
env.launch()
env.add(p560)
```

```python
# inv kienmatis example here
# jacobian
```

# Getting going

## Installing

### From GitHub

Requires Python ≥ 3.5.

```shell script
git clone https://github.com/petercorke/robotics-toolbox-python.git
cd robotics-toolbox-python
pip3 install -e .
```

### Using pip

Install a snapshot from PyPI

```shell script
pip3 install roboticstoolbox-python
```

## Run some examples

The [`notebooks`](https://github.com/petercorke/robotics-toolbox-python/tree/master/notebooks) folder contains some tutorial Jupyter notebooks which you can browse on GitHub.

Or you can run them, and experiment with them, at [mybinder.org](https://mybinder.org/v2/gh/petercorke/robotics-toolbox-python/master?filepath=notebooks).
