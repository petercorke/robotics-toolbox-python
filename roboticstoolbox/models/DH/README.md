# Denavit-Hartenberg models

## Shipped robot models

| Model name | Description |
| ---        | ---         |
| Ball       | an n-link robot that folds into a ball shape |
| Cobra600   | 4-axis Adept (now OMRON) SCARA robot |
| IRB140  | 6-axis ABB robot |
| KR5  | 6-axis Kuka robot |
| Panda | 7-axis Franka-Emika robot |
| Puma | 6-axis Unimation robot (with dynamics data) |
| Stanford | 6-axis Stanford resarch robot, 1 prismatic joint (with dynamics data) |
| Threelink | ?? |

## Creating a new robot model using Denavit-Hartenberg parameters

To begin, you must know:

1. The DH parameters for your robot.
2. Whether the model uses standard or modified Denavit-Hartenberg parameters, DH or MDH respectively.
3. The joint structure, does it have revolute, prismatic joints or both.


### Write the definition

Create a file called `MYROBOT.py` where `MYROBOT` is a descriptive name of your
robot that is a valid filename and Python class name.

Cut and paste the code blocks below into your empty file, modifying as you go.  We start with the definitions:


```python
from math import pi
import numpy as np
from roboticstoolbox import DHRobot, RevoluteDH, PrismaticDH, RevoluteMDH, PrismaticMDH
```

The last line is important, it defines all the classes you could possibly
need.  You won't use all of them, and to be tidy you could delete those you don't use.  This is their purpose:

* `RevoluteDH` for a revolute joint using standard DH parameters
* `PrismaticDH` for a prismatic joint using standard DH parameters
* `RevoluteMDH` for a revolute joint using modified DH parameters
* `PrismaticMDH` for a prismatic joint using modified DH parameters

Next, add as much description as you can, check out the other model files in this
folder for inspiration.

```python
class MYROBOT(DHRobot):
    """
    Create model of MYROBOT manipulator

       .
       .
       .

    :notes:
       .
       .
       .

    :references:
       .
       .
       .     

    """
```

Always a good idea to have notes about units used, DH convention, and
to include any references to the source of your model.  

Now for the main event:.  Inside the `__init__` method, define a set of link variables by creating instances of the appropriate link class.

```python
def __init__(self):

        deg = pi/180

        L0 = RevoluteDH(
            d=0,          # link length (Dennavit-Hartenberg notation)
            a=0,          # link offset (Dennavit-Hartenberg notation)
            alpha=pi/2,   # link twist (Dennavit-Hartenberg notation)
            I=[0, 0.35, 0, 0, 0, 0],  # inertia tensor of link with respect to
                                      # center of mass I = [L_xx, L_yy, L_zz,
                                      # L_xy, L_yz, L_xz]
            r=[0, 0, 0],  # distance of ith origin to center of mass [x,y,z]
                          # in link reference frame
            m=0,          # mass of link
            Jm=200e-6,    # actuator inertia
            G=-62.6111,   # gear ratio
            B=1.48e-3,    # actuator viscous friction coefficient (measured
                          # at the motor)
            Tc=[0.395, -0.435],  # actuator Coulomb friction coefficient for
                                 # direction [-,+] (measured at the motor)
            qlim=[-160*deg, 160*deg])    # minimum and maximum joint angle

        L1 = RevoluteDH(
            d=0, a=0.4318, alpha=0,
            qlim=[-45*deg, 225*deg])

            .
            .
            .   

```

Provide as many parameters as you can.  The definition of `L0` above includes
kinematic and dynamic parameters, whereas `L1` has only kinematic parameters.
The minimum requirement is for the kinematic parameters, and you don't even need
to know the joint limits, they are only required by a small number of Toolbox
functions.

For a robot with N joints you must define N joint instances.

Next we call the superclass constructor to do the heavy lifting.

```python
        super().__init__(
            [L0, L1, L2, L3, L4, L5],
            name="MYROBOT",
            manufacturer="COMPANY THAT BUILDS MYROBOTs")
```

We pass in an ordered list of all the joint objects we created earlier, and add 
some metadata.  The name gets used in plots.

We might like to define some useful joint configurations, maybe the home position, or where it goes to pick up a widget.  You can 
define as many of these as you like, but the pattern looks like this:

```python
        # zero angles, L shaped pose
        self._MYCONFIG = np.array([1, 2, 3, 4, 5, 6])  # create instance attribute

    @property
    def MYCONFIG(self):
        return self._MYCONFIG

```
where `MYCONFIG` is the name of this particular configuration. Define an instance attribute holding the joint configuration, it must be a
NumPy array with N elements.  Then define a property that will return that attribute.

Many of the
Toolbox models have a configuration called `qz` which is the set of zero
joint angles.

Finally, we can make the robot definition an executable script.  We do that by adding a hashbang line at the top of the file

```python
#!/usr/bin/env python
```

and a main code block at the bottom

```python
if __name__ == '__main__':

    robot = MYROBOT()
    print(robot)
```

so if you run it from your shell

```
% ./Puma560.py 
┏━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━┳━━━━━━━┓
┃   θⱼ    ┃   dⱼ    ┃   aⱼ   ┃  ⍺ⱼ   ┃
┣━━━━━━━━━╋━━━━━━━━━╋━━━━━━━━╋━━━━━━━┫
┃q0 + 0.0 ┃   0.672 ┃      0 ┃  90.0 ┃
┃q1 + 0.0 ┃       0 ┃ 0.4318 ┃   0.0 ┃
┃q2 + 0.0 ┃ 0.15005 ┃ 0.0203 ┃ -90.0 ┃
┃q3 + 0.0 ┃  0.4318 ┃      0 ┃  90.0 ┃
┃q4 + 0.0 ┃       0 ┃      0 ┃ -90.0 ┃
┃q5 + 0.0 ┃       0 ┃      0 ┃   0.0 ┃
┗━━━━━━━━━┻━━━━━━━━━┻━━━━━━━━┻━━━━━━━┛
```

we see the table of Denavit-Hartenberg parameters.

### Adding your model to the Toolbox

Edit the file `__init__.py` in this folder.  Add a line like:

```python
from roboticstoolbox.models.DH.Stanford import MYROBOT
```

and then add `'MYROBOT'` to the list that defines `__all__`.

### Testing

If you made your definition an executable Python script as above, then run it
and check that the parameters are what you expect them to be.

If you defined some configurations you can also test that they are correct.

```python
    myrobot = MYROBOT()  # instantiate an instance of your model
    print(myrobot)       # display its kinematic parameters
    print(myrobot.MYCONFIG)    # check that the joint configuration works
```

### Next steps

You can now use the power of the Toolbox to compute forward and inverse
kinematics, display a graphical model, and interactively teach the robot.
If you defined dynamic parameters then you can compute forward and inverse
rigid-body dyamics and simulate the response of the robot to applied torques.

Good luck and enjoy!

### Contribute your model

If you think your model might be interesting to others consider submitting a pull request.
