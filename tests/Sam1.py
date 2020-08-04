"""
Manual Test
Example tests for serial link to show functionality

"""

# creates a simple three link robot
from time import sleep
from roboticstoolbox.robot.threelink import *
from roboticstoolbox.robot.uprighttl import *

def test_threelink_animate():
    # add a new joint configuration
    q1 = [0.1,0.1,0.1]
    q2 = [pi/2,pi/2,pi/2]

    # plot robot
    tl.plot(q1)

    for i in range(5):
        # repeat for visual confirmation
        tl.animate(q1,q2, frames=50, fps=25)
        sleep(1)
        tl.animate(q2,q1, frames=50, fps=25)

def test_upright_animate():
    # add a new joint configuration
    q1 = [0,0,-pi/3]
    q2 = [pi/4,pi/4,0]

    # plot robot
    arm.plot(q1)

    for i in range(5):
        # repeat for visual confirmation
        arm.animate(q1,q2, frames=50, fps=25)
        sleep(0.5)
        arm.animate(q2,q1, frames=50, fps=25)

if __name__ == "__main__":
    # run the animate robot test by default
    test_threelink_animate()