"""
Manual Test
Example tests for serial link to show functionality

"""

# creates a simple three link robot
from time import sleep
from roboticstoolbox.robot.threelink import *

def test_robot_animate():
    # add a new joint configuration
    q2 = [pi/2,pi/2,pi/2]

    # plot robot
    tl.plot(qz)

    for i in range(3):
        # repeat for visual confirmation
        tl.animate(qz,q2, frames=50, fps=25)
        sleep(1)
        tl.animate(q2,qz, frames=50, fps=25)

if __name__ == "__main__":
    # run the animate robot test by default
    test_robot_animate()