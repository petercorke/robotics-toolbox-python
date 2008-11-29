import robot.parsedemo as p;
import sys

print sys.modules

if __name__ == '__main__':

    s = '''
# First we will define some joint coordinates
    from robot.puma560 import *

# The path will move the robot from its zero angle pose to the upright (or 
# READY) pose.
#
# First create a time vector, completing the motion in 2 seconds with a
# sample interval of 56ms.
    t = arange(0, 2, 0.056) 	% generate a time vector
pause % hit any key to continue
#
# A polynomial trajectory between the 2 poses is computed using jtraj()
#
   (q, qd, qdd) = jtraj(qz, qr, t);
pause % hit any key to continue

#
# For this particular trajectory most of the motion is done by joints 2 and 3,
# and this can be conveniently plotted using standard MATLAB operations
    subplot(2,1,1);
    plot(t, q[:,1]);
    title('Theta');
    xlabel('Time (s)');
    ylabel('Joint 2 (rad)');
    subplot(2,1,2);
    plot(t, q[:,2]);
    xlabel('Time (s)');
    ylabel('Joint 3 (rad)');
    show();

#
# We can also look at the velocity and acceleration profiles.  We could 
# differentiate the angle trajectory using diff(), but more accurate results 
# can be obtained by requesting that jtraj() return angular velocity and 
# acceleration as follows
    (q, qd, qdd) = jtraj(qz, qr, t);
#
#  which can then be plotted as before

    subplot(2,1,1);
    plot(t, qd[:,1]);
    title('Velocity');
    xlabel('Time (s)');
    ylabel('Joint 2 vel (rad/s)');
    subplot(2,1,2);
    plot(t, qd[:,2]);
    xlabel('Time (s)');
    ylabel('Joint 3 vel (rad/s)');

# and the joint acceleration profiles
    subplot(2,1,1);
    plot(t, qdd[:,2]);
    title('Acceleration');
    xlabel('Time (s)');
    ylabel('Joint 2 accel (rad/s2)');
    subplot(2,1,2);
    plot(t, qdd[:,3]);
    xlabel('Time (s)');
    ylabel('Joint 3 accel (rad/s2)');

'''

    p.parsedemo(s);
