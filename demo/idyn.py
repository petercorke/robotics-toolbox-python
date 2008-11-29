import robot.parsedemo as p;
import sys

print sys.modules

if __name__ == '__main__':

    s = '''
# First we will define a robot
    from robot.puma560 import *

# Inverse dynamics computes the joint torques required to achieve the specified
# state of joint position, velocity and acceleration.  
# The recursive Newton-Euler formulation is an efficient matrix oriented
# algorithm for computing the inverse dynamics, and is implemented in the 
# function rne().
#
# Inverse dynamics requires inertial and mass parameters of each link, as well
# as the kinematic parameters.  This is achieved by augmenting the kinematic 
# description matrix with additional columns for the inertial and mass 
# parameters for each link.
#
# For example, for a Puma 560 in the zero angle pose, with all joint velocities
# of 5rad/s and accelerations of 1rad/s/s, the joint torques required are
#
    tau = rne(p560, qz, 5*ones((1,6)), ones((1,6)))
pause % any key to continue

# As with other functions the inverse dynamics can be computed for each point 
# on a trajectory.  Create a joint coordinate trajectory and compute velocity 
# and acceleration as well
    t = arange(0, 2, 0.056); 		% create time vector
    (q,qd,qdd) = jtraj(qz, qr, t); % compute joint coordinate trajectory
    tau = rne(p560, q, qd, qdd); % compute inverse dynamics
#
#  Now the joint torques can be plotted as a function of time
    plot(t, tau[:,0:2]);
    xlabel('Time (s)');
    ylabel('Joint torque (Nm)');
    show()

#
# Much of the torque on joints 2 and 3 of a Puma 560 (mounted conventionally) is
# due to gravity.  That component can be computed using gravload()
    clf();
    taug = gravload(p560, q);
    plot(t, taug[:,0:2]);
    xlabel('Time (s)');
    ylabel('Gravity torque (Nm)');
    show()

# Now lets plot that as a fraction of the total torque required over the 
# trajectory
    clf();
    subplot(2,1,1);
    plot(t, hstack((tau[:,1], taug[:,1])));
    xlabel('Time (s)');
    ylabel('Torque on joint 2 (Nm)');
    subplot(2,1,2);
    plot(t, hstack((tau[:,2], taug[:,2])));
    xlabel('Time (s)');
    ylabel('Torque on joint 3 (Nm)');
pause % any key to continue
#
# The inertia seen by the waist (joint 1) motor changes markedly with robot 
# configuration.  The function inertia() computes the manipulator inertia matrix
# for any given configuration.
#
#  Let's compute the variation in joint 1 inertia, that is M(1,1), as the 
# manipulator moves along the trajectory (this may take a few seconds)
    M = inertia(p560, q);
    M11 = array([m[0,0] for m in M]);
    clf();
    plot(t, M11);
    xlabel('Time (s)');
    ylabel('Inertia on joint 1 (kgms2)')
# Clearly the inertia seen by joint 1 varies considerably over this path.
# This is one of many challenges to control design in robotics, achieving 
# stability and high-performance in the face of plant variation.  In fact 
# for this example the inertia varies by a factor of
    max(M11)/min(M11)
pause
'''

    p.parsedemo(s);
