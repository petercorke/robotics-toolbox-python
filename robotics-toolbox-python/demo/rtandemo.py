
#**************************animation********************************************
figure(2)
echo on
clf
#
# The trajectory demonstration has shown how a joint coordinate trajectory
# may be generated
    t = [0:.056:2]'; 	% generate a time vector
    q = jtraj(qz, qr, t); % generate joint coordinate trajectory
#
# the overloaded function plot() animates a stick figure robot moving 
# along a trajectory.

    plot(p560, q);
# The drawn line segments do not necessarily correspond to robot links, but 
# join the origins of sequential link coordinate frames.
#
# A small right-angle coordinate frame is drawn on the end of the robot to show
# the wrist orientation.
#
# A shadow appears on the ground which helps to give some better idea of the
# 3D object.

pause % any key to continue
#
# We can also place additional robots into a figure.
#
# Let's make a clone of the Puma robot, but change its name and base location

    p560_2 = p560;
    p560_2.name = 'another Puma';
    p560_2.base = transl(-0.5, 0.5, 0);
    hold on
    plot(p560_2, q);
pause % any key to continue

# We can also have multiple views of the same robot
    clf
    plot(p560, qr);
    figure
    plot(p560, qr);
    view(40,50)
    plot(p560, q)
pause % any key to continue
#
# Sometimes it's useful to be able to manually drive the robot around to
# get an understanding of how it works.

    drivebot(p560)
#
# use the sliders to control the robot (in fact both views).  Hit the red quit
# button when you are done.
echo off
