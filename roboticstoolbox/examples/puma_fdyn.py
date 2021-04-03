import roboticstoolbox as rtb

# load a model with inertial parameters
p560 = rtb.models.DH.Puma560()

# remove Coulomb friction
p560 = p560.nofriction()

# print the kinematic & dynamic parameters
p560.printdyn()

# simulate motion over 5s with zero torque input
d = p560.fdyn(5, p560.qr, dt=0.05)

# show the joint angle trajectory
rtb.tools.trajectory.qplot(d.q)

# animate it
p560.plot(d.q)  # movie='falling_puma.gif')
