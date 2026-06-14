data = """
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   PARAMETERS Returns a data structure containing the parameters of the
%   UNIVERSAL ROBOTS UR10.
%
%   Author: David J. Roldán. Universidad Miguel Hernandez de Elche. 
%   email: xxxx@umh.es date:   08/12/2017
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Copyright (C) 2012, by Arturo Gil Aparicio
%
% This file is part of ARTE (A Robotics Toolbox for Education).
% 
% ARTE is free software: you can redistribute it and/or modify
% it under the terms of the GNU Lesser General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
% 
% ARTE is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU Lesser General Public License for more details.
% 
% You should have received a copy of the GNU Leser General Public License
% along with ARTE.  If not, see <http://www.gnu.org/licenses/>.
function robot = parameters()

robot.name= 'UR10';
robot.DH.theta= '[q(1) q(2)+pi/2 q(3) q(4)-pi/2 q(5) q(6)]';
robot.DH.d='[0.128 0.176 -0.128 0.116 0.116 0.092]';
robot.DH.a='[0 0.612 0.572 0 0 0]';
robot.DH.alpha= '[pi/2 0 0 -pi/2 pi/2 0]';
robot.J=[];

% robot.inversekinematic_fn = 'inverse_kinematics_ur10(robot, T, q)';
robot.inversekinematic_fn = 'inverse_kinematics_jacobian_transpose(robot, T, q)';
robot.directkinematic_fn = 'directkinematic(robot, q)';


%number of degrees of freedom
robot.DOF = 6;

%rotational: 0, translational: 1
robot.kind=['R' 'R' 'R' 'R' 'R' 'R'];

%minimum and maximum rotation angle in rad
robot.maxangle =[-pi pi; %Axis 1, minimum, maximum
                deg2rad(-100) deg2rad(100); %Axis 2, minimum, maximum
                deg2rad(-220) deg2rad(60); %Axis 3
                deg2rad(-200) deg2rad(200); %Axis 4: Unlimited (400ï¿½ default)
                deg2rad(-120) deg2rad(120); %Axis 5
                deg2rad(-400) deg2rad(400)]; %Axis 6: Really Unlimited to (800ï¿½ default)

%maximum absolute speed of each joint rad/s or m/s
robot.velmax = [deg2rad(200); %Axis 1, rad/s
                deg2rad(200); %Axis 2, rad/s
                deg2rad(260); %Axis 3, rad/s
                deg2rad(360); %Axis 4, rad/s
                deg2rad(360); %Axis 5, rad/s
                deg2rad(450)];%Axis 6, rad/s
    
robot.accelmax=robot.velmax/0.1; % 0.1 is here an acceleration time
            
% end effectors maximum velocity
robot.linear_velmax = 2.5; %m/s



%base reference system
robot.T0 = eye(4);


%INITIALIZATION OF VARIABLES REQUIRED FOR THE SIMULATION
%position, velocity and acceleration
robot=init_sim_variables(robot);
robot.path = pwd;


% GRAPHICS
robot.graphical.has_graphics=1;
robot.graphical.color = [255 102 51]./255;
%for transparency
robot.graphical.draw_transparent=0;
%draw DH systems
robot.graphical.draw_axes=1;
%DH system length and Font size, standard is 1/10. Select 2/20, 3/30 for
%bigger robots
robot.graphical.axes_scale=1;
%adjust for a default view of the robot
robot.axis=[-2 2 -2 2 -2 2];
%read graphics files
robot = read_graphics(robot);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% DYNAMIC PARAMETERS
%   WARNING! These parameters do not correspond to the actual IRB 140
%   robot. They have been introduced to demonstrate the necessity of 
%   simulating the robot and should be used only for educational purposes
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
robot.has_dynamics=1;

%consider friction in the computations
robot.dynamics.friction=0;

%link masses (kg)
robot.dynamics.masses=[25 27 15 10 2.5 1.5];

%COM of each link with respect to own reference system
robot.dynamics.r_com=[0       0          0; %(rx, ry, rz) link 1
                     -0.05	 0.006	 0.1; %(rx, ry, rz) link 2
                    -0.0203	-0.0141	 0.070;  %(rx, ry, rz) link 3
                     0       0.019       0;%(rx, ry, rz) link 4
                     0       0           0;%(rx, ry, rz) link 5
                     0       0         0.032];%(rx, ry, rz) link 6

%Inertia matrices of each link with respect to its D-H reference system.
% Ixx	Iyy	Izz	Ixy	Iyz	Ixz, for each row
robot.dynamics.Inertia=[0      0.35	0   	0	0	0;
    .13     .524	.539	0	0	0;
    .066	.086	.0125	0	0	0;
    1.8e-3	1.3e-3	1.8e-3	0	0	0;
    .3e-3	.4e-3	.3e-3	0	0	0;
    .15e-3	.15e-3	.04e-3	0	0	0];



robot.motors=load_motors([5 5 5 4 4 4]);
%Speed reductor at each joint
robot.motors.G=[300 300 300 300 300 300];

%SPECIAL PARAMETERS TO SOLVE THE INVERSE KINEMATICS
robot.parameters.step_time=0.01;
%Error in XYZ to stop inverse kinematics
robot.parameters.epsilonXYZ=0.005;
%Error in Quaternion to stop inverse kinematics.
robot.parameters.epsilonQ=0.005;
robot.parameters.stop_iterations=500;

% 1: maximize manipulability.
% 0: standard.
% -1: minimize manipulability.
robot.maximize_manipulability = 0;
"""

import re
import numpy as np

parameter = re.compile(
    r"""
    robot\.(?P<param>[a-zA-Z0-9\._]+)
    \s*=\s*
    (?P<arg>
        (\[[^\]]+\])
        |
        ('[^']+')
    )
    """,
    re.VERBOSE,
)
# parameter2 = re.compile(r"""robot\.""", re.VERBOSE)

data = re.sub("%.*$", "", data, count=0, flags=re.MULTILINE)
data = re.sub(
    r"deg2rad\(\s*([0-9.-]+)\s*\)", r"\1*pi/180", data, count=0, flags=re.MULTILINE
)
robot = {}
for m in parameter.finditer(data):
    value = m.group("arg")
    if value.startswith("'"):
        # it's a string
        value = value.strip("'")
    elif value.startswith("["):
        # it's a vector/matrix
        # value = re.sub(r" +", ", ", value, count=0, flags=re.MULTILINE)
        value = value.strip("[]")
        try:
            if ";" in value:
                # it's a matrix
                first = True
                for s in value.split(";"):
                    x = np.fromstring(s, sep=" ")
                    if first:
                        print(s, x)
                        mat = x
                        first = False
                    else:
                        mat = np.vstack((mat, x))
                if len(mat.shape) == 2 and mat.shape[1] == 1:
                    mat = mat.flatten()
                value = mat

            else:
                # it's a vector
                print(value)
                value = np.fromstring(value, sep=" ")
        except:
            pass
    robot[m.group("param")] = value
for k, v in robot.items():
    print(f"{k:15s}: {v}")


x = np.fromstring("""0.128 0.176 -0.128 0.116 0.116 0.092""", sep=" ")
print(x)

x = np.fromstring(
    """0      0.35  0       0       0       0;
    .13     .524        .539    0       0       0
    .066        .086    .0125   0       0       0
    1.8e-3      1.3e-3  1.8e-3  0       0       0
    .3e-3       .4e-3   .3e-3   0       0       0
    .15e-3      .15e-3  .04e-3  0       0       0""",
    sep=" ",
)
print(x)
