function tbxStruct = demos
#DEMOS   Return demo information to the MATLAB Demo
#        for the Robot Toolbox

# Copyright 2002 Jiri Canderle 
# $Revision: 1.1 $ $Date: 2002/05/26 23:02:29 $

if nargout == 0, demo toolbox; return; end

tbxStruct.Name='Robotics Toolbox';
tbxStruct.Type='toolbox';
tbxStruct.Help={ ...
    ' The Robotics Toolbox provides many functions '
    ' that are useful in robotics including such things '
    ' as kinematics, dynamics, and trajectory generation. '
    '  '
    ' The Toolbox is useful for simulation '
    ' as well as analyzing results from experiments with real robots. '
    '  '
    '  - Full support for modified (Craig s) Denavit-Hartenberg notation, '
    '    forward and inverse kinematics, Jacobians and '
    '    forward and inverse dynamics. '
    '  - Simulink blockset library and demonstrations '
    '    included MEX implementation of recursive Newton-Euler algorithm'
    '    written in portable C. '};
tbxStruct.DemoList={...
     'Transformations','puma560;rttrdemo','';
     'Trajectory','puma560;rttgdemo','';
     'Forward kinematics','puma560;rtfkdemo','';
     'Animation','puma560;rtandemo','';
     'Inverse kinematics','puma560;rtikdemo','';
     'Jacobians','puma560;rtjademo','';
     'Inverse dynamics','puma560;rtidemo','';
     'Forward dynamics','puma560;rtfddemo',''};
