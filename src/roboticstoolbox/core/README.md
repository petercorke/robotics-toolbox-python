# Fast/optimized kinematics and dynamics

## Synopsis

## How it works


## Gotcha!

`fknm` works internally with C-major arrays.  All ETS transforms created internally have
this layout.  So to do results from fkine.  NumPy handles them perfectly in Python land, but
some fknm functions (ie. all IK functions) expect a row-major matrix and silently return an
incorrect result.

# Files

## Fast Forward Kinematics (fknm)


| File  |  Purpose  |
| ----  | --------- |
| Eigen | Vendored Eigen headers |
| fknm.cpp | Fast kinematics |
| fknm.h | " |
| ik.cpp | Fast inverse kinematics |
| ik.h | " |
| linalg.cpp | SE(3) matrix operations |
| linalg.h | " |
| methods.cpp | Jacobians, Hessians |
| methods.h | " |
| structs.cpp | ET, ETS mirror |
| structs.h | " |


## Fast Recursive Newton-Euler (frne)

These files were developed for a "MEX version" of robot dynamics in the
MATLAB version of the Robotics Toolbox. `ne.c` is virtually unchanged.

| File  |  Purpose  |
| ----  | --------- |
| frne.c | nanobind wrapper |
| frne.h 
| ne.c | implementation of RNE algorithm |
| vmath.c | Simple vector/matrix library |
| vmath.h | Simple vector/matrix library |

