cmake_minimum_required(VERSION 3.15)
project(${SKBUILD_PROJECT_NAME} LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python COMPONENTS Interpreter Development.Module NumPy REQUIRED)

# ---------------------------------------------------------------------------
# frne — Newton-Euler inverse dynamics (pure C)
# ---------------------------------------------------------------------------
python_add_library(frne MODULE WITH_SOABI
    src/roboticstoolbox/core/vmath.c
    src/roboticstoolbox/core/ne.c
    src/roboticstoolbox/core/frne.c
)
target_include_directories(frne PRIVATE
    src/roboticstoolbox/core
)

# ---------------------------------------------------------------------------
# fknm — forward kinematics, Jacobian, IK (C++ with Eigen + NumPy C API)
# Eigen is vendored as header-only in src/roboticstoolbox/core/Eigen/
# ---------------------------------------------------------------------------
python_add_library(fknm MODULE WITH_SOABI
    src/roboticstoolbox/core/methods.cpp
    src/roboticstoolbox/core/ik.cpp
    src/roboticstoolbox/core/linalg.cpp
    src/roboticstoolbox/core/fknm.cpp
)
target_include_directories(fknm PRIVATE
    src/roboticstoolbox/core
    ${Python_NumPy_INCLUDE_DIRS}
)

install(TARGETS frne fknm DESTINATION roboticstoolbox)
