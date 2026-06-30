cmake_minimum_required(VERSION 3.15)
project(${SKBUILD_PROJECT_NAME} LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Locate Python (interpreter needed to find nanobind cmake dir)
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

# Locate nanobind via the installed Python package
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import nanobind; print(nanobind.cmake_dir())"
    OUTPUT_VARIABLE NB_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
find_package(nanobind CONFIG REQUIRED HINTS "${NB_DIR}")

# ---------------------------------------------------------------------------
# _frne_c — Newton-Euler inverse dynamics
# ne.c / vmath.c are pure C maths; frne_nb.cpp is the nanobind glue.
# ---------------------------------------------------------------------------
nanobind_add_module(_frne_c
    src/roboticstoolbox/core/vmath.c
    src/roboticstoolbox/core/ne.c
    src/roboticstoolbox/core/frne_nb.cpp
)
target_include_directories(_frne_c PRIVATE
    src/roboticstoolbox/core
)

# ---------------------------------------------------------------------------
# _fknm_c — forward kinematics, Jacobian, Hessian, IK
# Eigen is vendored as header-only in src/roboticstoolbox/core/Eigen/
# fknm_nb.cpp is the nanobind glue; maths lives in methods/ik/linalg.
# ---------------------------------------------------------------------------
nanobind_add_module(_fknm_c
    src/roboticstoolbox/core/methods.cpp
    src/roboticstoolbox/core/ik.cpp
    src/roboticstoolbox/core/linalg.cpp
    src/roboticstoolbox/core/fknm_nb.cpp
)
target_include_directories(_fknm_c PRIVATE
    src/roboticstoolbox/core
)

install(TARGETS _frne_c _fknm_c DESTINATION roboticstoolbox)
