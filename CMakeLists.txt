cmake_minimum_required(VERSION 3.15)
project(${SKBUILD_PROJECT_NAME} LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python COMPONENTS Interpreter Development.Module NumPy REQUIRED)

# ---------------------------------------------------------------------------
# _frne_c — Newton-Euler inverse dynamics (pure C)
# Wrapped by roboticstoolbox/frne.py facade which provides a pure-Python
# fallback when this extension is unavailable (e.g. Pyodide without WASM build).
# ---------------------------------------------------------------------------
python_add_library(_frne_c MODULE WITH_SOABI
    src/roboticstoolbox/core/vmath.c
    src/roboticstoolbox/core/ne.c
    src/roboticstoolbox/core/frne.c
)
target_include_directories(_frne_c PRIVATE
    src/roboticstoolbox/core
)

# ---------------------------------------------------------------------------
# _fknm_c — forward kinematics, Jacobian, IK (C++ with Eigen + NumPy C API)
# Eigen is vendored as header-only in src/roboticstoolbox/core/Eigen/
# Wrapped by roboticstoolbox/fknm.py facade which provides pure-Python
# fallbacks when this extension is unavailable.
# ---------------------------------------------------------------------------
python_add_library(_fknm_c MODULE WITH_SOABI
    src/roboticstoolbox/core/methods.cpp
    src/roboticstoolbox/core/ik.cpp
    src/roboticstoolbox/core/linalg.cpp
    src/roboticstoolbox/core/fknm.cpp
)
target_include_directories(_fknm_c PRIVATE
    src/roboticstoolbox/core
    ${Python_NumPy_INCLUDE_DIRS}
)

install(TARGETS _frne_c _fknm_c DESTINATION roboticstoolbox)
