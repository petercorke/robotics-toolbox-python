cmake_minimum_required(VERSION 3.15)
project(${SKBUILD_PROJECT_NAME} LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Pyodide/Emscripten builds a genuine pure-Python wheel instead: both
# extensions have complete, tested pure-Python fallbacks (roboticstoolbox
# routes to them automatically whenever the compiled modules are absent --
# see fknm.py/frne.py), so there is nothing to gain from cross-compiling to
# wasm and a real cost (Emscripten toolchain fragility, ABI/Pyodide-version
# coupling -- see tech-debt.md's "aspire to a pure-Python wheel" writeup).
# CMAKE_SYSTEM_NAME is set to "Emscripten" by the toolchain file scikit-
# build-core picks up automatically when cibuildwheel's CIBW_PLATFORM=pyodide
# invokes this build.
if(NOT CMAKE_SYSTEM_NAME STREQUAL "Emscripten")

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
    src/roboticstoolbox/robot/cpp-extensions/vmath.c
    src/roboticstoolbox/robot/cpp-extensions/ne.c
    src/roboticstoolbox/robot/cpp-extensions/frne_nb.cpp
)
target_include_directories(_frne_c PRIVATE
    src/roboticstoolbox/robot/cpp-extensions
)

# ---------------------------------------------------------------------------
# _fknm_c — forward kinematics, Jacobian, Hessian, IK
# Eigen is vendored as header-only in src/roboticstoolbox/ets/cpp-extensions/Eigen/
# fknm_nb.cpp is the nanobind glue; maths lives in methods/ik/linalg.
# ---------------------------------------------------------------------------
nanobind_add_module(_fknm_c
    src/roboticstoolbox/ets/cpp-extensions/methods.cpp
    src/roboticstoolbox/ets/cpp-extensions/ik.cpp
    src/roboticstoolbox/ets/cpp-extensions/linalg.cpp
    src/roboticstoolbox/ets/cpp-extensions/fknm_nb.cpp
)
target_include_directories(_fknm_c PRIVATE
    src/roboticstoolbox/ets/cpp-extensions
)

install(TARGETS _frne_c _fknm_c DESTINATION roboticstoolbox)

endif()
