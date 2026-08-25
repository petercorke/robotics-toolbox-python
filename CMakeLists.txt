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

# Set -DRTB_BUILD_EXTENSIONS=OFF to skip both compiled extensions and build a
# genuinely pure-Python wheel instead -- both _fknm_c and _frne_c have
# complete, tested pure-Python fallbacks (see fknm.py/frne.py and
# tests/test_fknm_fallback.py), so this is a real, working configuration,
# not just a stub. Used for the Pyodide/JupyterLite wheel (see release.yml's
# build_pyodide job and pyproject.toml's scikit-build-core override): rather
# than cross-compiling for wasm32 via cibuildwheel's Pyodide platform (which
# ties the wheel to a specific Pyodide/Emscripten ABI version -- see git
# history on this file/release.yml for the JSPI/tag saga that caused),
# building on a normal runner with extensions off produces an ordinary
# py3-none-any wheel with no such coupling at all.
option(RTB_BUILD_EXTENSIONS "Build the compiled _fknm_c/_frne_c nanobind extensions" ON)

if(RTB_BUILD_EXTENSIONS)
    # -----------------------------------------------------------------------
    # _frne_c — Newton-Euler inverse dynamics
    # ne.c / vmath.c are pure C maths; frne_nb.cpp is the nanobind glue.
    # -----------------------------------------------------------------------
    nanobind_add_module(_frne_c
        src/roboticstoolbox/robot/cpp-extensions/vmath.c
        src/roboticstoolbox/robot/cpp-extensions/ne.c
        src/roboticstoolbox/robot/cpp-extensions/frne_nb.cpp
    )
    target_include_directories(_frne_c PRIVATE
        src/roboticstoolbox/robot/cpp-extensions
    )

    # -----------------------------------------------------------------------
    # _fknm_c — forward kinematics, Jacobian, Hessian, IK
    # Eigen is vendored as header-only in src/roboticstoolbox/ets/cpp-extensions/Eigen/
    # fknm_nb.cpp is the nanobind glue; maths lives in methods/ik/linalg.
    # -----------------------------------------------------------------------
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
