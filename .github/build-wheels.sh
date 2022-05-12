#!/bin/bash
set -e -u -x

function repair_wheel {
    wheel="$1"
    if ! auditwheel show "$wheel"; then
        echo "Skipping non-platform wheel $wheel"
    else
        auditwheel repair "$wheel" --plat "$PLAT" -w /io/wheelhouse/
    fi
}

cd ./io

# Compile wheels
for PYBIN in /opt/python/cp3[6-10]*/bin; do
    "${PYBIN}/pip" install .
    "${PYBIN}/pip" install pytest
    "${PYBIN}/pip" wheel /io/ --no-deps -w wheelhouse/
done

# Bundle external shared libraries into the wheels
for whl in wheelhouse/*.whl; do
    repair_wheel "$whl"
done

# Install packages and test
#for PYBIN in /opt/python/cp3[6-8]*/bin/; do
#    "${PYBIN}/pip" install roboticstoolbox-python[dev,collision,vpython] --no-index -f /io/wheelhouse
#    ("${PYBIN}/pytest")
#done

ls ./wheelhouse
