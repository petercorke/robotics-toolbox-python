#!/bin/bash

# Exit the script immediately if a command exits with a non-zero status,
# and print commands and their arguments as they are executed.
set -ex

# Print resource information
uname -a
free -m
df -h
ulimit -a

mkdir builds
pushd builds

# Build into own virtualenv
pip install -U virtualenv

virtualenv --python=python venv

source venv/bin/activate
python -V
gcc --version

popd

pip install --upgrade pip

pip install setuptools wheel





