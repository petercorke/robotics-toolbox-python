#!/bin/bash

set -ex

source builds/venv/bin/activate

# travis venv tests override python
PYTHON=${PYTHON:-python}
PIP=${PIP:-pip}

setup_base()
{
    $PIP install -v .
}

run_test()
{
  export PYTHONWARNINGS="ignore::DeprecationWarning:virtualenv"
}

setup_base
run_test
