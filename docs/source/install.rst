************
Installation
************

You will need Python >= 3.10.

Using pip
=========

Install a snapshot from PyPI::

    pip install roboticstoolbox-python

Available extras:

- ``swift`` install `Swift <https://github.com/jhavl/swift>`_, a web-based visualizer
- ``qp`` install quadratic-programming IK dependencies (``qpsolvers``, ``quadprog``)
- ``collision`` install collision checking with `coal <https://github.com/coal-library/coal>`_ and ``trimesh``
- ``tool`` install ``IPython`` and ``pygments``, needed to run the ``rtbtool`` interactive shell
- ``all`` install ``swift``, ``qp``, ``collision``, and ``tool``

.. warning:: ``coal`` does not publish Windows wheels on PyPI, so the
    ``collision``/``all`` extras skip it on Windows and collision checking
    is unavailable via ``pip`` there. It's available via
    ``conda install -c conda-forge coal-python`` if needed. Everything else
    in the Toolbox works normally on Windows.

Put extras in a comma-separated list::

    pip install roboticstoolbox-python[optionlist]

Install matrix:

- Core only::

    pip install roboticstoolbox-python

- Swift visualizer only::

    pip install roboticstoolbox-python[swift]

- QP solver dependencies only::

    pip install roboticstoolbox-python[qp]

- Collision checking dependencies only::

    pip install roboticstoolbox-python[collision]

- ``rtbtool`` interactive shell dependencies only::

    pip install roboticstoolbox-python[tool]

- Everything (swift + qp + collision + tool)::

    pip install roboticstoolbox-python[all]

- Multiple extras explicitly::

    pip install roboticstoolbox-python[swift,qp,collision]

From GitHub
===========

To install the bleeding-edge version from GitHub::

    git clone https://github.com/petercorke/robotics-toolbox-python.git
    cd robotics-toolbox-python
    pip install -e .
