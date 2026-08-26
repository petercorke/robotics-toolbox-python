# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys
import re

from sphinx_codeautolink import clean_ipython, clean_pycon

# Defined relative to configuration directory which is where this file conf.py lives
sys.path.append(os.path.abspath("exts"))

# -------- Project information -------------------------------------------------------#

project = "Robotics Toolbox for Python"
copyright = "2020-present, Jesse Haviland and Peter Corke"
author = "Jesse Haviland and Peter Corke"

# Parse version number out of pyproject.toml
_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
with open(os.path.join(_root, "pyproject.toml"), encoding="utf-8") as f:
    pyproject_src = f.read()
    m = re.search(r'^version\s*=\s*"([0-9.]*)"', pyproject_src, re.MULTILINE)
    version = m.group(1) if m else "unknown"

# -------- General configuration -----------------------------------------------------#

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "sphinx.ext.inheritance_diagram",
    "sphinx.ext.autosummary",
    "blockname",
    "sphinx.ext.intersphinx",
    "matplotlib.sphinxext.plot_directive",
    "format_example",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    # "scanpydoc.elegant_typehints",
    "sphinx_pyrunblock",
    "sphinx_favicon",
    "sphinxcontrib.bibtex",
    "sphinx_copybutton",
    "sphinx_codeautolink",
]

bibtex_bibfiles = ["refs.bib"]

autosummary_generate = True
autodoc_member_order = "bysource"

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

exclude_patterns = ["test_*"]

# Options for sphinx_pyrunblock, used for inline examples (same config API
# as its predecessor sphinx_autorun, which it renamed/replaced from)
# choose UTF-8 encoding to allow for Unicode characters, eg. ansitable
# Python session setup, turn off color printing for SE3, set NumPy precision
autorun_languages = {}
autorun_languages["pycon_output_encoding"] = "UTF-8"
autorun_languages["pycon_input_encoding"] = "UTF-8"
autorun_languages["pycon_runfirst"] = """
from spatialmath import SE3
SE3._color = False
import numpy as np
np.set_printoptions(precision=4, suppress=True)
from ansitable import ANSITable
ANSITable._color = False
"""

# -------- Options for HTML output ---------------------------------------------------#

html_theme = "sphinx_rtd_theme"

html_theme_options = {
    "logo_only": False,
    "prev_next_buttons_location": "None",
    "analytics_id": "G-11Q6WJM565",
    "style_external_links": False,
    "navigation_depth": 5,
}

html_logo = "../figs/RobToolBox_RoundLogoB.png"
html_last_updated_fmt = "%Y-%m-%d"
html_show_sourcelink = False
show_authors = True
html_show_sphinx = False

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static", "../../src/roboticstoolbox/blocks/Icons"]
html_css_files = [
    "css/custom.css",
]
default_role = "py:obj"

# -------- Options for LaTeX/PDF output ----------------------------------------------#

latex_engine = "xelatex"

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    "papersize": "a4paper",
    "fncychap": "\\usepackage{fncychap}",
}

# Use RVC book notation for maths
# see
# https://stackoverflow.com/questions/9728292/creating-latex-math-macros-within-sphinx
mathjax3_config = {
    "tex": {
        "macros": {
            # RVC Math notation
            #  - not possible to do the if/then/else approach
            #  - subset only
            "presup": [r"\,{}^{\scriptscriptstyle #1}\!", 1],
            # groups
            "SE": [r"\mathbf{SE}(#1)", 1],
            "SO": [r"\mathbf{SO}(#1)", 1],
            "se": [r"\mathbf{se}(#1)", 1],
            "so": [r"\mathbf{so}(#1)", 1],
            # vectors
            "vec": [r"\boldsymbol{#1}", 1],
            "dvec": [r"\dot{\boldsymbol{#1}}", 1],
            "ddvec": [r"\ddot{\boldsymbol{#1}}", 1],
            "fvec": [r"\presup{#1}\boldsymbol{#2}", 2],
            "fdvec": [r"\presup{#1}\dot{\boldsymbol{#2}}", 2],
            "fddvec": [r"\presup{#1}\ddot{\boldsymbol{#2}}", 2],
            "norm": [r"\Vert #1 \Vert", 1],
            # matrices
            "mat": [r"\mathbf{#1}", 1],
            "dmat": [r"\dot{\mathbf{#1}}", 1],
            "fmat": [r"\presup{#1}\mathbf{#2}", 2],
            # skew matrices
            "sk": [r"\left[#1\right]", 1],
            "skx": [r"\left[#1\right]_{\times}", 1],
            "vex": [r"\vee\left( #1\right)", 1],
            "vexx": [r"\vee_{\times}\left( #1\right)", 1],
            # quaternions
            "q": r"\mathring{q}",
            "fq": [r"\presup{#1}\mathring{q}", 1],
        }
    }
}

# -------- Options InterSphinx -------------------------------------------------------#

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "spatialmath": ("https://spatialmath-python.rai-inst.com/", None),
    "pgraph": ("https://petercorke.github.io/pgraph-python/", None),
}


# -------- Options Napoleon ----------------------------------------------------------#

# Include special members (like __membername__) with docstrings in
# the documentation
napoleon_include_special_with_doc = True

# Napoleon is still needed: tools/urdf/utils.py and urdf.py have genuine
# NumPy-style docstrings (vendored/adapted from an external URDF library)
# that depend on it. Everywhere else uses explicit reST fields directly.
# Bare NumPy-style section headers (Note, Examples, See Also, ...) mixed
# into otherwise-reST docstrings should use explicit directives
# (.. rubric:: Notes, etc.) instead — Napoleon's heuristic recognition of
# those bare headers can conflict with sphinx_autodoc_typehints (see
# https://github.com/petercorke/robotics-toolbox-python/issues/574).

# -------- Options AutoSummary -------------------------------------------------------#

# autodoc_default_flags = ["members"]

# -------- rst_epilog: shared substitutions available in all RST/docstrings ------#

rst_epilog = """
.. role:: raw-html(raw)
   :format: html
.. |BlockOptions| replace:: :raw-html:`<a href="https://petercorke.github.io/bdsim/internals.html?highlight=block%20__init__#bdsim.Block.__init__">common Block options</a>`
.. |GraphicsBlockOptions| replace:: :raw-html:`<a href="https://petercorke.github.io/bdsim/internals.html?highlight=graphicsblock%20__init__#bdsim.GraphicsBlock.__init__">common GraphicsBlock options</a>`
.. |ikargs| replace:: additional keyword arguments accepted by the underlying numerical IK solver -- ``ilimit``, ``slimit``, ``tol``, ``mask``, ``joint_limits``, ``seed``, ``k``, ``method``, ``kq``, ``km`` -- see :meth:`~roboticstoolbox.ETS.ikine_LM` for details of each
"""

# -------- Suppress common noisy warnings ----------------------------------------#

# Suppress "more than one target found" for short/common attribute names like
# n, m, robot, symbolic that appear across many classes, and duplicate-citation
# warnings for references (e.g. Yoshikawa85) defined in both a narrative page
# and a docstring pulled in from two classes.
#
# Note: "duplicate object description" (autodoc members documented both
# inline and via a separate IK/stubs/ page) is a different warning class
# with no type/subtype tag at all, so it can't be suppressed here — it's
# fixed at the source instead, via :no-index: in
# _templates/autosummary/method.rst.
suppress_warnings = [
    "ref.python",  # ambiguous cross-references (multiple targets for 'n', 'm', etc.)
    "ref.citation",  # duplicate citations (Yoshikawa85 etc in both arm_*.rst and docstrings)
    "codeautolink.match_block",
    "codeautolink.match_name",
    "config.cache",  # codeautolink_custom_blocks holds function refs, not picklable
]

# -------- sphinx-codeautolink options --------------------------------------------#

codeautolink_custom_blocks = {
    "pycon": clean_pycon,
    "ipython": clean_ipython,
    "ipython3": clean_ipython,
}
# Ensure pycon (Python console) blocks are included in the autolink search.
codeautolink_search_css_classes = ["highlight-python", "highlight-pycon"]

# -------- sphinx-copybutton options ----------------------------------------------#
# Strip interactive prompts (Python and shell) when users copy code snippets.

copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True
copybutton_only_copy_prompt_lines = False
copybutton_remove_prompts = True

# -------- Options favicon -------------------------------------------------------#

# create favicons online using https://favicon.io/favicon-converter/
favicons = [
    {
        "rel": "icon",
        "sizes": "16x16",
        "static-file": "favicon-16x16.png",
        "type": "image/png",
    },
    {
        "rel": "icon",
        "sizes": "32x32",
        "static-file": "favicon-32x32.png",
        "type": "image/png",
    },
    {
        "rel": "apple-touch-icon",
        "sizes": "180x180",
        "static-file": "apple-touch-icon.png",
        "type": "image/png",
    },
    {
        "rel": "android-chrome",
        "sizes": "192x192",
        "static-file": "android-chrome-192x192.png",
        "type": "image/png",
    },
    {
        "rel": "android-chrome",
        "sizes": "512x512",
        "static-file": "android-chrome-512x512.png",
        "type": "image/png",
    },
]
