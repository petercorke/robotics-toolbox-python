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

# Defined relative to configuration directory which is where this file conf.py lives
sys.path.append(os.path.abspath("exts"))

# -------- Project information -------------------------------------------------------#

project = "Robotics Toolbox for Python"
copyright = "2023, Jesse Haviland and Peter Corke"
author = "Jesse Haviland and Peter Corke"

# Parse version number out of setup.py
with open("../../setup.py", encoding="utf-8") as f:
    setup_py = f.read()
    m = re.search(r"version='([0-9\.]*)',", setup_py, re.MULTILINE)

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
    "sphinx_autorun",
    "sphinx_favicon",
]

autosummary_generate = True
autodoc_member_order = "bysource"

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

exclude_patterns = ["test_*"]

# Options for spinx_autorun, used for inline examples
# choose UTF-8 encoding to allow for Unicode characters, eg. ansitable
# Python session setup, turn off color printing for SE3, set NumPy precision
autorun_languages = {}
autorun_languages["pycon_output_encoding"] = "UTF-8"
autorun_languages["pycon_input_encoding"] = "UTF-8"
autorun_languages[
    "pycon_runfirst"
] = """
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
    "display_version": True,
    "prev_next_buttons_location": "None",
    "analytics_id": "G-11Q6WJM565",
    "style_external_links": False,
    "navigation_depth": 5,
}

html_logo = "../figs/RobToolBox_RoundLogoB.png"
html_last_updated_fmt = "%d-%b-%Y"
html_show_sourcelink = False
show_authors = True
html_show_sphinx = False

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
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
    "maketitle": "blah blah blah",
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
    "numpy": ("http://docs.scipy.org/doc/numpy/", None),
    "scipy": ("http://docs.scipy.org/doc/scipy/reference/", None),
    "matplotlib": ("http://matplotlib.sourceforge.net/", None),
    "spatialmath": ("https://petercorke.github.io/spatialmath-python/", None),
}


# -------- Options Napoleon ----------------------------------------------------------#

# Include special members (like __membername__) with docstrings in
# the documentation
napoleon_include_special_with_doc = True

napoleon_custom_sections = ["Synopsis"]

# -------- Options AutoSummary -------------------------------------------------------#

# autodoc_default_flags = ["members"]
autosummary_generate = True

# -------- Options favicon -------------------------------------------------------#

html_static_path = ["_static"]
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
        "static-file": "android-chrome-192x192.png ",
        "type": "image/png",
    },
    {
        "rel": "android-chrome",
        "sizes": "512x512",
        "static-file": "android-chrome-512x512.png ",
        "type": "image/png",
    },
]
