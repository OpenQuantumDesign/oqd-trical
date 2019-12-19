import os
import sys
import sphinx_math_dollar
import sphinx_rtd_theme

sys.path.append(os.path.abspath(os.path.join(__file__, "../../..")))

project = "TrICal"
copyright = "2019, QITI"
author = "QITI"

exclude_patterns = []
extensions = [
    "nbsphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx_math_dollar",
    "sphinx_rtd_theme",
]
html_show_sourcelink = False
html_static_path = ["_static"]
html_theme = "sphinx_rtd_theme"
source_suffix = [".rst", ".ipynb"]
templates_path = ["_templates"]
