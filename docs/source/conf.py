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
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx_math_dollar",
    "sphinx_rtd_theme",
]
html_show_sourcelink = False
html_static_path = ["_static"]
html_theme = "sphinx_rtd_theme"
intersphinx_mapping = {
    "python": ("https://docs.python.org/", None),
    "numpy": ("http://docs.scipy.org/doc/numpy/", None),
    "scipy": ("http://docs.scipy.org/doc/scipy/reference", None),
}
templates_path = ["_templates"]
