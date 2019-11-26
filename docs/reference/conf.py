# -*- coding: utf-8 -*-
#

import sys
import os

sys.path.insert(0, os.path.abspath("../../"))

extensions = ["sphinx.ext.autodoc", "sphinx_rtd_theme"]
source_suffix = ".rst"
master_doc = "index"
project = u"Hyperactive"
copyright = u"Simon Blanke, simon.blanke@yahoo.com"
exclude_patterns = ["_build"]
pygments_style = "sphinx"
html_theme = "default"
autoclass_content = "both"
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
