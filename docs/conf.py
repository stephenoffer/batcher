# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

# Add the project root to the path so autodoc can find the modules
sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------

project = "Batcher"
author = "Batcher Contributors"
copyright = f"{datetime.now(tz=timezone.utc):%Y}, Batcher Contributors"

# Track the installed package version (set in the workspace Cargo.toml) instead of a
# hardcoded literal; fall back when the docs are built without the engine installed.
try:
    release = _pkg_version("batcher-engine")
except PackageNotFoundError:
    release = "0.1.0"
version = release

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.doctest",
    "sphinx_autodoc_typehints",
    "myst_parser",
]

# Napoleon settings for Google-style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_use_keyword = True
napoleon_attr_annotations = True

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": False,
    "exclude-members": "__weakref__",
    "show-inheritance": True,
}
autodoc_typehints = "description"
autodoc_class_signature = "separated"
autosummary_generate = True

# Type hints settings
typehints_fully_qualified = False
always_document_param_types = True
typehints_document_rtype = True

# MyST parser for Markdown support
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "tasklist",
]
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "pyarrow": ("https://arrow.apache.org/docs/", None),
    "ray": ("https://docs.ray.io/en/latest/", None),
}

root_doc = "index"
templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "requirements.txt",
    "Makefile",
    # Build helpers, not documentation pages.
    "internals/README_PDF_GENERATION.md",
    # Standalone formal paper, rendered to PDF by internals/generate_pdf.py rather
    # than as a site page. It carries its own internal cross-reference scheme.
    "internals/mathematical_foundations.md",
]

# -- Options for HTML output -------------------------------------------------

html_theme = "furo"
html_static_path = ["_static"]
html_title = "Batcher Documentation"
html_logo = None
html_favicon = None

html_theme_options = {
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    "top_of_page_button": "edit",
}

# -- Options for autodoc -----------------------------------------------------

# Mock imports for modules that may not be installed
autodoc_mock_imports = [
    "ray",
    "torch",
    "tensorflow",
    "cuda",
    "vllm",
]
