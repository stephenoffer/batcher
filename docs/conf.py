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
    "sphinx_design",  # cards, grids, tabs, buttons (modern layout components)
    "sphinx_copybutton",  # one-click copy on code blocks
]

# Doctests run real queries; turn the per-query event log off so the docs build doesn't
# write JSON files into the builder's ~/.batcher/logs (tests/conftest.py does the same for
# pytest, but that fixture doesn't apply to the Sphinx doctest builder).
doctest_global_setup = """
import dataclasses
from batcher.config import active_config, set_config
_c = active_config()
set_config(_c.replace(observability=dataclasses.replace(_c.observability, event_log=False)))
"""

# MyST: enable the directives the landing/marketing pages use (card grids etc.).
myst_enable_extensions = ["colon_fence", "deflist", "tasklist", "attrs_inline"]

# Copy button: don't copy the `>>>`/`$` prompts or the expected-output comment lines.
copybutton_exclude = ".linenos, .gp, .go"
copybutton_copy_empty_lines = False

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
html_title = "Batcher"
html_favicon = "_static/favicon.svg"
html_css_files = ["custom.css"]

# Professional palette: a confident blue brand on light content, with a permanently
# dark slate sidebar in both modes (the enterprise-docs look). Dark mode uses lighter
# brand tints on slate surfaces. Structural styling + animations live in custom.css.
_BRAND = "#2563eb"  # blue-600 (light mode)
_BRAND_DARK = "#60a5fa"  # blue-400 (dark mode, on slate)

# A dark sidebar, applied in both light and dark mode for a consistent shell.
_SIDEBAR = {
    "color-sidebar-background": "#0f172a",
    "color-sidebar-background-border": "#1e293b",
    "color-sidebar-caption-text": "#94a3b8",
    "color-sidebar-link-text": "#cbd5e1",
    "color-sidebar-link-text--top-level": "#f1f5f9",
    "color-sidebar-item-background--hover": "#1e293b",
    "color-sidebar-item-expander-background--hover": "#334155",
    "color-sidebar-search-background": "#0b1120",
    "color-sidebar-search-background--focus": "#1e293b",
    "color-sidebar-search-border": "#334155",
    "color-sidebar-search-foreground": "#e2e8f0",
    "color-sidebar-search-icon": "#64748b",
    "color-sidebar-brand-text": "#f1f5f9",
}

html_theme_options = {
    # Show the "Batcher" project name as the sidebar brand (a text wordmark that
    # links home). No logo image — the mark + wordmark are styled in custom.css.
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    "top_of_page_button": "edit",
    "light_css_variables": {
        "color-brand-primary": _BRAND,
        "color-brand-content": _BRAND,
        "color-admonition-title-background--note": "rgba(37, 99, 235, 0.09)",
        **_SIDEBAR,
    },
    "dark_css_variables": {
        "color-brand-primary": _BRAND_DARK,
        "color-brand-content": _BRAND_DARK,
        "color-background-primary": "#0f172a",
        "color-background-secondary": "#131c31",
        "color-background-hover": "#1e293b",
        "color-background-border": "#243049",
        "color-foreground-primary": "#e2e8f0",
        "color-foreground-secondary": "#94a3b8",
        "color-code-background": "#131c31",
        **_SIDEBAR,
        "color-sidebar-background": "#0b1120",  # a touch darker than the content
    },
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

# The generated API reference (docs/api/complete.md) renders docstrings written in a
# light Markdown style. Treat a bare `backtick` span as inline code so single
# backticks don't need an explicit role, and suppress the docutils inline-markup
# warnings those Markdown-isms (e.g. `Dataset`s) would otherwise raise under -W.
default_role = "literal"
suppress_warnings = ["docutils"]
autodoc_member_order = "groupwise"
autodoc_typehints = "description"
autodoc_class_signature = "separated"
