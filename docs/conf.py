# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import atexit
import os
import shutil
import sys
import tempfile
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

# Add the project root to the path so autodoc can find the modules
sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------

project = "Batcher"
author = "Batcher Contributors"
copyright = f"{datetime.now(tz=UTC):%Y}, Batcher Contributors"

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

# Doctests run real queries against real files, so they need two things the Sphinx doctest
# builder does not give them (tests/conftest.py does the equivalent for pytest, but that
# fixture does not apply here):
#
# 1. A scratch working directory. Examples that write a file use a bare relative name
#    (``bt.read.csv("late.csv")``), and without this every one of them drops an artifact
#    into whatever directory the build was launched from. Four such files were committed
#    to the repository root before this chdir existed.
# 2. The per-query event log turned off, so the build does not write JSON into the
#    builder's ``~/.batcher/logs``.
_DOCTEST_SCRATCH = tempfile.mkdtemp(prefix="batcher-doctest-")
atexit.register(shutil.rmtree, _DOCTEST_SCRATCH, True)

doctest_global_setup = f"""
import dataclasses, os
os.chdir({_DOCTEST_SCRATCH!r})
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
    # The contributor working records, grouped by kind. These are measured ledgers, audits,
    # program logs and RFCs: each carries an explicit "what this did not do" or "still
    # unmeasured" register, which is exactly what a published page must not carry. They are
    # excluded by directory rather than one line per file, so adding a record does not mean
    # remembering to exclude it.
    "architecture/internals/parity/*",
    "architecture/internals/audits/*",
    "architecture/internals/programs/*",
    "architecture/internals/rfcs/*",
    # Standalone formal paper, rendered to PDF by internals/generate_pdf.py rather
    # than as a site page. It carries its own internal cross-reference scheme.
    "architecture/internals/mathematical_foundations.md",
    # A code-checked audit of Batcher's architecture against DuckDB / Polars / Spark /
    # Flink / Daft / Snowflake: where it genuinely wins, where it loses, the
    # structural ceilings, and the claims the code does not support. A working record for
    # contributors (and deliberately blunt about our own marketing), not a site page.
    "architecture/internals/competitive_architecture.md",
    # The parts list behind that scorecard: which specific mechanisms DuckDB / Polars /
    # DataFusion / Spark / Daft have that Batcher does not, each cited to the
    # competitor file it was read from, plus the ranked build order. A contributor working
    # record, not a site page.
    "architecture/internals/competitor_technique_review.md",
    # The authoring guide for the diagram sources that live beside it (palette, the
    # rsvg-convert render step). A contributor note in an asset directory, not a page.
    "_static/diagrams/README.md",
]

# -- Options for HTML output -------------------------------------------------

html_theme = "furo"
html_static_path = ["_static"]
html_title = "Batcher"
html_favicon = "_static/favicon.png"
html_logo = "_static/logo.png"
html_css_files = ["custom.css"]

# The palette is taken from the logo, whose bars run cyan -> electric blue -> violet ->
# magenta. Light mode reads its links in the logo's blue, deepened just enough to hold
# AA contrast on white; dark mode reads them in the logo's cyan end, which is the part of
# the gradient that stays legible on indigo. The sidebar is a permanent indigo night in
# both modes, so the logo's glow always sits on the ground it was drawn for. The full
# gradient, and every structural style and animation, lives in custom.css.
_BRAND = "#2a3bf4"  # the logo's electric blue, deepened for text on white
_BRAND_DARK = "#4ed5f9"  # the logo's cyan-sky, for text on indigo

# An indigo-night sidebar, applied in both light and dark mode for a consistent shell.
_SIDEBAR = {
    "color-sidebar-background": "#0b0d26",
    "color-sidebar-background-border": "#1b1f4a",
    "color-sidebar-caption-text": "#9097c8",
    "color-sidebar-link-text": "#c9cdf0",
    "color-sidebar-link-text--top-level": "#f2f3ff",
    "color-sidebar-item-background--hover": "#1b1f4a",
    "color-sidebar-item-expander-background--hover": "#2a2f66",
    "color-sidebar-search-background": "#070919",
    "color-sidebar-search-background--focus": "#1b1f4a",
    "color-sidebar-search-border": "#2a2f66",
    "color-sidebar-search-foreground": "#e3e5fa",
    "color-sidebar-search-icon": "#6b72a8",
    "color-sidebar-brand-text": "#f2f3ff",
}

html_theme_options = {
    # The logo mark sits beside the "Batcher" wordmark, both linking home. custom.css lays
    # the two out as one row rather than Furo's default stacked, centered block.
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    "top_of_page_button": "edit",
    "light_css_variables": {
        "color-brand-primary": _BRAND,
        "color-brand-content": _BRAND,
        "color-admonition-title-background--note": "rgba(42, 59, 244, 0.09)",
        **_SIDEBAR,
    },
    "dark_css_variables": {
        "color-brand-primary": _BRAND_DARK,
        "color-brand-content": _BRAND_DARK,
        "color-background-primary": "#0e1030",
        "color-background-secondary": "#13163a",
        "color-background-hover": "#1b1f4a",
        "color-background-border": "#262b5c",
        "color-foreground-primary": "#e3e5fa",
        "color-foreground-secondary": "#9aa0cc",
        "color-code-background": "#13163a",
        "color-admonition-title-background--note": "rgba(78, 213, 249, 0.10)",
        **_SIDEBAR,
        "color-sidebar-background": "#080a1f",  # a touch darker than the content
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

# The generated API reference (docs/api/complete/) renders docstrings written in a
# light Markdown style. Treat a bare `backtick` span as inline code so single
# backticks don't need an explicit role, and suppress the docutils inline-markup
# warnings those Markdown-isms (e.g. `Dataset`s) would otherwise raise under -W.
default_role = "literal"
# `docutils`: the Markdown-ism inline-markup warnings the light docstring style raises.
# `sphinx_autodoc_typehints.forward_reference`: the whole codebase uses
# `from __future__ import annotations` plus `if TYPE_CHECKING:` imports (mandated by
# CLAUDE.md), so a signature like `Iterator[dict[str, np.ndarray]]` carries names that
# exist only for type checkers. sphinx-autodoc-typehints cannot resolve those at build
# time and warns once per name — under `-W` that fails the build for using a correct,
# required Python idiom. The annotation still renders as its source text; only the
# cross-link is lost. Suppressing the category keeps the docs build robust as new modules
# adopt the same idiom, instead of breaking on each newly-referenced type name.
suppress_warnings = ["docutils", "sphinx_autodoc_typehints.forward_reference"]
autodoc_member_order = "groupwise"
autodoc_typehints = "description"
autodoc_class_signature = "separated"
