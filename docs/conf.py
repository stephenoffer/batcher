# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import atexit
import os
import shutil
import sys
import tempfile
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
    # Build helpers, not documentation pages.
    "internals/README_PDF_GENERATION.md",
    # Standalone formal paper, rendered to PDF by internals/generate_pdf.py rather
    # than as a site page. It carries its own internal cross-reference scheme.
    "internals/mathematical_foundations.md",
    # An engineering audit ledger (findings fixed / still open), kept in-tree so the
    # analysis is recoverable. A working record for contributors, not a site page.
    "internals/audit_ledger.md",
    # The same: a measured audit of the connectors at TB/PB scale, kept in-tree as the
    # record behind the scale work. A working document, not a published page.
    "internals/connector_scale_audit.md",
    # The same again: the running ledger of the engine-wide defect hunt (found / fixed /
    # open). A contributor's working record, not a site page.
    "internals/bug_hunt_ledger.md",
    # The running index of improvements to the estimators' mathematics (cardinality,
    # sketches, cost, learning, resource control). A working record for anyone touching
    # one of those closed forms, not a page a user reads.
    "internals/math_improvements_ledger.md",
    # The running index of improvements/fixes/features that harden Batcher's streaming
    # workloads (triggers, watermarks, windowed/stateful aggregation, checkpointing,
    # sources/sinks, the morsel-streaming executor). A working record, not a user page.
    "internals/streaming_improvements_ledger.md",
    # The running index of improvements/features that make Batcher better at running
    # user-supplied UDFs and callables (map_batches/map, class model UDFs, the batch-inference
    # plane, and the threads/processes/GPU scheduling beneath). A working record, not a user page.
    "internals/udf_improvements_ledger.md",
    # Design proposal (RFC), not a published page — kept in-tree for contributors,
    # excluded from the site build until/unless its proposals are accepted.
    "internals/rfc-gpu-transport.md",
    # RFC: the streaming/pipelined executor tier that closes the single-node scale gap to
    # DuckDB (fixes the sf100 OOM, kills the high-selectivity gather tax) while preserving
    # every invariant. In-tree for contributors; not a site page until accepted.
    "internals/rfc-streaming-executor.md",
    # Design/migration note for porting the optimizer's cold hot-path passes to Rust.
    # A contributor working record (scope + dependency-ordered migration), not a site page.
    "internals/rust_optimizer_port.md",
    # A code-checked audit of Batcher's architecture against DuckDB / Polars / Spark /
    # Flink / Ray Data / Daft / Snowflake: where it genuinely wins, where it loses, the
    # structural ceilings, and the claims the code does not support. A working record for
    # contributors (and deliberately blunt about our own marketing), not a site page.
    "internals/competitive_architecture.md",
    # A code-checked parity scorecard against the Databricks stack (Catalyst/AQE, Photon,
    # Reyden/Lakehouse//RT) covering the optimizer, vectorized execution, the distributed
    # path, and the enterprise surface — plus the Databricks claims that are secondary-
    # sourced and must not be cited. A working record for contributors, not a site page.
    "internals/databricks_parity.md",
    # A code-checked audit of which Ray performance pitfalls Batcher avoids by design and
    # which it still inherits, read against a field-engineering corpus. A contributor
    # working record like the parity scorecards above, not a site page.
    "internals/ray_pitfall_parity.md",
    # The authoring guide for the diagram sources that live beside it (palette, the
    # rsvg-convert render step). A contributor note in an asset directory, not a page.
    "_static/diagrams/README.md",
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
