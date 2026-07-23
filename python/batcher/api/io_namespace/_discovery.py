"""Discoverability machinery shared by the `bt.read` and `ds.write` namespaces.

Both namespaces are wide objects whose whole value is the *list* of formats they carry,
and neither used to admit to it: ``repr(bt.read)`` printed an object address, and
``bt.read.parquett`` raised a bare `AttributeError` naming no alternative. A user who
cannot see the formats has to read the source to find out that ``mcap`` or ``lance``
exists at all.

This module supplies the three answers, once, so `Reader` and `Writer` each spend three
lines on them rather than duplicating the introspection:

* `namespace_repr` — the grouped, truncated listing shown by `repr`.
* `namespace_dir` — the tab-completion surface, formats guaranteed present.
* `unknown_attribute` — the canonical did-you-mean error for a misspelled format.

The grouping is read from the ``# --- Lakehouse ---`` section comments **already in**
`reader.py` / `writer.py`, via `inspect.getsource`. That is deliberate: a hand-written
group table beside those files is a second source of truth, and it would drift the first
time a format is added under a heading and nowhere else. Source introspection can fail
(a zipped or frozen distribution), so it degrades to one flat group rather than raising —
a repr is never worth an exception.
"""

from __future__ import annotations

import functools
import inspect
import os
import re
import textwrap
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from batcher._internal.errors import BatcherError

__all__ = ["PathLike", "namespace_dir", "namespace_repr", "unknown_attribute"]

#: What every path-taking argument on these namespaces actually accepts. The IO layer
#: normalizes `os.PathLike` and ``~`` itself (`io.base._paths`), so the honest annotation
#: is this union — the old bare `str` was a lie that made a `pathlib.Path` look unsupported.
PathLike = str | os.PathLike[str]

#: ``# --- Lakehouse ---------------`` — the section comments that group the methods.
_SECTION = re.compile(r"^\s*#\s*-{2,}\s*(.*?)\s*-{2,}\s*$")

#: A method definition at class-body indentation. The leading ``[A-Za-z]`` drops the
#: dunders and the private helpers, which is exactly the public surface we want to list.
_METHOD = re.compile(r"^    def ([A-Za-z]\w*)\s*\(")

#: Methods defined above the first section comment (``__call__``'s neighbours).
_UNGROUPED = "general"

#: Names to show per group before truncating. A repr is a signpost, not the reference;
#: `dir()` and the unknown-name error both carry the full list.
_PER_GROUP = 8

_WIDTH = 88


def _short_label(heading: str) -> str:
    """Shorten a section heading to the one word a user would scan for.

    ``"File / object-store formats (path-addressed)"`` → ``"file"``.
    """
    head = heading.split("/")[0].split("(")[0].strip().lower()
    return head or "other"


@functools.cache
def _grouped(cls: type) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """The class's public methods, grouped by the section comment they sit under.

    Cached per class: parsing the source is cheap but repeated `repr` calls in a REPL
    should not pay for it, and the source cannot change within a process.
    """
    try:
        source = inspect.getsource(cls)
    except (OSError, TypeError):
        source = ""
    groups: dict[str, list[str]] = {}
    current = _UNGROUPED
    for line in source.splitlines():
        section = _SECTION.match(line)
        if section is not None:
            current = _short_label(section.group(1))
            continue
        method = _METHOD.match(line)
        if method is not None:
            groups.setdefault(current, []).append(method.group(1))
    if not groups:
        # No source (or no sections): fall back to the live class dictionary, so the repr
        # still lists something useful rather than claiming the namespace is empty.
        names = sorted(n for n in dir(cls) if not n.startswith("_") and callable(getattr(cls, n)))
        groups = {_UNGROUPED: names} if names else {}
    return tuple((label, tuple(names)) for label, names in groups.items())


def method_names(cls: type) -> tuple[str, ...]:
    """Every public method name on `cls`, sorted — the candidate pool for a suggestion.

    Args:
        cls: The namespace class to introspect.

    Returns:
        The public method names, sorted.
    """
    return tuple(sorted({name for _, names in _grouped(cls) for name in names}))


def namespace_repr(obj: object, label: str) -> str:
    """Render the namespace as its grouped format listing.

    Args:
        obj: The namespace instance (`Reader` or `Writer`).
        label: How the namespace is spelled in user code, e.g. ``"bt.read"``.

    Returns:
        A short multi-line listing, each group truncated so the whole repr stays scannable.
    """
    groups = _grouped(type(obj))
    total = len({name for _, names in groups for name in names})
    lines = [f"<{label}: {total} methods"]
    for group, names in groups:
        # Declaration order, deliberately NOT alphabetical — it is not a bug to fix. The
        # files declare each family most-used first (`parquet`, `csv`, `json`, …), so this
        # is the one ordering that survives truncation with the formats people actually
        # want still visible. Sorting put `parquet` just past the cut in the `file` group,
        # which made the listing worst at answering the question it exists to answer.
        # `dir()` and the unknown-name error remain sorted, being lookup surfaces.
        shown = list(names)[:_PER_GROUP]
        rendered = ", ".join(shown)
        hidden = len(names) - len(shown)
        if hidden > 0:
            rendered += f", ... (+{hidden} more)"
        lines.extend(
            textwrap.wrap(
                f"{group}: {rendered}", width=_WIDTH, initial_indent="  ", subsequent_indent="    "
            )
        )
    return "\n".join(lines) + ">"


def namespace_dir(obj: object) -> list[str]:
    """The attribute names `dir()` should report for a namespace.

    The default already finds the methods; naming them explicitly pins the tab-completion
    surface to the same list `repr` and the unknown-name error use, so the three cannot
    disagree.

    Args:
        obj: The namespace instance.

    Returns:
        The sorted attribute names.
    """
    return sorted(set(object.__dir__(obj)) | set(method_names(type(obj))))


def unknown_attribute(obj: object, label: str, name: str) -> BatcherError | AttributeError:
    """Build the error for an attribute the namespace does not have.

    A `_`-prefixed name gets a plain `AttributeError`, and that is load-bearing rather
    than tidiness: `copy`, `pickle`, and IPython all probe for ``__deepcopy__``,
    ``__getstate__``, and ``_ipython_canary_method_should_not_exist_`` and expect a miss
    to be an `AttributeError`. Answering one of those probes with a `FormatError` would
    break `deepcopy` on a namespace that has nothing to deep-copy.

    Args:
        obj: The namespace instance the lookup missed on.
        label: How the namespace is spelled in user code, e.g. ``"bt.read"``.
        name: The attribute the user asked for.

    Returns:
        The exception to raise — never raised here, so the traceback starts at the
        `__getattr__` that failed.
    """
    if name.startswith("_"):
        return AttributeError(f"{type(obj).__name__!r} object has no attribute {name!r}")
    from batcher._internal.errors import FormatError, unknown_value

    return unknown_value(
        FormatError,
        "format",
        name,
        method_names(type(obj)),
        label=f"Available on {label}",
        hint=f"see repr({label}) or dir({label}) for the full list.",
    )
