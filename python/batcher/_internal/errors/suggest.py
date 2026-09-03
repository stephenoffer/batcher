"""The one "did you mean ...?" engine, and the one unknown-name message shape.

Every layer of the engine has to tell a user they named something that does not exist:
an unknown column, an unknown function, an unknown IO format, an unknown cast dtype, an
unknown config key, an unknown metadata backend, an unknown registry entry. That message
is the single most-read thing Batcher ever prints, and it was being written out by hand
in each place — three separate `difflib.get_close_matches` calls with three different
cutoffs, three different phrasings, and no truncation, so a 400-column schema printed
400 column names into a traceback.

This module is that message, once. `did_you_mean` ranks candidates, `suggestion` renders
the phrase, `candidate_list` truncates the alternatives sensibly, and `unknown_message`
assembles the canonical four-part shape:

    what failed · the offending value · the closest matches · the valid alternatives

`_internal` (layer 0) is the only place this can live: the subsystems that need it
(`kyber`, `carbonite`, `core`, `governance`) may not import each other, so anywhere else
would mean copy-pasting it — which is exactly how the three divergent copies happened.
"""

from __future__ import annotations

import difflib
import functools
from collections.abc import Iterable

__all__ = [
    "absent_error",
    "candidate_list",
    "did_you_mean",
    "public_members",
    "suggestion",
    "unknown_message",
]

#: How many close matches to offer. More than three reads as a list, not a suggestion.
DEFAULT_MATCHES = 3

#: `difflib` similarity floor. 0.6 is `get_close_matches`' own default and is tuned so a
#: single-character typo matches while an unrelated name does not.
DEFAULT_CUTOFF = 0.6

#: How many alternatives to print before truncating. A wide table has hundreds of
#: columns; printing them all buries the actual error under its own context.
DEFAULT_LIMIT = 12


def did_you_mean(
    name: str,
    candidates: Iterable[str],
    *,
    n: int = DEFAULT_MATCHES,
    cutoff: float = DEFAULT_CUTOFF,
) -> tuple[str, ...]:
    """The candidates closest to `name`, best first.

    Three strategies, in precedence order, because each catches a mistake the others
    miss. A case-only difference (``"Name"`` for ``"name"``) is ranked first and always
    reported, since it is certainly what the user meant. Then `difflib`'s ratio, which
    catches transpositions and single-character typos. Then substring containment, which
    catches the abbreviation (``"cust"`` for ``"customer_id"``) that `difflib` scores far
    below its cutoff because the strings differ so much in length.

    Args:
        name: The name the user supplied.
        candidates: The names that do exist.
        n: The maximum number of suggestions to return.
        cutoff: The `difflib` similarity floor, between 0 and 1.

    Returns:
        Up to `n` candidates, best match first, with the candidates' original spelling.
        Empty when nothing is close enough, which is the signal to omit the suggestion
        rather than offer a misleading one.

    Examples:
        .. doctest::

            >>> from batcher._internal.errors import did_you_mean
            >>> did_you_mean("nmae", ["name", "age", "city"])
            ('name',)
            >>> did_you_mean("NAME", ["name", "age"])
            ('name',)
            >>> did_you_mean("cust", ["customer_id", "order_id"])
            ('customer_id',)
            >>> did_you_mean("zzzz", ["name", "age"])
            ()
    """
    if not name or not isinstance(name, str):
        return ()
    pool = [c for c in candidates if isinstance(c, str)]
    if not pool:
        return ()

    lowered = name.lower()
    # First key wins, so a later strategy never displaces an earlier one's ranking.
    ranked: dict[str, None] = {}

    for candidate in pool:
        if candidate.lower() == lowered and candidate != name:
            ranked[candidate] = None

    by_lower: dict[str, str] = {}
    for candidate in pool:
        by_lower.setdefault(candidate.lower(), candidate)
    for match in difflib.get_close_matches(lowered, list(by_lower), n=n, cutoff=cutoff):
        ranked.setdefault(by_lower[match], None)

    if len(ranked) < n:
        for candidate in pool:
            low = candidate.lower()
            if lowered in low or low in lowered:
                ranked.setdefault(candidate, None)

    return tuple(ranked)[:n]


def suggestion(
    name: str,
    candidates: Iterable[str],
    *,
    n: int = DEFAULT_MATCHES,
    cutoff: float = DEFAULT_CUTOFF,
) -> str:
    """The rendered ``Did you mean ...?`` sentence, or ``""`` when nothing is close.

    Returning an empty string rather than `None` lets a caller concatenate
    unconditionally, which is why no call site has to branch on whether a suggestion
    was found.

    Args:
        name: The name the user supplied.
        candidates: The names that do exist.
        n: The maximum number of suggestions to offer.
        cutoff: The `difflib` similarity floor, between 0 and 1.

    Returns:
        ``"Did you mean 'x'?"``, ``"Did you mean 'x' or 'y'?"``, or ``""``.

    Examples:
        .. doctest::

            >>> from batcher._internal.errors import suggestion
            >>> suggestion("nmae", ["name", "age"])
            "Did you mean 'name'?"
            >>> suggestion("qty", ["quantity", "qty_sold"])
            "Did you mean 'qty_sold'?"
            >>> suggestion("zzzz", ["name"])
            ''
    """
    matches = did_you_mean(name, candidates, n=n, cutoff=cutoff)
    if not matches:
        return ""
    rendered = " or ".join(repr(m) for m in matches)
    return f"Did you mean {rendered}?"


def candidate_list(
    candidates: Iterable[str],
    *,
    limit: int = DEFAULT_LIMIT,
    label: str = "Available",
    sort: bool = True,
) -> str:
    """The valid alternatives, truncated so a wide schema stays readable.

    A 400-column table must not print 400 names into a traceback: past `limit` the tail
    is replaced by a count, which tells the user the list was cut without drowning the
    error that prompted it.

    Args:
        candidates: The names that do exist.
        limit: How many to print before truncating.
        label: The lead-in noun, e.g. ``"Available columns"``.
        sort: Whether to sort. Pass False where declaration order is meaningful.

    Returns:
        ``"Available: 'a', 'b' (+3 more)"``, or ``""`` when there are no candidates —
        so a caller can concatenate it unconditionally.

    Examples:
        .. doctest::

            >>> from batcher._internal.errors import candidate_list
            >>> candidate_list(["b", "a"], label="Available columns")
            "Available columns: 'a', 'b'"
            >>> candidate_list(["a", "b", "c"], limit=2)
            "Available: 'a', 'b' (+1 more)"
            >>> candidate_list([])
            ''
    """
    names = [c for c in candidates if isinstance(c, str)]
    if not names:
        return ""
    if sort:
        names = sorted(names)
    shown = names[: max(limit, 1)]
    rendered = ", ".join(repr(n) for n in shown)
    hidden = len(names) - len(shown)
    if hidden > 0:
        rendered += f" (+{hidden} more)"
    return f"{label}: {rendered}"


def unknown_message(
    kind: str,
    name: object,
    candidates: Iterable[str] = (),
    *,
    limit: int = DEFAULT_LIMIT,
    label: str | None = None,
    sort: bool = True,
    hint: str | None = None,
) -> str:
    """The canonical unknown-name message, assembled from its four parts.

    The shape every "you named something that does not exist" error in Batcher uses:
    what failed, the offending value, the closest matches, and the valid alternatives.
    Keeping it here is what stops the phrasing drifting between the optimizer, the IO
    registry, and the expression builder.

    Args:
        kind: What was being looked up, singular and lowercase, e.g. ``"column"``.
        name: The value the user supplied. Rendered with `repr`, so a non-string
            (a common mistake in itself) is visible as one.
        candidates: The names that do exist.
        limit: How many alternatives to print before truncating.
        label: The alternatives' lead-in. Defaults to ``"Available <kind>s"``.
        sort: Whether to sort the alternatives.
        hint: A next action, appended last. End it with a period.

    Returns:
        A single-line message. Empty parts are omitted, never rendered as blanks.

    Examples:
        .. doctest::

            >>> from batcher._internal.errors import unknown_message
            >>> unknown_message("column", "nmae", ["name", "age"])
            "Unknown column 'nmae'. Did you mean 'name'? Available columns: 'age', 'name'"
            >>> unknown_message("format", "parqet", [], hint="See bt.read.")
            "Unknown format 'parqet'. See bt.read."
    """
    pool = [c for c in candidates if isinstance(c, str)]
    parts = [f"Unknown {kind} {name!r}."]
    if isinstance(name, str):
        phrase = suggestion(name, pool)
        if phrase:
            parts.append(phrase)
    listed = candidate_list(pool, limit=limit, label=label or f"Available {kind}s", sort=sort)
    if listed:
        # Terminate the list when a hint follows it, because the hint is a new sentence and
        # the parts are joined with a bare space. Without this the two run together —
        # "Available columns: 'id', 'text' Pass an existing column to column." — which is
        # what all 34 hint-passing call sites produced. Unterminated when it is last, so the
        # common no-hint message is unchanged.
        parts.append(f"{listed}." if hint else listed)
    if hint:
        parts.append(hint)
    return " ".join(parts)


@functools.lru_cache(maxsize=256)
def public_members(cls: type) -> tuple[str, ...]:
    """The public attribute names of `cls`, for a did-you-mean pool.

    Cached on the class, because `dir` is not free and the answer cannot change between
    two lookups on the same type: a class's member list is fixed once it is defined. The
    callers are attribute-error paths that the optimizer walks into thousands of times per
    query while probing for optional IR fields, so computing this per *failure* rather
    than per *type* was measurable on its own.

    Args:
        cls: The type whose public members to list.

    Returns:
        The names in ``dir(cls)`` that do not start with an underscore.

    Examples:
        .. doctest::

            >>> from batcher._internal.errors import public_members
            >>> "upper" in public_members(str)
            True
    """
    return tuple(n for n in dir(cls) if not n.startswith("_"))


def absent_error(
    label: str,
    name: str,
    table: dict[str, str],
    members: Iterable[str],
) -> AttributeError:
    """The `AttributeError` for a failed attribute lookup, with migration guidance.

    A migrant types the method they know from pandas, Polars, or PySpark, the normal
    lookup fails, and the traceback is the only documentation they read at that moment.
    `Dataset`, `Expr`, and `GroupBy` all render that traceback through this one function,
    so the phrasing never drifts between them: a name in `table` gets its curated "why it
    is absent, what to type instead" sentence, and anything else gets a `Did you mean
    ...?` computed against the object's real members.

    Args:
        label: The type name for the message, e.g. ``"Dataset"``, ``"Expr"``.
        name: The attribute name that was not found.
        table: The known-absent ecosystem APIs, mapping the name a migrant types to the
            guidance half of the message.
        members: The object's real public attribute names, for the did-you-mean fallback.

    Returns:
        An `AttributeError`. For a name in `table`, the curated guidance; otherwise the
        bare "no attribute" line plus a `Did you mean ...?` when a close member exists.

    Examples:
        .. doctest::

            >>> from batcher._internal.errors import absent_error
            >>> raise absent_error("Expr", "map_elements", {"map_elements": "Use bt.udf."}, [])
            Traceback (most recent call last):
            AttributeError: Expr has no attribute 'map_elements'. Use bt.udf.
            >>> raise absent_error("Expr", "meen", {}, ["mean", "count"])
            Traceback (most recent call last):
            AttributeError: Expr has no attribute 'meen'. Did you mean 'mean'?
    """
    if name in table:
        return AttributeError(f"{label} has no attribute {name!r}. {table[name]}")
    return AttributeError(_Guidance(label, name, tuple(members)))


def _absent_message(label: str, name: str, members: tuple[str, ...]) -> str:
    """The rendered "no attribute" line for a name with no curated guidance."""
    msg = f"{label} has no attribute {name!r}."
    phrase = suggestion(name, members)
    return f"{msg} {phrase}" if phrase else msg


class _Guidance:
    """An `absent_error` message rendered only when something actually reads it.

    This is an optimization with a correctness-shaped justification, so it is worth
    stating why it is not premature. Building the message is the expensive half of a
    failed attribute lookup: `suggestion` runs a `difflib` ratio against *every* member of
    the type, and an `Expr` has 319 of them. The lookup itself is cheap.

    That would be a fair price if the message were read, but on the dominant path it never
    is. A probe -- ``getattr(node, "left", None)``, ``hasattr(...)`` -- discards the
    exception without ever formatting it, and the optimizer probes constantly: one
    profile of a six-join plan spent **82% of the entire query** in `difflib`, building
    2,172 messages that were all thrown away. Deferring the render made that cost
    proportional to the errors a user actually sees, which is zero on a working query.

    It is deliberately *not* an `AttributeError` subclass. The exception class has to stay
    exactly `AttributeError` so a traceback still prints ``AttributeError: ...`` rather
    than a private class path -- which is what the doctests above, and every caller's
    ``pytest.raises(..., match=...)``, read. Making only ``args[0]`` lazy keeps the type,
    the traceback, `str`, and `repr` identical to the eager message.
    """

    __slots__ = ("_parts", "_text")

    def __init__(self, label: str, name: str, members: tuple[str, ...]) -> None:
        self._parts: tuple[str, str, tuple[str, ...]] | None = (label, name, members)
        self._text: str | None = None

    def _render(self) -> str:
        if self._text is None:
            assert self._parts is not None
            self._text = _absent_message(*self._parts)
            # Drop the member list once rendered: it can be a few hundred strings, and an
            # exception that is caught and stored would otherwise pin them.
            self._parts = None
        return self._text

    def __str__(self) -> str:
        return self._render()

    def __repr__(self) -> str:
        return repr(self._render())

    def __eq__(self, other: object) -> bool:
        return self._render() == other

    def __hash__(self) -> int:
        return hash(self._render())

    def __reduce__(self) -> tuple[type[str], tuple[str]]:
        # Pickled (across the Ray boundary, say) it becomes the plain string it renders to,
        # so nothing outside this module ever meets the wrapper.
        return (str, (self._render(),))
