"""Reader/writer keyword aliasing — one spelling table per format, one error shape.

Users arrive from pandas, Polars, and Spark, and each spells the same option
differently: `sep` / `separator` / `delimiter`, `na_values` / `null_values`,
`skiprows` / `skip_rows`, `partitionBy` / `partition_cols` / `partition_by`. A migrating
script should run, not fail on vocabulary. So each format declares one **canonical**
option name and the aliases that mean it, and this module folds the aliases in before
the format sees its keywords.

Three failure modes get distinct, actionable messages, because they need different
fixes:

* An **alias** resolves silently — it is a real spelling of a real option.
* An option Batcher deliberately does not have (pandas' `index_col`, because there is no
  index) raises a message that says *why* and what to do instead. Naming the concept is
  the fix; a "did you mean" would be actively misleading.
* An **unknown** keyword raises the canonical unknown-name error, with a "did you mean"
  ranked over the canonical names *and* their aliases — so a typo of an alias still
  suggests the alias the user was reaching for.

An alias that collides with the canonical option in the same call is an error rather
than a silent precedence rule: `read.csv(path, sep=";", delimiter="|")` has no correct
answer, and picking one quietly parses the file wrong.
"""

from __future__ import annotations

from typing import Any

from batcher._internal.errors import FormatError, unknown_value

__all__ = [
    "BASE_SINK_OPTIONS",
    "BASE_SOURCE_ALIASES",
    "BASE_SOURCE_OPTIONS",
    "OptionSpec",
    "split_base_options",
]

#: Keywords `FileSource` itself consumes, which every file-format spec must therefore also
#: accept — a format's spec validates its *extras*, not the whole constructor, and rejecting
#: `on_error=` as "unknown" because CSV's spec never listed it would be a worse error than
#: the `TypeError` it replaced. Pass these to `OptionSpec(base=...)`.
BASE_SOURCE_OPTIONS: tuple[str, ...] = (
    "columns",
    "files",
    "filesystem",
    "n_rows",
    "on_error",
    "schema_mode",
    "storage_options",
)

#: The same, for `FileSink`.
BASE_SINK_OPTIONS: tuple[str, ...] = ("filesystem", "storage_options")

#: Aliases for the base options, applied to every format that accepts them. `columns` and
#: `n_rows` are implemented once in `FileSource` for all formats, so their pandas spellings
#: have to be registered once too — leaving them to each format's own table is how
#: ``usecols`` worked on CSV and nowhere else.
BASE_SOURCE_ALIASES: dict[str, str] = {
    "usecols": "columns",
    "nrows": "n_rows",
    "num_rows": "n_rows",
}


class OptionSpec:
    """The keyword vocabulary of one format's reader or writer.

    Args:
        format_name: The format these options belong to, for error messages.
        canonical: The option names the format itself implements.
        base: Option names the format's *base class* consumes, accepted and passed
            through untouched. Use `BASE_SOURCE_OPTIONS` / `BASE_SINK_OPTIONS`.
        aliases: Mapping of accepted alias to the canonical name it means.
        unsupported: Mapping of a deliberately-absent option to the explanation
            raised when it is passed.
        ignored: Options accepted and discarded, mapped to the note explaining that
            they are a no-op here. These exist so a pandas/Polars script runs unchanged
            when the option only ever described that engine's internals.
        passthrough: When true, unknown keywords are forwarded untouched instead of
            raising. Connector-backed formats need this — their driver keywords are
            open-ended and not knowable here.

    Examples:
        .. doctest::

            >>> from batcher.io.base._options import OptionSpec
            >>> spec = OptionSpec("csv", canonical=("delimiter",), aliases={"sep": "delimiter"})
            >>> spec.resolve({"sep": ";"})
            {'delimiter': ';'}
    """

    __slots__ = ("_aliases", "_canonical", "_format", "_ignored", "_passthrough", "_unsupported")

    def __init__(
        self,
        format_name: str,
        *,
        canonical: tuple[str, ...] = (),
        base: tuple[str, ...] = (),
        aliases: dict[str, str] | None = None,
        unsupported: dict[str, str] | None = None,
        ignored: dict[str, str] | None = None,
        passthrough: bool = False,
    ) -> None:
        self._format = format_name
        self._canonical = tuple(canonical) + tuple(base)
        # A base alias applies only when the format actually accepts the option it names,
        # and a format's own table wins — so a format that spells `usecols` differently
        # can still say so, and one that takes no `columns` never advertises `usecols`.
        inherited = {k: v for k, v in BASE_SOURCE_ALIASES.items() if v in base}
        self._aliases = {**inherited, **(aliases or {})}
        self._unsupported = dict(unsupported or {})
        self._ignored = dict(ignored or {})
        self._passthrough = passthrough

    @property
    def accepted(self) -> tuple[str, ...]:
        """Every keyword this spec accepts, canonical names and aliases alike, sorted.

        Returns:
            The accepted keyword names, for an error message or a `dir()` listing.

        Examples:
            .. doctest::

                >>> from batcher.io.base._options import OptionSpec
                >>> spec = OptionSpec(
                ...     "csv", canonical=("delimiter",), aliases={"sep": "delimiter"}
                ... )
                >>> spec.accepted
                ('delimiter', 'sep')
        """
        return tuple(sorted({*self._canonical, *self._aliases, *self._ignored}))

    def resolve(self, opts: dict[str, Any]) -> dict[str, Any]:
        """Fold aliases into canonical names, rejecting unsupported and unknown keywords.

        Args:
            opts: The keywords the user passed.

        Returns:
            The same options keyed by canonical name, with ignored keywords dropped.

        Examples:
            .. doctest::

                >>> from batcher.io.base._options import OptionSpec
                >>> spec = OptionSpec(
                ...     "csv", canonical=("n_rows",), aliases={"nrows": "n_rows"}
                ... )
                >>> spec.resolve({"nrows": 10})
                {'n_rows': 10}
        """
        out: dict[str, Any] = {}
        seen: dict[str, str] = {}  # canonical name -> the spelling that set it
        for key, value in opts.items():
            if key in self._unsupported:
                raise FormatError(
                    f"{self._format}: {key!r} is not a Batcher option. {self._unsupported[key]}"
                )
            if key in self._ignored:
                continue
            canonical = self._aliases.get(key, key)
            if canonical not in self._canonical:
                if self._passthrough:
                    out[key] = value
                    continue
                raise unknown_value(
                    FormatError,
                    f"{self._format} option",
                    key,
                    self.accepted,
                    label="Accepted options",
                    hint=(
                        "pandas, Polars, and Spark spellings are accepted where the option "
                        "exists; see the reader's docstring for the full list."
                    ),
                )
            if canonical in seen:
                raise FormatError(
                    f"{self._format}: {seen[canonical]!r} and {key!r} are two spellings of the "
                    f"same option ({canonical!r}), and you passed both with no way to tell "
                    f"which you meant. Pass one of them."
                )
            seen[canonical] = key
            out[canonical] = value
        return out


def split_base_options(
    kwargs: dict[str, Any], base_names: set[str]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split constructor keywords into the base class's and the format's own.

    A base alias is folded in **first**, which is the whole reason this is a function
    rather than a dict comprehension at each call site: `usecols` is a spelling of
    `columns`, and `columns` belongs to `FileSource`. Splitting before aliasing left
    `usecols` looking like a CSV-specific keyword, so it was resolved to `columns` and then
    handed to the format's own option builder, which has no such field.

    Args:
        kwargs: The keywords the caller passed.
        base_names: The base class's parameter names, usually read from its signature so
            a keyword added to the base reaches this without an edit here.

    Returns:
        A ``(base_kwargs, own_kwargs)`` pair.

    Examples:
        .. doctest::

            >>> from batcher.io.base._options import split_base_options
            >>> split_base_options({"usecols": ["a"], "sep": ";"}, {"columns"})
            ({'columns': ['a']}, {'sep': ';'})
    """
    renamed = {
        BASE_SOURCE_ALIASES.get(k, k) if BASE_SOURCE_ALIASES.get(k, k) in base_names else k: v
        for k, v in kwargs.items()
    }
    return (
        {k: v for k, v in renamed.items() if k in base_names},
        {k: v for k, v in renamed.items() if k not in base_names},
    )
