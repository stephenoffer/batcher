"""The Batcher exception hierarchy.

A single rooted hierarchy, with each subclass mapped to the layer that raises it.
Native (Rust) errors are translated into the matching subclass at the PyO3 boundary.
Feedback/observability writes never raise into the hot path — they log and drop — so
none of these should ever surface from a metrics or learning call.

Two properties every class here shares, and that call sites may rely on:

**Structured fields, not message text.** `BatcherError` carries `message`, `suggestion`,
`available`, `hint`, and `doc`. Tooling and tests assert on those attributes; the
rendered `str()` is for humans and may be reworded. Asserting on a substring of a
message is how an error-quality change becomes a test failure for no reason.

**Catchable as what a Python user would reach for.** A user who has never read Batcher's
source writes ``except KeyError`` for a missing column and ``except ValueError`` for a
bad argument. The classes where that expectation is unambiguous also inherit the
builtin, so both spellings work. ``except BatcherError`` keeps catching everything —
the root is unchanged, and the builtin is always the *second* base, so `BatcherError`
wins the MRO and its `__str__` (not `KeyError`'s repr-quoting one) is what renders.

Three deliberate non-conversions: `IOError` does **not** inherit `OSError`,
`FormatError` does not inherit `ValueError`, and `ResourceError` does not inherit
`MemoryError`. The engine has thirty-odd narrow ``except OSError``/``except ValueError``
guards around third-party filesystem and parsing calls; widening these would let one of
them silently swallow a real Batcher failure and fall back to a wrong path, which is
precisely the failure mode that passes every gate while being wrong.
"""

from __future__ import annotations

from collections.abc import Iterable

from batcher._internal.errors.suggest import candidate_list
from batcher._internal.errors.suggest import suggestion as _suggestion

__all__ = [
    "AccessDeniedError",
    "BackendError",
    "BackpressureAbort",
    "BatcherError",
    "ColumnNotFoundError",
    "CommitError",
    "CompileError",
    "ConfigError",
    "DataQualityError",
    "DataWarning",
    "ExecutionError",
    "FatalShuffleError",
    "FormatError",
    "IOError",
    "MemoryBudgetExceededError",
    "MissingDependencyError",
    "OptimizationError",
    "PerformanceWarning",
    "PlanError",
    "PlanTooDeepError",
    "QueryCancelledError",
    "ResourceError",
    "RetryableShuffleError",
    "SchemaError",
    "SecurityWarning",
    "TransportError",
    "unknown_value",
]


class PerformanceWarning(UserWarning):
    """A correctness-neutral usage pattern that will run far slower than intended.

    Raised (via `warnings.warn`) for the documented foot-guns Ray Data users hit —
    e.g. a plain-function UDF on a GPU stage that reloads the model every batch. The
    query still runs and returns the right answer; the warning points at the faster
    spelling."""


class DataWarning(UserWarning):
    """The read succeeded but returned less than the data on disk appears to hold.

    The third member of the "correct but not what you meant" family, alongside
    `PerformanceWarning` and `SecurityWarning`. Those two warn about *how* a correct
    query runs; this one warns that the *result itself* is quietly narrower than the
    source — the case a wrong answer and a right one look identical, because the missing
    part never appears anywhere to be compared against.

    Raised (via `warnings.warn`) when reading a Hive-partitioned directory with a reader
    that does not recover partition columns, so ``k=1/part.parquet`` returns every row
    but no ``k``.

    Examples:
        .. doctest::

            >>> from batcher._internal.errors import DataWarning
            >>> issubclass(DataWarning, UserWarning)
            True
    """


class SecurityWarning(UserWarning):
    """A usage pattern that works but weakens security.

    Raised (via `warnings.warn`) when a crypto key is passed inline instead of as a
    reference (`env:NAME` / `file:PATH`), so the secret is embedded in the query and its
    serialized plan. The query is correct; the warning points at the safer spelling."""


class BatcherError(Exception):
    """Base class for every error Batcher raises.

    Beyond the message it carries the parts of a good error as separate fields, so a
    caller can build on them and a test can assert on them without matching prose:
    `suggestion` (the closest valid name), `available` (what the valid names are),
    `hint` (the next action to take), and `doc` (where to read more).

    `doc` is also attached as an exception *note*, so it renders on its own line at the
    bottom of a traceback — where a reader looks for "and now what" — instead of
    lengthening the one-line message.

    Examples:
        .. doctest::

            >>> from batcher._internal.errors import BatcherError
            >>> err = BatcherError("Cannot spill.", hint="Raise memory_limit.")
            >>> print(err)
            Cannot spill. Raise memory_limit.
            >>> err.hint
            'Raise memory_limit.'
    """

    def __init__(
        self,
        message: str = "",
        *,
        suggestion: str = "",
        available: Iterable[str] = (),
        available_label: str = "Available",
        hint: str = "",
        doc: str = "",
    ) -> None:
        """Build an error from its parts.

        Args:
            message: What failed, including the offending value. One sentence.
            suggestion: The closest valid name, already rendered (see `suggestion`).
            available: The valid alternatives. Rendered truncated, never in full.
            available_label: The alternatives' lead-in, e.g. ``"Available columns"``.
            hint: The next action the user should take. End it with a period.
            doc: A documentation path or URL, attached as an exception note.
        """
        super().__init__(message)
        self.message = message
        self.suggestion = suggestion
        self.available: tuple[str, ...] = tuple(available)
        self.available_label = available_label
        self.hint = hint
        self.doc = doc
        if doc:
            self.add_note(f"See: {doc}")

    def __str__(self) -> str:
        """The message with its suggestion, alternatives, and next action appended.

        Composed here rather than baked into `message` so that every field stays
        independently readable, and so a caller that wants only the headline has it.
        """
        parts = [self.message] if self.message else []
        if self.suggestion:
            parts.append(self.suggestion)
        listed = candidate_list(self.available, label=self.available_label)
        if listed:
            parts.append(listed)
        if self.hint:
            parts.append(self.hint)
        return " ".join(parts)


class PlanError(BatcherError, ValueError):
    """Invalid plan or schema mismatch (raised by `plan`/`api` at build time).

    Also a `ValueError`: it is raised when a user passes an argument the API cannot
    accept, which is the condition Python spells `ValueError`.

    Examples:
        .. doctest::

            >>> from batcher._internal.errors import BatcherError, PlanError
            >>> issubclass(PlanError, BatcherError) and issubclass(PlanError, ValueError)
            True
    """


class ColumnNotFoundError(PlanError, KeyError):
    """A column was referenced that the input schema does not have.

    Also a `KeyError`, because looking a column up by name is a mapping lookup and that
    is what a user reaches for. `column` and `available` carry the facts, so a caller
    building a better message on top does not re-parse this one.

    Examples:
        .. doctest::

            >>> from batcher._internal.errors import ColumnNotFoundError
            >>> err = ColumnNotFoundError.of("nmae", ["name", "age"])
            >>> err.column
            'nmae'
            >>> print(err)
            Unknown column 'nmae'. Did you mean 'name'? Available columns: 'age', 'name'
    """

    def __init__(self, message: str = "", *, column: str = "", **kwargs: object) -> None:
        """Build a column error.

        Args:
            message: What failed. Prefer building it with `of`.
            column: The column name that was not found.
            **kwargs: The `BatcherError` fields (`suggestion`, `available`, ...).
        """
        super().__init__(message, **kwargs)  # type: ignore[arg-type]
        self.column = column

    @classmethod
    def of(
        cls,
        column: str,
        available: Iterable[str],
        *,
        where: str = "",
        hint: str = "",
    ) -> ColumnNotFoundError:
        """The canonical unknown-column error for `column` against `available`.

        Args:
            column: The name that was not found.
            available: The columns the input does have, in schema order.
            where: Optional context, e.g. ``"in group_by"``, appended to the message.
            hint: A next action.

        Returns:
            A ready-to-raise error whose message names the column, the closest match,
            and the available columns (truncated for a wide schema).

        Examples:
            .. doctest::

                >>> from batcher._internal.errors import ColumnNotFoundError
                >>> err = ColumnNotFoundError.of("nmae", ["name", "age"])
                >>> err.column
                'nmae'
                >>> "Did you mean 'name'?" in str(err)
                True
        """
        columns = list(available)
        context = f" {where}" if where else ""
        return cls(
            f"Unknown column {column!r}{context}.",
            column=column,
            suggestion=_suggestion(column, columns),
            available=columns,
            available_label="Available columns",
            hint=hint,
        )


class ConfigError(BatcherError, ValueError):
    """An invalid configuration value (out of range, inconsistent limits).

    Raised by `Config.validate()` at the config entry points so a bad tunable fails
    early and clearly instead of being silently accepted and surfacing as a confusing
    runtime failure. Also a `ValueError` — a rejected tunable is a rejected argument.

    Examples:
        .. doctest::

            >>> from batcher._internal.errors import BatcherError, ConfigError
            >>> issubclass(ConfigError, BatcherError) and issubclass(ConfigError, ValueError)
            True
    """


class OptimizationError(BatcherError):
    """The optimizer (Kyber) failed to produce a valid physical plan.

    Examples:
        .. doctest::

            >>> from batcher._internal.errors import BatcherError, OptimizationError
            >>> issubclass(OptimizationError, BatcherError)
            True
    """


class ResourceError(BatcherError):
    """The resource manager (Carbonite) could not satisfy a request.

    Examples:
        .. doctest::

            >>> from batcher._internal.errors import BatcherError, ResourceError
            >>> issubclass(ResourceError, BatcherError)
            True
    """


class BackpressureAbort(ResourceError):
    """Execution was aborted because backpressure could not be relieved."""


class ExecutionError(BatcherError):
    """An operator failed at runtime (raised by the engine / Core).

    Examples:
        .. doctest::

            >>> from batcher._internal.errors import BatcherError, ExecutionError
            >>> issubclass(ExecutionError, BatcherError)
            True
    """


class BackendError(ExecutionError):
    """A specific execution backend failed; wraps the underlying error.

    Examples:
        .. doctest::

            >>> from batcher._internal.errors import ExecutionError, BackendError
            >>> issubclass(BackendError, ExecutionError)
            True
    """


class MissingDependencyError(BackendError, ImportError):
    """An optional dependency this feature needs is not installed.

    Also an `ImportError`, so ``except ImportError`` around an optional feature works
    the way it does for a plain import. `install` is the exact command to run, and is
    also the message's last sentence — a user should never have to guess the extra's
    name from a module name that differs from it (``hudi-rs`` vs ``hudi``).

    Examples:
        .. doctest::

            >>> from batcher._internal.errors import MissingDependencyError
            >>> err = MissingDependencyError.of(
            ...     feature="Hudi read support", provides="hudi-rs", extra="hudi"
            ... )
            >>> err.install
            "pip install 'batcher-engine[hudi]'"
            >>> print(err)
            Hudi read support requires hudi-rs, which is not installed. Install it with:
                pip install 'batcher-engine[hudi]'
    """

    def __init__(self, message: str = "", *, install: str = "", **kwargs: object) -> None:
        """Build a missing-dependency error.

        Args:
            message: What failed. Prefer building it with `of`.
            install: The exact shell command that installs the dependency.
            **kwargs: The `BatcherError` fields (`hint`, `doc`, ...).
        """
        super().__init__(message, **kwargs)  # type: ignore[arg-type]
        self.install = install

    @classmethod
    def of(
        cls, *, feature: str, provides: str, extra: str, doc: str = ""
    ) -> MissingDependencyError:
        """The canonical missing-extra error, naming the exact install command.

        Args:
            feature: What the user was trying to do, e.g. ``"Hudi read support"``.
            provides: The distribution that supplies it, as a user would recognize it.
            extra: The Batcher extra that installs it.
            doc: An optional documentation path.

        Returns:
            A ready-to-raise error carrying the install command in both `install` and
            the message.

        Examples:
            .. doctest::

                >>> from batcher._internal.errors import MissingDependencyError
                >>> err = MissingDependencyError.of(
                ...     feature="Hudi read support", provides="the hudi package", extra="hudi"
                ... )
                >>> err.install
                "pip install 'batcher-engine[hudi]'"
        """
        install = f"pip install 'batcher-engine[{extra}]'"
        return cls(
            f"{feature} requires {provides}, which is not installed.",
            install=install,
            hint=f"Install it with:\n    {install}",
            doc=doc,
        )


class CompileError(ExecutionError):
    """JIT compilation of a pipeline failed (the interpreter remains a fallback).

    Examples:
        .. doctest::

            >>> from batcher._internal.errors import ExecutionError, CompileError
            >>> issubclass(CompileError, ExecutionError)
            True
    """


class TransportError(BatcherError):
    """The distributed data plane (shared memory / Arrow Flight) failed.

    Examples:
        .. doctest::

            >>> from batcher._internal.errors import BatcherError, TransportError
            >>> issubclass(TransportError, BatcherError)
            True
    """


class IOError(BatcherError):
    """A source or sink failed to read, write, list, or open a path.

    Raised by the `io` layer, including for a missing optional filesystem/format
    dependency (e.g. reading ``s3://`` without the ``cloud`` extra installed).

    Deliberately **not** an `OSError`: the engine wraps third-party filesystem calls in
    narrow ``except OSError`` fallbacks, and one of those swallowing a Batcher IO
    failure would silently reroute a query instead of reporting it.

    Examples:
        .. doctest::

            >>> from batcher._internal.errors import BatcherError, IOError
            >>> issubclass(IOError, BatcherError)
            True
    """


class FormatError(IOError):
    """An unknown/unsupported IO format, or a file malformed for its format.

    Examples:
        .. doctest::

            >>> from batcher._internal.errors import IOError, FormatError
            >>> issubclass(FormatError, IOError)
            True
    """


class CommitError(IOError):
    """An atomic write commit failed.

    For example, a concurrent-writer conflict on a transactional (Delta/Iceberg)
    table, or a partial multi-file publish.

    Examples:
        .. doctest::

            >>> from batcher._internal.errors import IOError, CommitError
            >>> issubclass(CommitError, IOError)
            True
    """


class SchemaError(IOError):
    """Schemas could not be reconciled.

    For example, reading multiple files whose column types are incompatible under the
    requested ``schema_mode``, or an input whose schema diverges from an expected one.

    Examples:
        .. doctest::

            >>> from batcher._internal.errors import IOError, SchemaError
            >>> issubclass(SchemaError, IOError)
            True
    """


class DataQualityError(BatcherError, ValueError):
    """A data-quality expectation failed.

    Raised by ``ds.dq...fail()`` when one or more constraints have violating rows.
    Carries the per-constraint violation counts.

    Examples:
        .. doctest::

            >>> from batcher._internal.errors import BatcherError, DataQualityError
            >>> issubclass(DataQualityError, BatcherError), issubclass(DataQualityError, ValueError)
            (True, True)
    """

    def __init__(
        self, message: str, violations: dict[str, int] | None = None, **kwargs: object
    ) -> None:
        """Build a data-quality failure.

        Args:
            message: What failed.
            violations: Violating row count per constraint name.
            **kwargs: The `BatcherError` fields (`hint`, `doc`, ...).
        """
        super().__init__(message, **kwargs)  # type: ignore[arg-type]
        self.violations = violations or {}


class AccessDeniedError(BatcherError, PermissionError):
    """A principal referenced a table or column it holds no `SELECT` privilege on.

    Raised by `batcher.governance` while rewriting the plan, before the optimizer runs
    and before any data is read. It names the table and the columns denied, but never
    the *values* behind them. Also a `PermissionError`, the builtin for exactly this.

    Examples:
        .. doctest::

            >>> from batcher._internal.errors import BatcherError, AccessDeniedError
            >>> issubclass(AccessDeniedError, BatcherError)
            True
            >>> issubclass(AccessDeniedError, PermissionError)
            True
    """

    def __init__(
        self,
        message: str,
        *,
        table: str = "",
        columns: tuple[str, ...] = (),
        **kwargs: object,
    ) -> None:
        """Build an access-denied error.

        Args:
            message: What was denied.
            table: The table the principal may not read.
            columns: The columns it may not read.
            **kwargs: The `BatcherError` fields (`hint`, `doc`, ...).
        """
        super().__init__(message, **kwargs)  # type: ignore[arg-type]
        self.table = table
        self.columns = columns


def unknown_value(
    error: type[BatcherError],
    kind: str,
    name: object,
    candidates: Iterable[str] = (),
    *,
    label: str | None = None,
    hint: str = "",
    doc: str = "",
) -> BatcherError:
    """Build the canonical "you named something that does not exist" error.

    The one call every unknown-name site should make, so the phrasing, the suggestion
    cutoff, and the truncation of a wide candidate list are decided in one place rather
    than re-invented per subsystem.

    Args:
        error: The exception class to build, e.g. `FormatError` or `ConfigError`.
        kind: What was being looked up, singular and lowercase, e.g. ``"format"``.
        name: The value the user supplied. A non-string is rendered as one, which makes
            "you passed a list where a name goes" visible rather than confusing.
        candidates: The names that do exist.
        label: The alternatives' lead-in. Defaults to ``"Available <kind>s"``.
        hint: A next action.
        doc: A documentation path, attached as an exception note.

    Returns:
        An instance of `error`, ready to raise.

    Examples:
        .. doctest::

            >>> from batcher._internal.errors import FormatError, unknown_value
            >>> err = unknown_value(FormatError, "format", "parqet", ["parquet", "csv"])
            >>> print(err)
            Unknown format 'parqet'. Did you mean 'parquet'? Available formats: 'csv', 'parquet'
            >>> isinstance(err, FormatError)
            True
    """
    pool = [c for c in candidates if isinstance(c, str)]
    # Built from the *fields*, not from a pre-rendered string: `BatcherError.__str__`
    # already composes headline + suggestion + alternatives + hint, so formatting the
    # message with `unknown_message` here would print each of them twice.
    return error(
        f"Unknown {kind} {name!r}.",
        suggestion=_suggestion(name, pool) if isinstance(name, str) else "",
        available=pool,
        available_label=label or f"Available {kind}s",
        hint=hint,
        doc=doc,
    )


# Classified shuffle-fetch exceptions. They originate at the PyO3 boundary
# (`bc_py`), where the Rust transport's `classify()` verdict is preserved across the
# FFI: a `Retryable` fault (an unreachable/idle/cancelled peer) means worker loss —
# the reduce loop recomputes the source and retries — while a `Fatal` fault
# (decode/protocol/auth) propagates and fails the query fast, rather than burning
# recompute rounds on a fault a rerun cannot fix. They are re-exported here so the
# control plane has one import site. The pure-Python shims keep this module
# importable where the native extension is not built (config-only / unit-test
# imports); the distributed path that raises them needs the native engine anyway.
#
# `QueryCancelledError` rides it too: the executor raises it when it finds the cancellation
# flag set at a morsel boundary. It is deliberately not a subclass of `ExecutionError`'s
# "something went wrong" family in spirit — nothing went wrong, someone asked for this — but
# it is one in the hierarchy so a caller catching engine failures broadly still catches it
# rather than having a cancellation escape a `try` that meant to cover execution.
#
# `PlanTooDeepError` rides the same channel for a different reason: a plan nesting past
# the native depth limit used to overflow the deserializer's stack, which Rust reports as
# an *uncatchable* `SIGABRT`. It is now an error, and giving it a type is what lets a
# caller building plans in a loop catch it rather than lose the process.
#
# `MemoryBudgetExceededError` is the memory envelope's own refusal. Most stateful operators
# spill when they would exceed it; a few cannot, because they need one global order over the
# whole relation (a window with no PARTITION BY, an ASOF join with no `by` keys, the right
# side of a range join). Those raise instead of risking the process. It is the one execution
# failure with an obvious programmatic answer — raise the envelope, or re-plan so the
# non-spillable operator is not on the path — and it only ever reaches a caller who set a
# ceiling to begin with, so it earns a type rather than a message to match on.
try:
    from batcher._native import (
        FatalShuffleError,
        MemoryBudgetExceededError,
        PlanTooDeepError,
        QueryCancelledError,
        RetryableShuffleError,
    )
except ImportError:

    class RetryableShuffleError(TransportError):  # type: ignore[no-redef]
        """A shuffle fetch failed transiently (unreachable/idle peer) — recompute + retry."""

    class FatalShuffleError(TransportError):  # type: ignore[no-redef]
        """A shuffle fetch failed fatally (decode/protocol/auth) — retrying cannot help."""

    class PlanTooDeepError(PlanError):  # type: ignore[no-redef]
        """The plan nests deeper than the native stack can deserialize."""

    class QueryCancelledError(ExecutionError):  # type: ignore[no-redef]
        """The query was cancelled; it stopped at the next morsel boundary."""

    class MemoryBudgetExceededError(ResourceError):  # type: ignore[no-redef]
        """An operator that cannot spill needs more than the configured memory budget."""
