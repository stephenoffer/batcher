"""Dotted-string option access over the frozen `Config` tree.

`Config` is a nested frozen dataclass, which is the right shape for the engine and the
wrong shape for a user who wants to change one number. This module is the flat view of
it: every leaf tunable addressed by a dotted path (``"execution.morsel_rows"``), with
`get_option` / `set_option` / `reset_option` / `option_context` mirroring the
`pandas.set_option` family and Spark's ``spark.conf.set``.

Nothing here holds state. Every setter composes `Config.replace` and hands the result to
`set_config`, so the precedence rules and the validation chokepoint in `config.py` apply
unchanged — a dotted setter is a spelling, not a second configuration system.
"""

from __future__ import annotations

import contextlib
import dataclasses
import difflib
import fnmatch
from collections.abc import Iterator

from batcher._internal.errors import ConfigError
from batcher.config.config import Config, active_config, config_context, set_config

__all__ = [
    "describe_options",
    "get_option",
    "option_context",
    "option_names",
    "reset_option",
    "set_option",
]


def _leaves(obj: object, prefix: str = "") -> Iterator[tuple[str, object]]:
    """Yield ``(dotted_path, value)`` for every scalar leaf of a nested config object."""
    for field in dataclasses.fields(obj):  # type: ignore[arg-type]
        value = getattr(obj, field.name)
        path = f"{prefix}{field.name}"
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            yield from _leaves(value, f"{path}.")
        else:
            yield path, value


def _all_paths() -> tuple[str, ...]:
    """Every settable dotted option path, in declaration order."""
    return tuple(path for path, _ in _leaves(Config()))


def _resolve(name: str) -> str:
    """Resolve a user-supplied option name to a canonical dotted path.

    Accepts the exact path, or an unambiguous trailing segment of one (so
    ``"morsel_rows"`` finds ``"execution.morsel_rows"``). Anything else raises
    `ConfigError` naming the closest known options.
    """
    paths = _all_paths()
    if name in paths:
        return name
    suffix_hits = [p for p in paths if p.endswith(f".{name}")]
    if len(suffix_hits) == 1:
        return suffix_hits[0]
    if len(suffix_hits) > 1:
        msg = f"option {name!r} is ambiguous; it matches {', '.join(sorted(suffix_hits))}"
        raise ConfigError(msg)
    raise ConfigError(_unknown_message(name, paths))


def _unknown_message(name: str, paths: tuple[str, ...]) -> str:
    """The did-you-mean error text for an unrecognized option name."""
    # Match on the full path first, then on the bare leaf name — a user who typed
    # "moprsel_rows" has no section to match against, and a user who typed the wrong
    # section ("exec.morsel_rows") is best served by the leaf hit.
    close = difflib.get_close_matches(name, paths, n=3, cutoff=0.6)
    leaf = name.rsplit(".", 1)[-1]
    leaf_names = {p.rsplit(".", 1)[-1] for p in paths}
    for hit in difflib.get_close_matches(leaf, sorted(leaf_names), n=3, cutoff=0.7):
        close += [p for p in paths if p.rsplit(".", 1)[-1] == hit]
    suggestions = list(dict.fromkeys(close))[:3]
    hint = f" Did you mean {', '.join(suggestions)}?" if suggestions else ""
    return (
        f"unknown config option {name!r}.{hint} "
        f"Call batcher.config.option_names() to list all {len(paths)} options, "
        f"or describe_options('<glob>') to search them."
    )


def _match(pattern: str) -> tuple[str, ...]:
    """Every option path matching `pattern` as a glob, or exactly, or by suffix."""
    paths = _all_paths()
    if any(ch in pattern for ch in "*?["):
        return tuple(p for p in paths if fnmatch.fnmatch(p, pattern))
    prefix_hits = tuple(p for p in paths if p == pattern or p.startswith(f"{pattern}."))
    return prefix_hits or (_resolve(pattern),)


def get_option(name: str) -> object:
    """Read one config option by its dotted path.

    Accepts the full path (``"execution.morsel_rows"``) or any unambiguous trailing
    segment of it (``"morsel_rows"``). Reads the config active in the current context,
    so it sees `set_option` and any enclosing `option_context` or `config_context`.

    Examples:
        .. doctest::

            >>> from batcher.config import get_option
            >>> get_option("execution.morsel_rows")
            16384
            >>> get_option("morsel_rows")  # unambiguous short form
            16384

    Args:
        name: The dotted option path, or an unambiguous trailing segment of one.

    Returns:
        The option's current value.

    Raises:
        ConfigError: If the name matches no option, or matches more than one.
    """
    obj: object = active_config()
    for part in _resolve(name).split("."):
        obj = getattr(obj, part)
    return obj


def set_option(*args: object, **options: object) -> None:
    """Set one or more config options process-wide by dotted path.

    Two spellings, both accepted: positional ``set_option("execution.morsel_rows", 4096)``
    for a single option (the `pandas` and Spark form), or keywords for several at once.
    Because a dotted path is not a Python identifier, the keyword form takes underscore
    paths (``execution_morsel_rows=4096``) or a dict splat.

    The whole batch is applied and validated as one config, so a bad value leaves the
    previous configuration in place rather than half-applying the change.

    Examples:
        .. doctest::

            >>> from batcher.config import get_option, reset_option, set_option
            >>> set_option("execution.morsel_rows", 4096)
            >>> get_option("execution.morsel_rows")
            4096
            >>> reset_option("execution.morsel_rows")

    Args:
        *args: A ``(name, value)`` pair, or an even-length flat sequence of them.
        **options: Further ``name=value`` options; underscores are read as path
            separators when the underscore form is not itself an option name.

    Raises:
        ConfigError: If a name is unknown or ambiguous, if the positional arguments are
            not name/value pairs, or if the resulting config fails validation.
    """
    set_config(_with_options(active_config(), _pairs(args, options)))


def reset_option(pattern: str = "*") -> None:
    """Reset every option matching `pattern` to its built-in default.

    Takes an exact path, a section prefix (``"optimizer"`` resets the whole section), or
    a glob (``"execution.*"``). The default is ``"*"``, which resets everything — the
    fastest way back to a known state after experimenting.

    Note that "default" means the value declared in the dataclass, not the value your
    ``BATCHER_*`` environment variables produced at import.

    Examples:
        .. doctest::

            >>> from batcher.config import get_option, reset_option, set_option
            >>> set_option("execution.morsel_rows", 4096)
            >>> reset_option("execution.*")
            >>> get_option("execution.morsel_rows")
            16384

    Args:
        pattern: An option path, a section prefix, or a glob over the option paths.

    Raises:
        ConfigError: If the pattern is not a glob and matches no known option.
    """
    defaults = dict(_leaves(Config()))
    set_config(_with_options(active_config(), [(p, defaults[p]) for p in _match(pattern)]))


@contextlib.contextmanager
def option_context(*args: object, **options: object) -> Iterator[None]:
    """Apply options for the duration of a `with` block, then restore them.

    Same argument forms as `set_option`, and the same all-or-nothing validation. Restores
    on the way out even if the block raises, and nests correctly because it is built on
    the same `ContextVar` as `config_context` — which also makes it safe under asyncio and
    per-thread rather than process-wide.

    Examples:
        .. doctest::

            >>> from batcher.config import get_option, option_context
            >>> with option_context("execution.morsel_rows", 1024):
            ...     get_option("execution.morsel_rows")
            1024
            >>> get_option("execution.morsel_rows")
            16384

    Args:
        *args: A ``(name, value)`` pair, or an even-length flat sequence of them.
        **options: Further ``name=value`` options, as in `set_option`.

    Yields:
        None. The options are active for the body of the block.

    Raises:
        ConfigError: If a name is unknown or the resulting config fails validation.
    """
    with config_context(_with_options(active_config(), _pairs(args, options))):
        yield


def option_names(pattern: str = "*") -> tuple[str, ...]:
    """Every option path matching `pattern`, for discovery and tab-completion.

    Examples:
        .. doctest::

            >>> from batcher.config import option_names
            >>> "execution.morsel_rows" in option_names()
            True
            >>> all(n.startswith("memory.") for n in option_names("memory.*"))
            True

    Args:
        pattern: A glob over the dotted option paths. Defaults to all of them.

    Returns:
        The matching option paths, in declaration order.
    """
    if any(ch in pattern for ch in "*?["):
        return tuple(p for p in _all_paths() if fnmatch.fnmatch(p, pattern))
    return _match(pattern)


def describe_options(pattern: str = "*") -> str:
    """A printable table of matching options with their current and default values.

    The searchable index of every tunable: run ``print(describe_options("spill"))`` to
    find the spill knobs without reading the source. Options whose current value differs
    from the default are flagged, so it doubles as "what have I changed?".

    Examples:
        .. doctest::

            >>> from batcher.config import describe_options
            >>> describe_options("execution.morsel_rows").startswith("execution.morsel_rows = ")
            True

        .. doctest::

            >>> from batcher.config import describe_options
            >>> "memory.spill_dir" in describe_options("spill")  # substring search
            True

    Args:
        pattern: A glob over the option paths, or a substring to search for. A bare word
            with no glob characters is treated as a section prefix when one matches, and
            otherwise as a substring search.

    Returns:
        One line per matching option, or a message naming the pattern if none matched.
    """
    matches = _search(pattern)
    if not matches:
        return f"no config options match {pattern!r} (see option_names() for all of them)"
    current = dict(_leaves(active_config()))
    defaults = dict(_leaves(Config()))
    lines = []
    for path in matches:
        value = current[path]
        marker = "" if value == defaults[path] else f"   (default {defaults[path]!r})"
        lines.append(f"{path} = {value!r}{marker}")
    return "\n".join(lines)


def _search(pattern: str) -> tuple[str, ...]:
    """Glob, prefix, or substring match — the forgiving lookup `describe_options` uses."""
    paths = _all_paths()
    if any(ch in pattern for ch in "*?["):
        return tuple(p for p in paths if fnmatch.fnmatch(p, pattern))
    prefixed = tuple(p for p in paths if p == pattern or p.startswith(f"{pattern}."))
    return prefixed or tuple(p for p in paths if pattern in p)


def _pairs(args: tuple[object, ...], options: dict[str, object]) -> list[tuple[str, object]]:
    """Normalize the positional/keyword calling conventions into name/value pairs."""
    if len(args) % 2:
        msg = (
            "set_option takes alternating name/value arguments, e.g. "
            f"set_option('execution.morsel_rows', 4096) — got {len(args)} argument(s)"
        )
        raise ConfigError(msg)
    pairs = [(str(args[i]), args[i + 1]) for i in range(0, len(args), 2)]
    pairs += [(_from_identifier(k), v) for k, v in options.items()]
    return pairs


def _from_identifier(key: str) -> str:
    """Map a keyword-argument spelling onto a dotted path.

    ``execution_morsel_rows`` and ``execution.morsel_rows`` both resolve, but a field
    whose own name contains underscores (nearly all of them) must not be mangled — so an
    exact suffix match is tried first and the underscore split is only a fallback.
    """
    if "." in key:
        return key
    paths = _all_paths()
    if any(p.endswith(f".{key}") or p == key for p in paths):
        return key
    dotted = [p for p in paths if p.replace(".", "_") == key]
    return dotted[0] if dotted else key


def _with_options(base: Config, pairs: list[tuple[str, object]]) -> Config:
    """Return `base` with every ``(dotted_name, value)`` applied, resolved and validated."""
    nested: dict[str, object] = {}
    for name, value in pairs:
        cursor = nested
        parts = _resolve(name).split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})  # type: ignore[assignment]
        cursor[parts[-1]] = value
    return _apply(base, nested)


def _apply(obj: Config, updates: dict[str, object]) -> Config:
    """Recursively rebuild a frozen config object with a nested dict of leaf overrides."""
    replaced: dict[str, object] = {}
    for name, value in updates.items():
        current = getattr(obj, name)
        if isinstance(value, dict) and dataclasses.is_dataclass(current):
            replaced[name] = _apply(current, value)  # type: ignore[arg-type]
        else:
            replaced[name] = value
    return dataclasses.replace(obj, **replaced)  # type: ignore[arg-type,return-value]
