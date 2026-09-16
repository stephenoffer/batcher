"""The shape of one migration-registry row: a competitor's name and what it becomes here.

Batcher keeps one spelling per capability and never adds a competitor's name as a second
one. A migrating user therefore needs somewhere that says, for every name they already
type, what the Batcher spelling is, whether it means the same thing, and if not, what is
different. That somewhere is the registry, and this module is its row type.

The registry is data rather than code for two reasons. It is large (every public name in
PySpark, Polars, Daft and Ray Data, several thousand rows), and it is read by consumers in
very different layers: the migration-error guidance on `Expr` (layer 1) and `Dataset`
(layer 5), the `batcher.migrate` codemod, the census harnesses and the documentation
generator. Holding it here, at layer 0 with no imports above `_internal`, is what lets all of
them read the same rows.

Each row carries a `Status`, and the status decides which other fields must be present.
`validate` enforces that, so a row missing the one field that makes it actionable (a
mismatch with no note saying what differs, a gap with no wave) fails at load time rather
than surfacing as a blank cell in the generated docs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

__all__ = ["ENGINES", "WAVES", "Mapping", "RegistryError", "Status", "validate"]

ENGINES = ("pyspark", "polars", "daft", "ray_data")

# The delivery waves a not-yet-parity row is scheduled into. `WF` is the foundations wave
# that has to land before the codemod can translate the rest.
WAVES = ("WF", *(f"W{i}" for i in range(15)))

_TARGET = re.compile(r"^(op:[a-z_]+|[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*)$")


class RegistryError(ValueError):
    """A registry file holds a row that cannot be used as written."""


class Status(StrEnum):
    """How a competitor's name relates to Batcher's surface.

    `CANONICAL` means the capability exists with the same semantics under the Batcher
    spelling in `batcher`. `ALIAS` means it is reachable today only through a second
    spelling that is being removed, and `batcher` names the spelling that stays. `PARAM`
    means the capability exists but lacks an option the competitor offers (`need`).
    `GAP` means there is no capability yet. `MISMATCH` means both engines have it and
    answer differently (`note`). `OUT_OF_SCOPE` is a deliberate decline (`reason`).
    """

    CANONICAL = "canonical"
    ALIAS = "alias"
    PARAM = "param"
    GAP = "gap"
    MISMATCH = "mismatch"
    OUT_OF_SCOPE = "out_of_scope"


# Which optional fields each status requires. `batcher` is the target spelling.
_REQUIRED: dict[Status, tuple[str, ...]] = {
    Status.CANONICAL: ("batcher",),
    Status.ALIAS: ("batcher",),
    Status.PARAM: ("batcher", "need", "wave"),
    Status.GAP: ("need", "wave"),
    Status.MISMATCH: ("batcher", "note", "wave"),
    Status.OUT_OF_SCOPE: ("reason",),
}

_FIELDS = frozenset({"status", "batcher", "template", "need", "note", "reason", "wave"})


@dataclass(frozen=True)
class Mapping:
    """One competitor name on one of its surfaces, and its relation to Batcher.

    Attributes:
        engine: The competitor, one of `ENGINES`.
        surface: The receiver the name is typed on, such as `DataFrame` or `Expr.str`.
        name: The competitor's spelling.
        status: How the name relates to Batcher's surface.
        batcher: The Batcher spellings the capability is reached through, as dotted paths
            rooted at a receiver (`Dataset.with_columns`, `Expr.str.starts_with`,
            `bt.coalesce`) or an operator (`op:add`). Empty for a pure gap.
        template: How a call is rewritten when the arguments do not carry over unchanged,
            in the codemod's argument DSL.
        need: What is missing, for a parameter gap or a gap.
        note: What differs, for a mismatch.
        reason: Why the name is declined, for an out-of-scope row.
        wave: The delivery wave that closes a parameter gap, gap, or mismatch.
    """

    engine: str
    surface: str
    name: str
    status: Status
    batcher: tuple[str, ...] = ()
    template: str | None = None
    need: str | None = None
    note: str | None = None
    reason: str | None = None
    wave: str | None = None


def validate(engine: str, surface: str, name: str, raw: dict[str, object]) -> Mapping:
    """Build a `Mapping` from one TOML entry, rejecting any row that cannot be acted on.

    Args:
        engine: The competitor the file belongs to.
        surface: The TOML table the entry sits under.
        name: The entry's key.
        raw: The entry's inline table.

    Returns:
        The validated row.

    Raises:
        RegistryError: On an unknown field or status, a missing required field, a
            malformed Batcher target, or an unknown wave.
    """
    where = f"{engine}:{surface}.{name}"
    unknown = set(raw) - _FIELDS
    if unknown:
        raise RegistryError(f"{where}: unknown field(s) {sorted(unknown)}")
    try:
        status = Status(str(raw.get("status")))
    except ValueError as exc:
        raise RegistryError(f"{where}: status {raw.get('status')!r} is not a Status") from exc
    for field in _REQUIRED[status]:
        if not raw.get(field):
            raise RegistryError(f"{where}: status {status.value} requires {field!r}")
    target = raw.get("batcher", ())
    targets = (target,) if isinstance(target, str) else tuple(target)  # type: ignore[arg-type]
    for t in targets:
        if not isinstance(t, str) or not _TARGET.match(t):
            raise RegistryError(f"{where}: batcher target {t!r} is not a dotted path")
    wave = raw.get("wave")
    if wave is not None and wave not in WAVES:
        raise RegistryError(f"{where}: wave {wave!r} is not one of {WAVES}")
    return Mapping(
        engine=engine,
        surface=surface,
        name=name,
        status=status,
        batcher=targets,
        template=_opt_str(raw, "template"),
        need=_opt_str(raw, "need"),
        note=_opt_str(raw, "note"),
        reason=_opt_str(raw, "reason"),
        wave=_opt_str(raw, "wave"),
    )


def _opt_str(raw: dict[str, object], key: str) -> str | None:
    value = raw.get(key)
    return None if value is None else str(value)
