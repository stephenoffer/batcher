"""Shared support for the optimizer/metadata property tests (not a test module).

These three concerns recur across ``test_prop_optimizer_result_invariance``,
``test_prop_metadata_fidelity`` and ``test_prop_rule_families`` and are lifted here so
the property files stay DRY:

- ``run_with_rules`` — execute a *logical* plan through Kyber with an explicit rule
  set (the full ``DEFAULT_REGISTRY`` vs an empty one), bypassing the public collect so
  the "full optimizer == no optimizer" invariance can be checked directly. The
  no-optimizer path reads every source column (a raw ``Scan`` still emits its whole
  schema — only the projection-pushdown *rewrite* prunes it, so honoring the pruned
  read against an un-pruned scan would desync).
- ``rowset`` / ``coerce`` — the order-independent, type-tolerant multiset view the
  differential harness uses (int↔float, Decimal→float, float rounding, date→iso), so a
  Batcher result and a DuckDB result compare on value not representation.

It is import-only (leading underscore keeps pytest from collecting it as a test).
"""

from __future__ import annotations

import datetime as _dt
from decimal import Decimal

import pyarrow as pa

from batcher.core import execute_local
from batcher.io.source import read_source
from batcher.kyber.optimizer import Optimizer
from batcher.kyber.registry import DEFAULT_REGISTRY

# The full built-in rule set (154 rules at time of writing) vs the empty baseline.
FULL_RULES = DEFAULT_REGISTRY.rules()
NO_RULES: list = []


def run_with_rules(logical, sources, rules) -> pa.Table:
    """Optimize ``logical`` over ``sources`` with exactly ``rules`` and execute it.

    With a non-empty rule set the projection/predicate pushdown has pruned each
    ``Scan``'s schema, so the read is pruned to match (``source_projections`` /
    ``source_predicates``). With no rules the scans still emit every column, so every
    column is read — otherwise the raw scan would reference a column the pruned read
    dropped. Either way the *result* must be identical; that is the invariant.
    """
    opt = Optimizer(sources=list(sources), rules=rules).optimize(logical)
    if rules:
        resolved = [
            read_source(s, opt.source_projections.get(i), opt.source_predicates.get(i))
            for i, s in enumerate(sources)
        ]
    else:
        resolved = [read_source(s, None, None) for s in sources]
    batches = execute_local(opt, resolved)
    schema = batches[0].schema if batches else _fallback_schema(logical)
    return pa.Table.from_batches(batches, schema=schema)


def _fallback_schema(logical) -> pa.Schema:
    """A best-effort empty schema when a run yields zero batches (int64 columns).

    Only used to build an empty ``pa.Table`` for the multiset compare; column *names*
    are what matter there, and both sides of a comparison hit this identically.
    """
    return pa.schema([(c, pa.int64()) for c in logical.available_columns()])


def coerce(v: object) -> object:
    """Type-tolerant scalar view: numbers→rounded float, dates→iso, bool/str unchanged."""
    if isinstance(v, bool):
        return v
    if isinstance(v, int):
        return float(v)
    if isinstance(v, float):
        return round(v, 9)
    if isinstance(v, Decimal):
        return round(float(v), 9)
    if isinstance(v, (_dt.date, _dt.datetime)):
        return v.isoformat()
    return v


def rowset(table: pa.Table) -> list[tuple]:
    """Order-independent, type-tolerant multiset view of a table (sorted tuples)."""
    cols = table.column_names
    rows = [tuple(coerce(r[c]) for c in cols) for r in table.to_pylist()]
    return sorted(rows, key=lambda t: tuple((v is None, str(type(v)), str(v)) for v in t))


def ordered_rows(table: pa.Table) -> list[tuple]:
    """Row-order-preserving, type-tolerant view (for ORDER BY / LIMIT comparisons)."""
    cols = table.column_names
    return [tuple(coerce(r[c]) for c in cols) for r in table.to_pylist()]
