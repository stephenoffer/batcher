"""Filter selectivity derived from what Core measured, per plan signature.

`learning` records learned values *into* the keyed-parameter store; this module derives one
*out of* the per-operator feedback history instead, the way `learning._cardinality_corrections`
derives its q-error factors. Kyber cannot hook the recording — `core` builds the
`OperatorFeedback` rows and the two subsystems are independent — so reading the hub's history
is the only correct layering for it.

## The loop this closes

Core already records, for every filter it runs, the measured `rows_out / rows_in` under the
stable signature `annotate_ops` stamped on that operator. That happens on *every* execution,
profiled or not, because both executor paths pass `feedback=hub`.

Nothing consumed it. The estimator reads a `selectivity` key per signature
(`StatsEstimator._selectivity`, where a measured value always beats the structural guess), and
the only writer of that key was `learning.record_selectivity` — which is handed the **query's**
final row count and therefore guards on `_filter_over_scan`: the whole plan must be a filter
over a single scan, modulo row-preserving projections. Every filter underneath a join,
aggregate, sort or limit — 21 of the 22 TPC-H queries, and essentially every real analytical
query — re-derived a structural guess the engine had already measured, on every run forever.

Measured on TPC-H sf1: q12's `lineitem` filter measures 0.0869 selectivity (521,289 of
6,001,215 rows) and was estimated at 0.327 — the flat range constant — identically on ten
consecutive runs, with all six of its signed feedback rows present in the hub and none of
their signatures carrying a `selectivity` entry.

## What this does not fix

A signature is structural, so this learns "how selective is *this predicate shape* here", not
a correlation model. It also cannot help a shape's first execution, by construction: one run
must be measured before there is anything to consume.
"""

from __future__ import annotations

from batcher.kyber.measured_fold import fold_measured
from batcher.metadata import MetadataHub

__all__ = ["measured_selectivities"]


def measured_selectivities(hub: MetadataHub) -> dict[str, float]:
    """`{signature: measured selectivity}` for filters with enough recent observations.

    The mean of the most recent samples per signature, gated on their count and their spread
    by `measured_fold.fold_measured`, which documents both gates and the incremental fold
    they run over. An arithmetic mean is right here where the cardinality correction factor
    needs a geometric one — a selectivity is a probability, not a multiplicative error.

    The spread gate is what keeps this honest where one signature names two relations: see
    the scan-collision note in `kyber.signature`, and `_CORRECTABLE` in the estimator, which
    excludes `Scan` from learned row counts for the same reason.
    """
    return fold_measured(hub, _selectivity_of, what="filter selectivities")


def _selectivity_of(row: dict) -> float | None:
    """The selectivity one feedback row contributes, or `None` if it contributes nothing.

    Only a filter measures a selectivity, and a ratio outside [0, 1] is not one.
    """
    sel = row.get("selectivity")
    if row.get("kind") != "filter" or not isinstance(sel, (int, float)):
        return None
    ratio = float(sel)
    return ratio if 0.0 <= ratio <= 1.0 else None
