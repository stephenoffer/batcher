"""Output row width derived from what Core measured, per plan signature.

The byte width of a relation decides broadcast eligibility, shuffle volume, spill prediction and
every `mem` axis in the cost model. For a **scan** it is well answered: learned per-column
averages, Arrow type priors, a connector's own `byte_size`, and the cheap per-column reading
`_learn_row_bytes` takes of every column a query touches. For an **intermediate** — the output of
a join, an aggregate, a projection — none of those apply, so the width is re-derived by summing
per-column priors through operators that reshape the row, and the error compounds with depth.

`cost/model.py` says so itself, in the one place it deliberately declines to use width:

    The gather really does cost `rows x width` ... so charging for width looks obviously
    right - and it was tried. It made TPC-H *worse* ... because `row_bytes` of an
    intermediate is itself an estimate ... The width signal is real but the width
    *estimate* is not yet good enough to rank on; it belongs here once intermediate widths
    are measured rather than inferred.

This is that measurement. The engine already reports each operator's `result_bytes` — the Arrow
byte size of its output — beside the rows it emitted, on every execution. Their ratio is the
operator's true output width.

## Why this shipped, was reverted, and is back

The first version of this module was correct in everything except its key, and that was enough
to make it wrong. A signature is structural, and `plan_signature` rendered every scan as the bare
token `["scan"]` — so two queries of the same shape over different relations shared one entry,
and a 4 KiB table's width was applied to a 16-byte one (257x, measured). It was reverted rather
than patched.

`kyber.signature` now carries the source's **data-stable** identity in that token, so relations
with durable identities no longer share. Two things follow, and both are load-bearing here:

* An **in-memory** relation still contributes no identity, deliberately (see
  `plan.source_stats.stable_source_key`), so same-shaped in-memory frames still share a
  signature. That residual is what the spread gate in `measured_fold` is for.
* Where the key *is* correct, the measurement is worth having: a scan of a 4 KiB payload column
  measured 4,108 B/row against a 44 B/row structural guess.

## Why a ratio, and why `max` at the consumer

`result_bytes` alone is a property of one run's input size; `result_bytes / n_actual` is a
property of the *shape*, and generalizes to a run over ten times the data the way a selectivity
does. It is deliberately unscoped by hardware: a row's byte width is a property of the data and
the schema, not of the machine that materialized it (`metadata.hardware_scope` draws exactly this
line). The estimator combines it with `max` rather than substitution — see `row_width`.

## What this does not fix

A signature is structural, so this learns "how wide is the output of *this shape*", not a model
of width. It cannot help a shape's first execution. And `result_bytes` measures the **output**, so
it says nothing about an operator's peak working set — that is `m_peak_bytes`, which Carbonite's
memory model consumes.
"""

from __future__ import annotations

from batcher.kyber.measured_fold import fold_measured
from batcher.metadata import MetadataHub

__all__ = ["measured_widths"]

# Widths beyond this are not believed. A row of Arrow data wider than a gibibyte is not a row, it
# is a mis-attributed measurement (an operator whose reported output rows were zero-ish while its
# bytes were not), and admitting one would swamp the mean for its signature. The real multimodal
# ceiling is far below: a 1080p RGB frame is ~6 MiB.
_MAX_CREDIBLE_ROW_BYTES = float(1 << 30)


def measured_widths(hub: MetadataHub) -> dict[str, float]:
    """`{signature: measured bytes per output row}` for shapes with enough consistent evidence.

    The mean of the most recent samples per signature, gated on their count and their spread by
    `measured_fold.fold_measured` — the same two questions `measured_selectivity` asks, of the
    same history. An arithmetic mean, because a width is a magnitude rather than a
    multiplicative error.

    The spread gate is what covers the residual this module's header describes: an in-memory
    relation contributes no identity to the scan token, so two same-shaped frames still share a
    signature, and a 4 KiB payload averaged with a 16-byte key row describes neither.

    Args:
        hub: The metadata hub holding the measured operator history.

    Returns:
        Measured output width in bytes per row, per plan signature.
    """
    return fold_measured(hub, _width_of, what="output row widths")


def _width_of(row: dict) -> float | None:
    """The output width one feedback row contributes, or `None` if it contributes nothing.

    Three rows are skipped and each for its own reason. A row that emitted nothing has no width
    to measure (and would divide by zero). A row whose `result_bytes` is zero is an engine that
    did not report the field rather than an operator that produced zero-byte rows, so it is
    absent evidence and not evidence of zero. A width past `_MAX_CREDIBLE_ROW_BYTES` is a
    mis-attribution, and one of those would dominate the mean for its signature.
    """
    rows_out = row.get("n_actual")
    result_bytes = row.get("result_bytes")
    if not isinstance(rows_out, (int, float)) or not isinstance(result_bytes, (int, float)):
        return None
    if rows_out <= 0 or result_bytes <= 0:
        return None
    width = float(result_bytes) / float(rows_out)
    return width if width <= _MAX_CREDIBLE_ROW_BYTES else None
