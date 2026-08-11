"""The one place a `Source` becomes a `Dataset`.

Every session constructor funnels through `_scan`, which is what makes the
governance rewrite unbypassable: a scan that skipped it would read ungoverned
rows. Keep new constructors going through here rather than building a `Scan`
node themselves.
"""

from __future__ import annotations

import pyarrow as pa

from batcher.api.dataset import Dataset
from batcher.io.source import Source
from batcher.plan.logical import LogicalPlan, Scan
from batcher.plan.schema import SchemaRef
from batcher.plan.source_stats import stable_source_key


def _scan(source: Source) -> Dataset:
    """Build the `Dataset` for `source`, governed by the active security policy.

    The single place a source becomes a plan, and therefore the single place governance
    has to be applied for it to be unbypassable — see `api.security`.
    """
    from batcher.api.security import govern_scan

    # The scan carries *which relation* it reads, not just its position in the source list.
    # Without it every query's first scan is `source_id=0`, so anything keyed by plan
    # signature — a measured selectivity, a q-error correction — is shared across unrelated
    # tables of the same query shape. Only a *data-stable* identity qualifies; see
    # `stable_source_key` for why an in-memory source deliberately contributes nothing.
    plan: LogicalPlan = Scan(
        source_id=0,
        schema=SchemaRef.from_arrow(source.schema()),
        source_key=stable_source_key(source),
    )
    return Dataset(govern_scan(plan, source), sources=[source])


def _empty_batch(schema: pa.Schema) -> pa.RecordBatch:
    """A zero-row RecordBatch carrying `schema` (so empty inputs keep their types)."""
    return pa.RecordBatch.from_arrays([pa.array([], type=f.type) for f in schema], schema=schema)
