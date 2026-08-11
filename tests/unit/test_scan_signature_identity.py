"""A plan signature names which relation it reads, not just its position.

`Scan.source_id` is an index into *this plan's* own source list, so the first source of every
query is `0`. `kyber.signature` therefore rendered every scan as the bare token `["scan"]`, and
any learned value keyed by a plan signature was shared across unrelated tables of the same query
shape. Measured before this fix: `k < 40` keeping 40 of 20,000 rows in one table taught the
optimizer to estimate a second table's identical filter at **40 rows against an actual 20,000** —
a 500x error, and worse than the structural guess it replaced.

The codebase already knew the shape of this. `signature._struct` names it "the scan-collision
defect" in its `MapBatches` arm and fixes it there by carrying the UDF's identity;
`estimator._CORRECTABLE` works around it by excluding `Scan` from learned row counts. Both are
per-consumer patches for a defect in the key. This fixes the key.

Two properties have to hold together, and they pull in opposite directions:

* different relations must not share — the defect above;
* the **same** relation must still share across runs and across sizes, because a selectivity is
  a *ratio* and its whole value is that it survives the table growing.

That is why the token carries only a **data-stable** identity. A file path stays put as the file
grows, so both properties hold. An in-memory relation identifies by schema plus row count, which
satisfies neither — two unrelated frames of the same shape collide, and one frame grown by a row
does not match itself — so it contributes nothing and keeps the old shared token. The residual
in-memory collision is covered by the confidence gate in `kyber.measured_selectivity`.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import batcher as bt
from batcher.kyber.signature import plan_signature
from batcher.plan.logical import Scan
from batcher.plan.schema import SchemaRef

pytestmark = pytest.mark.unit

_ROWS = 2_000


@pytest.fixture
def two_files(tmp_path):
    """Two parquet files of the same schema whose filters have very different selectivity."""
    selective = tmp_path / "selective.parquet"
    permissive = tmp_path / "permissive.parquet"
    pq.write_table(pa.table({"k": list(range(_ROWS))}), selective)
    pq.write_table(pa.table({"k": [i % 39 for i in range(_ROWS)]}), permissive)
    return str(selective), str(permissive)


def test_two_relations_of_the_same_shape_get_different_signatures(two_files):
    """The defect, stated as the case it produces."""
    selective, permissive = two_files
    one = bt.read_parquet(selective).filter(bt.col("k") < 40)
    two = bt.read_parquet(permissive).filter(bt.col("k") < 40)
    assert plan_signature(one._plan) != plan_signature(two._plan)


def test_the_same_relation_keeps_one_signature_across_handles(two_files):
    """The property that must survive: learning has to accumulate for one table."""
    selective, _ = two_files
    first = bt.read_parquet(selective).filter(bt.col("k") < 40)
    second = bt.read_parquet(selective).filter(bt.col("k") < 40)
    assert plan_signature(first._plan) == plan_signature(second._plan)


def test_literal_values_are_still_normalized(two_files):
    """The generalization the signature exists for is untouched."""
    selective, _ = two_files
    frame = bt.read_parquet(selective)
    assert plan_signature(frame.filter(bt.col("k") < 40)._plan) == plan_signature(
        frame.filter(bt.col("k") < 41)._plan
    )


def test_one_tables_measurement_no_longer_answers_for_another(two_files):
    """End to end through the real engine and the real learning loop."""
    import batcher.core as core
    from batcher.kyber.cardinality import CardinalityEstimator
    from batcher.kyber.learning import load_learned_stats

    selective, permissive = two_files
    hub = core.default_hub()
    other = bt.read_parquet(permissive)
    query = other.filter(bt.col("k") < 40)
    cold = CardinalityEstimator(other._sources).estimate(query._plan).rows

    measured = bt.read_parquet(selective).filter(bt.col("k") < 40)
    for _ in range(4):
        measured.collect()

    warm = CardinalityEstimator(other._sources, load_learned_stats(hub)).estimate(query._plan).rows
    assert warm == pytest.approx(cold), "the other file's selectivity must not reach this plan"
    assert warm > 0.05 * _ROWS, f"still poisoned ({warm} rows of {_ROWS})"


def test_an_in_memory_relation_contributes_no_identity():
    """Neither half of an in-memory identity is usable, so it deliberately contributes none.

    Its `identity()` is schema plus row count: two unrelated frames of the same shape collide,
    and the same frame grown by a row stops matching itself. Keying on it would trade one
    defect for the loss of the size-generalization that makes a learned *ratio* worth having.
    """
    from batcher.plan.source_stats import stable_source_key

    frame = bt.from_pydict({"k": [1, 2, 3]})
    assert stable_source_key(frame._sources[0]) == ""


def test_a_file_source_contributes_a_stable_identity(two_files):
    from batcher.plan.source_stats import stable_source_key

    selective, _ = two_files
    key = stable_source_key(bt.read_parquet(selective)._sources[0])
    assert key and "selective" in key


def test_the_key_is_not_on_the_wire(two_files):
    """`to_ir` is the contract with Rust and must not gain a second copy of this."""
    selective, _ = two_files
    scan = bt.read_parquet(selective)._plan
    while not isinstance(scan, Scan):
        scan = scan.input
    assert scan.source_key
    assert "source_key" not in scan.to_ir()


def test_the_key_is_outside_plan_equality():
    """Rewrites compare nodes to decide whether anything changed; that is about shape.

    Making two structurally identical scans unequal because they read different files would
    perturb rule fixpoints to fix a problem that is not about rewriting.
    """
    schema = SchemaRef.from_arrow(pa.schema([pa.field("k", pa.int64())]))
    assert Scan(0, schema, source_key="id:a") == Scan(0, schema, source_key="id:b")


def test_a_source_id_remap_preserves_the_key():
    """Combining two plans shifts `source_id`; the identity must survive that rebuild.

    Rebuilding the node fresh instead of replacing the field would silently return the plan to
    the collided key, which is exactly the failure this whole change removes.
    """
    from batcher.plan.logical.transforms import remap_sources

    schema = SchemaRef.from_arrow(pa.schema([pa.field("k", pa.int64())]))
    shifted = remap_sources(Scan(0, schema, source_key="id:table-a"), 3)
    assert isinstance(shifted, Scan)
    assert shifted.source_id == 3
    assert shifted.source_key == "id:table-a"
