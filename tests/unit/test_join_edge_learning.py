"""A join selectivity measured on one tree prices the same edge in a tree nobody ran.

The loop in `kyber.stats.selectivity.join_edges`: the optimizer registers each executed join's
edge and input signatures, Core measures rows under those signatures, and the estimator prices
any later inner join over the same edge from the measured selectivity -- including a join with
different inputs, on the other side, in another order.
"""

from __future__ import annotations

import math

import pyarrow as pa

from batcher.kyber.learning import load_learned_stats
from batcher.kyber.signature import plan_signature
from batcher.kyber.stats.estimator import StatsEstimator
from batcher.kyber.stats.selectivity.join_edges import (
    edge_key,
    measured_edge_selectivities,
    register_join_edges,
)
from batcher.metadata import MetadataHub
from batcher.metadata.backends import InProcessBackend
from batcher.plan.expr_ir import Col, Lit
from batcher.plan.feedback import OperatorFeedback
from batcher.plan.ids import OpId
from batcher.plan.logical import Filter, Join, JoinOutputCol, Project, Projection, Scan
from batcher.plan.schema import SchemaRef
from batcher.plan.source_stats import SourceStatistics
from batcher.plan.stats import ColumnStat, Provenance


class _Named:
    """The estimator hooks edge identification reads: signatures, relation names, keys."""

    signature_of = staticmethod(plan_signature)

    @staticmethod
    def scan_identity(scan: Scan) -> str:
        return scan.source_key

    @staticmethod
    def key_is_unique(scan: Scan, column: str) -> bool:
        return column == "id"  # every fixture table's `id`, and nothing else, is a key


_NAMES = _Named()


def _scan(sid: int, key: str, names: list[str]) -> Scan:
    schema = SchemaRef(pa.schema([pa.field(n, pa.int64()) for n in names]))
    return Scan(sid, schema, source_key=key)


def _join(left, right, lk: str, rk: str, out: list[tuple[str, str]]) -> Join:
    cols = tuple(JoinOutputCol(side, name, name) for side, name in out)
    return Join(left, right, (lk,), (rk,), "inner", cols)


def _cast() -> Filter:
    return Filter(_scan(0, "cast_info", ["movie_id", "role_id"]), Col("role_id") > Lit(5))


def _chars() -> Scan:
    return _scan(1, "char_name", ["id", "n"])


def _measured_tree() -> Join:
    """`cast ⋈ chars` on `role_id = id`, the tree that ran and was measured."""
    return _join(_cast(), _chars(), "role_id", "id", [("left", "movie_id"), ("right", "n")])


def _record(hub: MetadataHub, node, rows: int, runs: int = 2) -> None:
    for _ in range(runs):
        hub.record(
            OperatorFeedback(
                op_id=OpId(0),
                kind="hash_join" if isinstance(node, Join) else "filter",
                n_actual=rows,
                t_op_ms=1.0,
                m_peak_bytes=0,
                selectivity=1.0,
                batch_size=1024,
                signature=plan_signature(node),
                n_estimated=float(rows),
            )
        )


def _stats() -> list[SourceStatistics]:
    def col(ndv: int) -> ColumnStat:
        return ColumnStat(min=0, max=ndv, ndv=ndv, provenance=Provenance.EXACT)

    return [
        SourceStatistics(row_count=100_000, columns={"role_id": col(1_000), "movie_id": col(500)}),
        SourceStatistics(row_count=1_000, columns={"id": col(1_000)}),
    ]


def test_the_edge_key_is_the_same_on_either_side_and_through_renames():
    measured = _measured_tree()
    # The other orientation, with `chars` renamed through a projection on the way in.
    renamed = Project(_chars(), (Projection("char_id", Col("id")), Projection("n", Col("n"))))
    flipped = _join(renamed, _cast(), "char_id", "role_id", [("left", "n"), ("right", "movie_id")])
    assert edge_key(measured, _NAMES) is not None
    assert edge_key(measured, _NAMES).key == edge_key(flipped, _NAMES).key


def test_an_edge_through_a_computed_column_has_no_key():
    computed = Project(_chars(), (Projection("id", Col("id") + Lit(1)),))
    join = _join(_cast(), computed, "role_id", "id", [("left", "movie_id")])
    assert edge_key(join, _NAMES) is None


def test_two_foreign_keys_meeting_on_a_shared_value_are_not_learned():
    # `movie_info.movie_id = movie_keyword.movie_id`: neither side is a key, so the fraction
    # depends on which movies each input holds and does not transfer (the q29a case).
    mi = _scan(0, "movie_info", ["movie_id", "info"])
    mk = _scan(1, "movie_keyword", ["movie_id", "keyword_id"])
    join = _join(mi, mk, "movie_id", "movie_id", [("left", "info"), ("right", "keyword_id")])
    assert edge_key(join, _NAMES) is None


def test_a_measured_edge_prices_a_join_over_it_that_never_ran():
    hub = MetadataHub(InProcessBackend())
    measured = _measured_tree()
    register_join_edges(hub, measured, _NAMES)
    _record(hub, measured.left, 50_000)  # the filtered cast_info side
    _record(hub, measured.right, 1_000)  # char_name
    _record(hub, measured, 10)  # the join kept 10 of 50M pairs
    edges = measured_edge_selectivities(hub)
    assert len(edges) == 1
    assert math.isclose(next(iter(edges.values())), 10 / (50_000 * 1_000), rel_tol=1e-9)

    # A different tree over the same edge: no filter on cast_info, char_name on the left.
    other = _join(
        _chars(),
        _scan(0, "cast_info", ["movie_id", "role_id"]),
        "id",
        "role_id",
        [("left", "n"), ("right", "movie_id")],
    )
    learned = StatsEstimator([None, None], load_learned_stats(hub), source_stats=_stats())
    structural = StatsEstimator([None, None], {}, source_stats=_stats())
    assert structural.estimate(other).rows == 100_000  # containment: each cast row, one char
    # 1,000 x 100,000 x (10 / 50M) = 20 rows: the measured edge, applied to these inputs.
    assert math.isclose(learned.estimate(other).rows, 20.0, rel_tol=1e-9)
    assert learned.estimate(other).provenance is Provenance.LEARNED


def test_an_unmeasured_input_leaves_the_estimate_structural():
    hub = MetadataHub(InProcessBackend())
    measured = _measured_tree()
    register_join_edges(hub, measured, _NAMES)
    _record(hub, measured, 10)
    _record(hub, measured.right, 1_000)  # the left input was never measured
    assert measured_edge_selectivities(hub) == {}


def test_an_empty_join_is_learned_as_selective_not_impossible():
    hub = MetadataHub(InProcessBackend())
    measured = _measured_tree()
    register_join_edges(hub, measured, _NAMES)
    _record(hub, measured.left, 50_000)
    _record(hub, measured.right, 1_000)
    _record(hub, measured, 0)
    (sel,) = measured_edge_selectivities(hub).values()
    assert 0.0 < sel < 1 / (50_000 * 1_000)


def test_a_learned_edge_never_prices_a_join_above_the_foreign_key_side():
    hub = MetadataHub(InProcessBackend())
    measured = _measured_tree()
    register_join_edges(hub, measured, _NAMES)
    # Measured on a sliver: one char kept nearly every one of its cast rows.
    _record(hub, measured.left, 100)
    _record(hub, measured.right, 1)
    _record(hub, measured, 90)
    other = _join(
        _chars(),
        _scan(0, "cast_info", ["movie_id", "role_id"]),
        "id",
        "role_id",
        [("left", "n"), ("right", "movie_id")],
    )
    learned = StatsEstimator([None, None], load_learned_stats(hub), source_stats=_stats())
    # 0.9 x 1,000 x 100,000 = 90M by the fraction alone; each cast row meets at most one char.
    assert learned.estimate(other).rows <= 100_000
