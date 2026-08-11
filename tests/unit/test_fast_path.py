"""The opt-in small-query fast path: it must be off, narrow, and result-invariant.

`api/orchestration/fast_path.py` skips Carbonite admission, the memory reservation, profile
assembly, the event bus and the whole write side of the learned-stats loop. Every one of
those is load-bearing somewhere, so the tests that matter are the ones pinning *where it
declines* — a gate that quietly widened would take those subsystems out of a query that
needed them, and the result would still be right, so nothing else would notice.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher.api.orchestration.fast_path import (
    MAX_FAST_PATH_NODES,
    MAX_FAST_PATH_ROWS,
    eligible,
)
from batcher.config import Config, ExecutionConfig, active_config, config_context


@pytest.fixture
def frame():
    """A small in-memory frame with nulls, duplicate keys and a string column."""
    return bt.from_pydict(
        {
            "id": list(range(200)),
            "grp": [i % 7 for i in range(200)],
            "s": [f"k{i % 11}" for i in range(200)],
            "v": [None if i % 13 == 0 else float(i) for i in range(200)],
        }
    )


def _gate(plan, sources, **overrides):
    """Call `eligible` with the all-clear defaults, overriding one condition at a time."""
    kwargs = {
        "distributed": False,
        "adaptive": False,
        "spill": False,
        "backend": "cpu",
        "cache": False,
    }
    kwargs.update(overrides)
    return eligible(plan, sources, **kwargs)


class TestItIsOffByDefault:
    """The flag guards a documented trade (no cross-query learning). A silent default
    flip would opt every deployment out of the moat with nothing turning red."""

    def test_the_config_default_is_false(self):
        assert active_config().execution.fast_path is False

    def test_an_eligible_plan_is_refused_while_the_flag_is_off(self, frame):
        ds = frame.filter(bt.col("id") > 100)
        assert _gate(ds._plan, ds._sources) is False

    def test_the_same_plan_is_accepted_once_the_flag_is_on(self, frame):
        ds = frame.filter(bt.col("id") > 100)
        with config_context(Config(execution=ExecutionConfig(fast_path=True))):
            assert _gate(ds._plan, ds._sources) is True


class TestTheGateDeclinesWhatItMustNotSkip:
    """Each condition routes to a different executor, or to a subsystem the fast path
    removes. A `False` is always safe; a wrong `True` is silent."""

    @pytest.fixture(autouse=True)
    def _enabled(self):
        with config_context(Config(execution=ExecutionConfig(fast_path=True))):
            yield

    @pytest.mark.parametrize(
        "override",
        [
            {"distributed": True},
            {"adaptive": True},
            {"spill": True},
            {"cache": True},
            {"backend": "gpu"},
            {"backend": "auto"},
        ],
        ids=["distributed", "adaptive", "spill", "cache", "gpu", "auto"],
    )
    def test_another_route_wins(self, frame, override):
        ds = frame.filter(bt.col("id") > 100)
        assert _gate(ds._plan, ds._sources, **override) is False

    def test_a_udf_is_refused(self, frame):
        """`map_batches` runs Python per batch through a different executor entirely, and
        its cost dwarfs the orchestration this path exists to remove."""
        ds = frame.map_batches(lambda b: b)
        assert _gate(ds._plan, ds._sources) is False

    def test_a_deep_plan_is_refused(self, frame):
        """Past the node cap the admission and adaptive machinery are what get a plan's
        breakers and join order right, and skipping them there is the trade this must
        not make silently."""
        ds = frame
        for i in range(MAX_FAST_PATH_NODES + 2):
            ds = ds.filter(bt.col("id") > i)
        assert _gate(ds._plan, ds._sources) is False

    def test_a_plan_with_no_sources_is_refused(self, frame):
        assert _gate(frame._plan, []) is False


class TestTheResultIsIdentical:
    """The whole safety argument: same optimized plan, same engine call, same rows. If any
    of these diverge the path is not an optimization, it is a second semantics."""

    @staticmethod
    def _both(build):
        """Run `build()` with the fast path off, then on, and return both results."""
        off = build()
        with config_context(Config(execution=ExecutionConfig(fast_path=True))):
            on = build()
        return off, on

    def test_a_filter_and_projection_agree(self, frame):
        off, on = self._both(lambda: frame.filter(bt.col("id") > 100).select("id", "v").to_pydict())
        assert off == on

    def test_an_aggregate_over_nulls_agrees(self, frame):
        """Nulls are where a skipped code path most often shows up as a different answer."""
        off, on = self._both(
            lambda: (
                frame.group_by("grp")
                .agg(s=bt.col("v").sum(), c=bt.col("v").count(), m=bt.col("v").mean())
                .to_pydict()
            )
        )
        assert off == on

    def test_a_join_agrees(self, frame):
        other = bt.from_pydict({"id": list(range(0, 200, 3)), "w": [1.5] * 67})
        off, on = self._both(lambda: frame.join(other, on="id").to_pydict())
        assert off == on

    def test_a_descending_sort_agrees_in_order(self, frame):
        """Compared as an ordered list on purpose: `assert_same` is order-independent by
        design, so an order-independent comparison here could not see a sort bug."""
        off, on = self._both(lambda: frame.sort("id", descending=True).limit(25).to_pydict()["id"])
        assert off == on
        assert off == sorted(off, reverse=True)

    def test_an_empty_result_keeps_its_schema(self, frame):
        """The empty case takes a different branch for its schema, and a fast path that
        lost the column names would still 'return no rows'."""
        off, on = self._both(lambda: frame.filter(bt.col("id") < 0).collect())
        assert off.schema == on.schema
        assert on.num_rows == 0

    def test_a_distinct_agrees(self, frame):
        off, on = self._both(lambda: sorted(frame.select("s").distinct().to_pydict()["s"]))
        assert off == on


class TestTheRowCapIsRealAndChecked:
    """The row cap is what stands in for the skipped admission: it keeps the input small
    enough that the memory envelope could not have bound."""

    def test_the_cap_is_a_positive_row_count(self):
        assert MAX_FAST_PATH_ROWS > 0
        assert MAX_FAST_PATH_NODES > 0

    def test_a_source_over_the_cap_is_refused(self, monkeypatch, frame):
        """Faked rather than materialized — allocating past the cap to prove the cap
        works would make this a memory test, not a gate test."""
        import batcher.api.orchestration.fast_path as fp

        monkeypatch.setattr(fp, "MAX_FAST_PATH_ROWS", 10)
        ds = frame.filter(bt.col("id") > 100)
        with config_context(Config(execution=ExecutionConfig(fast_path=True))):
            assert _gate(ds._plan, ds._sources) is False
