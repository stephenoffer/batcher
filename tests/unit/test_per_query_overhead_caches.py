"""The memos that cut the fixed per-query control-plane cost return what recomputing would.

Each cache here exists because a trivial query re-derived the same answer several times per
terminal op (or once per op from inputs that had not changed). Every one of them is only
worth having if it is *exact*, so each test pins two things: a repeated call returns the
same answer as a fresh computation, and the thing that should invalidate it does.

The cgroup reader is the one that guards memory safety, so its tests pin the stronger
property: it is not a cache of the *value* at all. A held descriptor re-reads the kernel's
current figure on every call.
"""

from __future__ import annotations

import os

import pyarrow as pa
import pytest

import batcher as bt
from batcher._internal.hardware import sysfs
from batcher.carbonite.memory import probe
from batcher.carbonite.policies import cpu_budget
from batcher.kyber import learning
from batcher.plan.visitor import walk_with_base_names

pytestmark = pytest.mark.unit


# --- sysfs.read_live_text: held descriptors, live values -------------------------------


@pytest.mark.skipif(not os.path.exists("/proc/self/statm"), reason="needs Linux /proc")
def test_a_held_descriptor_reads_the_live_value_not_a_cached_one():
    """The RSS figure the pressure ladder reads must move when the process allocates.

    A reader that cached the *contents* would keep returning the first reading; the held
    descriptor is re-read from offset 0, which makes the kernel regenerate it.
    """
    sysfs._forget_live_fds()
    before = probe.process_rss_bytes()
    assert "/proc/self/statm" in sysfs._LIVE_FDS  # the descriptor is held ...
    ballast = bytearray(64 << 20)
    for i in range(0, len(ballast), 4096):  # touch every page so it is resident
        ballast[i] = 1
    after = probe.process_rss_bytes()
    assert before is not None and after is not None
    assert after - before >= 32 << 20  # ... and still reports the allocation
    del ballast


def test_a_path_outside_the_kernel_trees_is_opened_on_every_read(tmp_path):
    """A regular file replaced by rename must not keep serving its old inode."""
    path = tmp_path / "memory.current"
    path.write_text("4096\n")
    assert sysfs.read_live_text(str(path)) == "4096\n"
    replacement = tmp_path / "next"
    replacement.write_text("8192\n")
    os.replace(replacement, path)
    assert sysfs.read_live_text(str(path)) == "8192\n"
    assert str(path) not in sysfs._LIVE_FDS


def test_a_held_descriptor_that_fails_is_dropped_and_the_file_reopened(tmp_path, monkeypatch):
    """A cgroup removed under a held descriptor reads like a fresh `open`, not an error."""
    sysfs._forget_live_fds()
    monkeypatch.setattr(sysfs, "_LIVE_PREFIXES", (str(tmp_path) + os.sep,))
    path = tmp_path / "memory.current"
    path.write_text("1\n")
    assert sysfs.read_live_text(str(path)) == "1\n"
    held = sysfs._LIVE_FDS[str(path)]
    path.write_text("2\n")  # rewritten in place: the held descriptor sees it
    assert sysfs.read_live_text(str(path)) == "2\n"
    os.close(held)  # the descriptor goes bad, as a removed cgroup's does
    assert sysfs.read_live_text(str(path)) == "2\n"  # re-opened, not an error
    path.unlink()
    sysfs._forget_live_fds()
    assert sysfs.read_live_text(str(path)) is None


@pytest.mark.skipif(not os.path.exists("/proc/self/statm"), reason="needs Linux /proc")
def test_the_post_fork_hook_closes_and_forgets_every_held_descriptor():
    """`register_at_fork(after_in_child=...)` runs this in a child, which re-opens its own."""
    sysfs.read_live_text("/proc/self/statm")
    held = list(sysfs._LIVE_FDS.values())
    assert held
    sysfs._forget_live_fds()
    assert sysfs._LIVE_FDS == {}
    for fd in held:
        with pytest.raises(OSError):
            os.fstat(fd)


# --- cpu_budget: the contention check comes first --------------------------------------


def test_a_quiet_machine_skips_the_core_count_probe(monkeypatch):
    """Inside the deadband the answer is `None` whatever the permitted count is."""
    calls = []
    monkeypatch.setattr(cpu_budget, "available_cpu_count", lambda: calls.append(1) or 8)
    monkeypatch.setattr(cpu_budget, "cpu_oversubscription", lambda: 1.0)
    assert cpu_budget.reduced_core_budget() is None
    assert calls == []


@pytest.mark.parametrize("pressure", [1.0, 1.25, 1.3, 2.0, 8.0])
def test_the_reduced_budget_is_what_measuring_everything_gives(monkeypatch, pressure):
    monkeypatch.setattr(cpu_budget, "available_cpu_count", lambda: 16)
    monkeypatch.setattr(cpu_budget, "cpu_oversubscription", lambda: pressure)
    permitted, budget, _ = cpu_budget._measure()
    expected = budget if budget < permitted else None
    assert cpu_budget.reduced_core_budget() == expected


# --- learning.load_column_tables: the column slice of the bundle -----------------------


def test_the_column_tables_are_the_bundles_column_tables():
    from batcher.metadata import MetadataHub
    from batcher.metadata.backends import InProcessBackend

    hub = MetadataHub(InProcessBackend())
    learning.record_column_stats(hub, {"a": 10.0}, {}, {"a": 8.0}, source_key="src")
    learning.record_execution(hub, bt.from_pydict({"a": [1, 2]})._plan, 2)
    bundle = learning.load_learned_stats(hub)
    tables = learning.load_column_tables(hub)
    column_keys = [k for k in bundle if k.startswith("__column_")]
    assert column_keys
    for key in column_keys:
        assert tables[key] is bundle[key]
    assert learning.load_column_tables(None) == {}


# --- plan memos ------------------------------------------------------------------------


def _join_plan():
    a = bt.from_pydict({"k": [1, 2, 3], "v": [1.0, 2.0, 3.0]})
    b = bt.from_pydict({"t": [1, 2], "x": [5, 6]})
    return a.join(b, left_on="k", right_on="t").select("k", "x").filter(bt.col("x") > 1)._plan


def test_the_base_name_walk_is_memoized_on_the_node_and_matches_a_fresh_walk():
    plan = _join_plan()
    first = walk_with_base_names(plan)
    assert walk_with_base_names(plan) is first
    fresh = _join_plan()  # an equal plan, never walked
    assert [(type(n).__name__, m) for n, m in walk_with_base_names(fresh)] == [
        (type(n).__name__, m) for n, m in first
    ]


def test_a_trivial_query_still_returns_its_rows_with_every_memo_warm():
    """End to end: repeated trivial queries agree with the first, memos and all."""
    ta = pa.table({"k": list(range(100)), "g": [i % 7 for i in range(100)]})
    ds = bt.from_arrow(ta)
    first = ds.filter(bt.col("k") > 5).group_by("g").agg(n=bt.count()).sort("g").to_pydict()
    for _ in range(3):
        again = ds.filter(bt.col("k") > 5).group_by("g").agg(n=bt.count()).sort("g").to_pydict()
        assert again == first
    assert first["n"] == [sum(1 for k in range(6, 100) if k % 7 == g) for g in range(7)]


# --- config adaptation: the same adaptation is the same object, and stays resolved ------


def test_the_same_adaptation_of_the_same_config_is_the_same_object():
    from batcher.carbonite import manager
    from batcher.config import Config

    base = Config()
    first = manager._adapted(base, (("morsel_rows", 4096),))
    assert first.execution.morsel_rows == 4096
    assert manager._adapted(base, (("morsel_rows", 4096),)) is first
    other = manager._adapted(base, (("morsel_rows", 2048),))
    assert other is not first and other.execution.morsel_rows == 2048
    rebuilt = manager._adapted(Config(), (("morsel_rows", 2048),))  # a new base object
    assert rebuilt is not other and rebuilt == other
    assert base.execution.morsel_rows != 4096  # the base is never touched


def test_two_alternating_configs_both_stay_resolved():
    """The auto-config and the adapted config alternate every query; neither evicts the other."""
    from batcher.config import Config
    from batcher.config.config import _resolved, reset_resolution_memo

    reset_resolution_memo()
    auto, adapted = Config(), Config()
    resolved_auto, resolved_adapted = _resolved(auto), _resolved(adapted)
    for _ in range(3):
        assert _resolved(auto) is resolved_auto
        assert _resolved(adapted) is resolved_adapted
    reset_resolution_memo()
    assert _resolved(auto) is not resolved_auto  # reset still forces a fresh resolution


# --- plan building: memos a fresh query plan inherits from the scans it is built on ------


def test_a_moved_scan_inherits_only_the_schema_memos():
    """A join renumbers its right side's scans; the copy keeps what depends on schema alone."""
    from batcher.plan.logical import Scan
    from batcher.plan.logical.transforms import remap_sources

    right = bt.from_pydict({"t": [1, 2], "x": [3.0, 4.0]})._plan
    assert isinstance(right, Scan)
    moved = remap_sources(right, 5)
    assert moved.source_id == right.source_id + 5
    assert moved.available_schema() is right.available_schema()  # carried, not re-derived
    # The IR carries the source id, so it is never carried.
    assert moved.to_ir() != right.to_ir()
    assert moved.to_ir()["source_id"] == right.source_id + 5
    # And the moved scan's answers are the ones a fresh derivation gives.
    fresh = Scan(moved.source_id, moved.schema, moved.source_key)
    assert moved.available_schema().arrow.equals(fresh.available_schema().arrow)
    assert moved.content_key() == fresh.content_key()


def test_the_identity_suffix_is_memoized_and_still_distinguishes_schemas():
    a = bt.from_pydict({"k": [1, 2]})._plan
    b = bt.from_pydict({"k": [1.5, 2.5]})._plan
    assert a.identity_suffix() is a.identity_suffix()
    assert a.identity_suffix() != b.identity_suffix()
    assert a.content_key() != b.content_key()
    # Carried to a renumbered copy, like the other schema-only memos.
    from batcher.plan.logical.transforms import remap_sources

    assert remap_sources(a, 3).identity_suffix() is a.identity_suffix()


def test_filter_still_refuses_udf_options_without_a_callable():
    from batcher._internal.errors import PlanError
    from batcher.api.dataset._udf import build

    ds = bt.from_pydict({"x": [1, 2, 3]})
    assert ds.filter(bt.col("x") > 1).to_pydict() == {"x": [2, 3]}
    with pytest.raises(PlanError, match="num_gpus"):
        ds.filter(bt.col("x") > 1, num_gpus=1)

    def method(a, b=1, c="x"):
        return a

    assert build._defaults(method) == {"a": build._defaults(method)["a"], "b": 1, "c": "x"}
    build.refuse_callable_options(method, {"b": 1, "c": "x"})
    with pytest.raises(PlanError, match=r"\['c'\]"):
        build.refuse_callable_options(method, {"b": 1, "c": "y"})
