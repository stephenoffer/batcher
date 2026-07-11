"""The distributed aggregate reducer folds partials incrementally (bounded memory).

`_reduce_task` merges the mappers' partials with `combine` into one running state instead
of materializing every mapper's partial for the reducer at once, so a high-fan-in / skewed
reducer's peak memory is one running state + one input, not the sum of all inputs. `combine`
is associative+commutative, so the fold is result-identical to a single combine over all
partials. These tests stub the native combine + IPC so the *orchestration* (incremental
fold, single finalize, empty handling) is verified without the Rust engine.
"""

from __future__ import annotations

import sys

import pytest

from batcher.dist.executors import aggregate

pytestmark = pytest.mark.unit


class _Batch:
    def __init__(self, value: int, num_rows: int = 1) -> None:
        self.value = value
        self.num_rows = num_rows


class _FakeNative:
    """A combine that sums partial 'state' values — associative, like the real one."""

    def __init__(self) -> None:
        self.merged_sizes: list[int] = []  # len of each `combine` input list
        self.finalize_count = 0

    def combine(self, _gk, _aj, partials):
        self.merged_sizes.append(len(partials))
        return _Batch(sum(getattr(p, "value", 0) for p in partials))

    def combine_finalize(self, _gk, _aj, partials):
        self.finalize_count += 1
        total = sum(getattr(p, "value", 0) for p in partials)
        return _Batch(total, num_rows=(1 if total else 0))


@pytest.fixture
def wired(monkeypatch):
    import types

    import batcher

    nat = _FakeNative()
    # `import batcher._native as nat` binds via the PARENT ATTRIBUTE `batcher._native`
    # (not sys.modules), so a stub must set that attribute — otherwise the real compiled
    # module (when the engine is built) shadows a sys.modules-only patch. Provide a proper
    # module object exposing the fake's combine/combine_finalize, working whether or not
    # the native engine is present.
    mod = types.ModuleType("batcher._native")
    mod.combine = nat.combine
    mod.combine_finalize = nat.combine_finalize
    monkeypatch.setitem(sys.modules, "batcher._native", mod)
    monkeypatch.setattr(batcher, "_native", mod, raising=False)
    written: list = []
    monkeypatch.setattr(
        "batcher.dist.shuffle_io.write_ipc", lambda batches, path: written.append((path, batches))
    )
    return nat, written


def _wire_reads(monkeypatch, mapping):
    monkeypatch.setattr("batcher.dist.shuffle_io.read_ipc", lambda path: mapping[path])


def test_fold_is_incremental_and_bounded(wired, monkeypatch, tmp_path):
    nat, written = wired
    # 5 mappers, each contributes one partial batch to this reducer.
    paths = [f"m{i}" for i in range(5)]
    _wire_reads(monkeypatch, {p: [_Batch(i + 1)] for i, p in enumerate(paths)})

    path, rows = aggregate._reduce_task("gk", "aj", paths, str(tmp_path), 3)

    # Never combines all 5 at once: each combine sees at most running(1) + one input(1) = 2.
    assert max(nat.merged_sizes) <= 2
    # Exactly one finalize, over the single running state.
    assert nat.finalize_count == 1
    # Result preserved: sum(1..5) == 15 (fold == single combine over all).
    assert written[0][1][0].value == 15
    assert rows == 1
    assert path.endswith("reduce_3.arrow")


def test_multi_batch_input_still_bounded(wired, monkeypatch, tmp_path):
    nat, _ = wired
    # One mapper file carrying 3 partial batches, plus two single-batch mappers.
    _wire_reads(
        monkeypatch,
        {"a": [_Batch(1), _Batch(2), _Batch(3)], "b": [_Batch(4)], "c": [_Batch(5)]},
    )
    aggregate._reduce_task("gk", "aj", ["a", "b", "c"], str(tmp_path), 0)
    # Bounded by running(1) + one file's batches (3) = 4 — not the grand total of 5 batches.
    assert max(nat.merged_sizes) <= 4
    assert nat.finalize_count == 1


def test_empty_inputs_return_none(wired, monkeypatch, tmp_path):
    nat, written = wired
    _wire_reads(monkeypatch, {"x": [], "y": []})
    result = aggregate._reduce_task("gk", "aj", ["x", "y"], str(tmp_path), 1)
    assert result == (None, 0)
    assert nat.finalize_count == 0  # nothing to finalize
    assert written == []


def test_finalize_zero_rows_returns_none(wired, monkeypatch, tmp_path):
    _nat, written = wired
    # A partial that finalizes to zero rows (value 0) → treated as an empty bucket.
    _wire_reads(monkeypatch, {"z": [_Batch(0)]})
    result = aggregate._reduce_task("gk", "aj", ["z"], str(tmp_path), 2)
    assert result == (None, 0)
    assert written == []


def test_some_empty_inputs_skipped(wired, monkeypatch, tmp_path):
    _nat, written = wired
    _wire_reads(monkeypatch, {"a": [_Batch(10)], "empty": [], "b": [_Batch(20)]})
    _path, rows = aggregate._reduce_task("gk", "aj", ["a", "empty", "b"], str(tmp_path), 0)
    assert written[0][1][0].value == 30  # empty input contributes nothing, others preserved
    assert rows == 1
