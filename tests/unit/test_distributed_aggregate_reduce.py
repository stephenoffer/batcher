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

    nat = _FakeNative()
    # The `sys.modules` entry is the load-bearing one, and it is the only one needed. Every
    # path to the engine goes through `_internal.native.engine()`, which is
    # `importlib.import_module("batcher._native")` -- a `sys.modules` lookup -- and that
    # accessor's own docstring names this exact mechanism: "distributed reducers install a
    # stub `batcher._native` in `sys.modules`".
    #
    # There used to be a `monkeypatch.setattr(batcher, "_native", mod, raising=False)` here
    # too, explained as necessary because `import batcher._native as nat` binds the PARENT
    # ATTRIBUTE and would shadow a sys.modules-only patch. That was true before the accessor
    # existed. It is not now: `tests/unit/test_native_is_reached_through_the_accessor.py`
    # holds that exactly one module in the tree imports the extension directly, and it is
    # `_internal/errors/hierarchy.py` lifting error types -- not anything on this path. So
    # the attribute was never read, `raising=False` meant it was *created* rather than
    # overridden, and deleting it changes nothing: five tests pass either way, with or
    # without `hierarchy` pre-imported so the attribute genuinely exists.
    mod = types.ModuleType("batcher._native")
    mod.combine = nat.combine
    mod.combine_finalize = nat.combine_finalize
    monkeypatch.setitem(sys.modules, "batcher._native", mod)
    written: list = []
    monkeypatch.setattr(
        "batcher.dist.shuffle_io.write_ipc", lambda batches, path: written.append((path, batches))
    )
    # `_ensure_ray` rebinds `aggregate._reduce_task = ray.remote(aggregate._reduce_task)` on the
    # *module*, permanently and process-wide. These tests call the task as a plain function, so
    # once any earlier test in the session has started a distributed run they were calling a
    # `RemoteFunction` instead and failed with "Remote functions cannot be called directly".
    # That made them pass alone and fail in the full suite — the shape that gets rerun rather
    # than fixed. Unwrap to the underlying function so a unit test of the *fold* does not depend
    # on whether Ray has been wired up; monkeypatch restores the module attribute afterwards.
    task = aggregate._reduce_task
    monkeypatch.setattr(aggregate, "_reduce_task", getattr(task, "_function", task))
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
