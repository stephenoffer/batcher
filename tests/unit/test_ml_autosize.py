"""Auto-sized workers / actor pool for the ML path (zero-config inference parallelism)."""

from __future__ import annotations

import os

import pyarrow as pa
import pytest

import batcher as bt
from batcher.ml.gpu import gpu_aware_pool_default, resolve_num_workers

pytestmark = pytest.mark.unit

_resolve_num_workers = resolve_num_workers
_gpu_aware_pool_default = gpu_aware_pool_default


def test_resolve_num_workers_auto_cpu_uses_all_cores():
    assert _resolve_num_workers("auto", 0.0) == max(1, os.cpu_count() or 1)


def test_resolve_num_workers_auto_gpu_is_one_context():
    # A GPU stage keeps one model/CUDA context per worker; scale-out is the actor pool.
    assert _resolve_num_workers("auto", 1.0) == 1
    assert _resolve_num_workers("auto", 0.5) == 1


def test_resolve_num_workers_explicit_wins():
    assert _resolve_num_workers(4, 0.0) == 4
    assert _resolve_num_workers(1, 1.0) == 1
    assert _resolve_num_workers(0, 0.0) == 1  # floored


def test_gpu_aware_pool_cpu_keeps_fallback():
    # CPU stage (num_gpus == 0) keeps the cluster worker count.
    assert _gpu_aware_pool_default(0.0, fallback=7, num_partitions=100) == 7


def test_gpu_aware_pool_sizes_to_cluster_gpus(monkeypatch):
    # GPU stage: replicas = total_GPUs / per-actor num_gpus, clamped to partitions.
    import sys

    class _FakeRay:
        @staticmethod
        def cluster_resources():
            return {"GPU": 8.0, "CPU": 64.0}

    monkeypatch.setitem(sys.modules, "ray", _FakeRay)
    # 8 GPUs, 1 GPU/actor → 8 actors (not the fallback of 1 — the Ray Data foot-gun).
    assert gpu_aware_pool_default(1.0, fallback=1, num_partitions=100) == 8
    # Half-GPU packing → 16 actors.
    assert gpu_aware_pool_default(0.5, fallback=1, num_partitions=100) == 16
    # Clamped to the partition count (no idle actors).
    assert gpu_aware_pool_default(1.0, fallback=1, num_partitions=4) == 4


def test_gpu_aware_pool_sizes_to_pinned_accelerator_class(monkeypatch):
    # Heterogeneous cluster: 4 A100 + 8 T4. A stage pinned to A100 must size to the
    # 4 A100s, not all 12 GPUs (it can't run on the T4s).
    import sys

    class _FakeRay:
        @staticmethod
        def cluster_resources():
            return {
                "GPU": 12.0,
                "accelerator_type:A100": 4.0,
                "accelerator_type:T4": 8.0,
                "CPU": 64.0,
            }

    monkeypatch.setitem(sys.modules, "ray", _FakeRay)
    assert gpu_aware_pool_default(1.0, 1, 100, accelerator_type="A100") == 4
    assert gpu_aware_pool_default(1.0, 1, 100, accelerator_type="T4") == 8
    # Unpinned still sees the whole cluster.
    assert gpu_aware_pool_default(1.0, 1, 100) == 12
    # An absent/sentinel typed resource only sizes down — never above total GPUs.
    assert gpu_aware_pool_default(1.0, 1, 100, accelerator_type="H100") == 12


def test_auto_workers_is_result_invariant():
    # The headline correctness gate: auto workers regroup rows only — identical output.
    def add(b):
        return b.append_column("y", pa.array([v.as_py() * 2 for v in b.column("x")]))

    t = pa.table({"x": list(range(2000))})
    auto = bt.from_arrow(t).ml.map_batches(add).to_pydict()
    one = bt.from_arrow(t).ml.map_batches(add, num_workers=1).to_pydict()
    assert auto == one


def test_autobatch_engages_for_class_fn_and_is_result_invariant():
    # A load-once class `fn` with no batch_size auto-tunes the batch size online; the
    # output must equal a fixed batch_size (rebatching only regroups rows).
    class Scale:
        def __call__(self, b):
            return b.append_column("y", pa.array([v.as_py() * 3 for v in b.column("x")]))

    t = pa.table({"x": list(range(4000))})
    auto = bt.from_arrow(t).ml.map_batches(Scale).to_pydict()
    fixed = bt.from_arrow(t).ml.map_batches(Scale, batch_size=512).to_pydict()
    assert auto == fixed


def test_autobatch_handles_row_multiplying_fn():
    # A row-doubling (flat_map-style) class fn under autobatch == a fixed batch_size.
    class Dup:
        def __call__(self, b):
            return pa.Table.from_batches([b, b]).combine_chunks().to_batches()[0]

    t = pa.table({"x": list(range(3000))})
    auto = sorted(bt.from_arrow(t).ml.map_batches(Dup).to_pydict()["x"])
    fixed = sorted(bt.from_arrow(t).ml.map_batches(Dup, batch_size=777).to_pydict()["x"])
    assert auto == fixed and len(auto) == 6000


def test_plain_function_not_autobatched_but_correct():
    # A plain (non-class, CPU) fn keeps the engine morsel path — no autobatch warm-up —
    # and is still correct.
    t = pa.table({"x": list(range(2000))})
    out = bt.from_arrow(t).ml.map_batches(lambda b: b).to_pydict()
    assert out["x"] == list(range(2000))


def test_vllm_batch_defaults_enable_prefix_and_chunked():
    # Zero-config vLLM batch path turns on prefix caching + chunked prefill (the
    # throughput/TTFT wins Ray Data users must enable by hand); explicit values win.
    from batcher.ml.llm import _vllm_batch_defaults

    assert _vllm_batch_defaults({}) == {
        "enable_prefix_caching": True,
        "enable_chunked_prefill": True,
    }
    overridden = _vllm_batch_defaults({"enable_prefix_caching": False, "max_model_len": 4096})
    assert overridden["enable_prefix_caching"] is False  # user wins
    assert overridden["enable_chunked_prefill"] is True  # default still applied
    assert overridden["max_model_len"] == 4096
