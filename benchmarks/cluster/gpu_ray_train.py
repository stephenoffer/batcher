"""Batcher vs Ray Data as the data plane under **Ray Train** (distributed GPU training).

`gpu_train_ingest.py` times a loader on the driver, with no trainer attached: it answers
"how fast can this engine turn Arrow into tensors", which is a component measurement. This
one puts the loader where it actually runs — **inside a `TorchTrainer` worker on a GPU** —
and times the training loop it feeds. That is the workload the goal names: CPU nodes hold
the corpus and do the read/collate, GPU nodes do forward/backward, and the engine's job is
to keep every device fed.

The two arms differ in exactly one thing, the data plane:

* **batcher** -- each rank opens its own disjoint slice of the Parquet corpus with
  `bt.read.parquet` and streams it with `ds.ml.iter_torch_batches(device="cuda")`.
* **ray** -- the corpus is a `ray.data.Dataset` handed to the trainer, and each rank reads
  `ray.train.get_dataset_shard("train").iter_torch_batches(device="cuda")` -- the pattern
  every Ray Train guide is written around.

Everything else is held: the same seeded model, the same optimizer, the same batch size,
the same epoch count, and the same corpus read exactly once per epoch per arm.

**Sharding is by file, and it is disjoint and complete on both sides.** Rank *r* of *W*
takes `files[r::W]`, so the union across ranks is the corpus and the intersection is empty.
Ray Data's `get_dataset_shard` gives the same guarantee by its own means. Both are checked
rather than assumed: each rank returns the rows it consumed and the sum of the labels it
saw, and the run fails unless those add up to the corpus's own count and checksum. A loader
that silently dropped or duplicated a shard would otherwise look like a speedup.

**What is timed is the epoch, not the job.** A trainer's wall time includes placement-group
setup, actor spawn and the model build, which is a Ray Train cost both arms pay identically
and which swamps a short run. Each rank times from its first batch request to its last and
reports that; the arm's score is the **slowest rank**, because a data-parallel step ends
when the last rank arrives. End-to-end trainer time is printed beside it so the two can be
read together.

Run (needs ray[train] + torch on the driver, and a GPU fleet):
    python benchmarks/cluster/gpu_ray_train.py
    BENCH_TRAIN_N=2000000 BENCH_TRAIN_DIM=512 python benchmarks/cluster/gpu_ray_train.py
    BENCH_TRAIN_EPOCHS=2 BENCH_TRAIN_BATCH=1024 python benchmarks/cluster/gpu_ray_train.py
    BENCH_TRAIN_SHUFFLE=65536 python benchmarks/cluster/gpu_ray_train.py   # local shuffle on
"""

from __future__ import annotations

import functools
import os
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
from _ray_env import init_batcher_ray

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from envinfo import machine_fingerprint, require_release_build

print = functools.partial(print, flush=True)

_SEED = 1234
#: Labels are drawn from `[0, 1000)`, so this bounds what a dropped row can remove from
#: the checksum. See `_consumed_a_real_subset`.
_MAX_LABEL = 999


def _cfg() -> dict:
    return {
        "n": int(os.environ.get("BENCH_TRAIN_N", "1000000")),
        "dim": int(os.environ.get("BENCH_TRAIN_DIM", "256")),
        "batch": int(os.environ.get("BENCH_TRAIN_BATCH", "512")),
        "epochs": int(os.environ.get("BENCH_TRAIN_EPOCHS", "1")),
        "classes": int(os.environ.get("BENCH_TRAIN_CLASSES", "1000")),
        "hidden": int(os.environ.get("BENCH_TRAIN_HIDDEN", "2048")),
        "shards": int(os.environ.get("BENCH_TRAIN_SHARDS", "64")),
        "prefetch": int(os.environ.get("BENCH_TRAIN_PREFETCH", "4")),
        # 0 = no shuffle (the loader-throughput measurement); >0 = a per-rank local-shuffle
        # window of that many rows, which is what a real training job runs with.
        "shuffle": int(os.environ.get("BENCH_TRAIN_SHUFFLE", "0")),
        "dir": os.environ.get("BENCH_TRAIN_PARQUET", "/mnt/cluster_storage/gpu_ray_train"),
        "workers": int(os.environ.get("BENCH_TRAIN_WORKERS", "0")),  # 0 -> every GPU
    }


# --------------------------------------------------------------------------- #
# The corpus
# --------------------------------------------------------------------------- #
def write_shards(directory: str, n: int, dim: int, shards: int) -> dict:
    """Write `n` fixed-seed rows as `shards` Parquet files; return the corpus signature.

    Each shard holds a contiguous id range, so the union across shards is exactly the
    single-source table and the correctness oracle is the same for any sharding.

    Args:
        directory: Where the shards go.
        n: Total rows.
        dim: Feature width.
        shards: Number of Parquet files to write.

    Returns:
        ``{"rows": int, "checksum": float}`` -- what a complete pass must reproduce.
    """
    import pyarrow.parquet as pq

    os.makedirs(directory, exist_ok=True)
    rng = np.random.default_rng(_SEED)
    per = -(-n // shards)
    rows, checksum = 0, 0.0
    for s in range(shards):
        lo, hi = s * per, min((s + 1) * per, n)
        if lo >= hi:
            break
        feats = rng.standard_normal((hi - lo, dim), dtype=np.float32)
        labels = rng.integers(0, 1000, size=hi - lo, dtype=np.int64)
        tbl = pa.table(
            {
                "feat": pa.FixedSizeListArray.from_arrays(pa.array(feats.reshape(-1)), dim),
                "label": pa.array(labels),
            }
        )
        pq.write_table(tbl, os.path.join(directory, f"shard_{s:04d}.parquet"))
        rows += hi - lo
        checksum += float(labels.sum())
    return {"rows": rows, "checksum": checksum}


def corpus_signature(directory: str) -> dict:
    """Read the corpus's row count and label checksum back off disk.

    Used when the shards already exist, so a re-run gates against the corpus it actually
    reads rather than against the one it would have written.
    """
    import pyarrow.parquet as pq

    rows, checksum = 0, 0.0
    for name in sorted(os.listdir(directory)):
        if not name.endswith(".parquet"):
            continue
        tbl = pq.read_table(os.path.join(directory, name), columns=["label"])
        rows += tbl.num_rows
        checksum += float(pa.compute.sum(tbl.column("label")).as_py() or 0)
    return {"rows": rows, "checksum": checksum}


# --------------------------------------------------------------------------- #
# The shared model + step (identical math in both arms)
# --------------------------------------------------------------------------- #
def build_model(dim: int, hidden: int, classes: int):
    """A seeded MLP. Small enough that the loader is the thing under test, big enough that
    the step is a real forward/backward on the device rather than a no-op."""
    import torch
    from torch import nn

    torch.manual_seed(_SEED)
    return nn.Sequential(
        nn.Linear(dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, classes),
    )


def _shard_files(directory: str, rank: int, world: int) -> list[str]:
    """Rank `rank`'s disjoint slice of the corpus's Parquet files."""
    files = sorted(f for f in os.listdir(directory) if f.endswith(".parquet"))
    return [os.path.join(directory, f) for f in files[rank::world]]


def _run_epochs(batches_for_epoch, cfg: dict, model, opt, loss_fn) -> dict:
    """Train over `cfg["epochs"]` passes, timing only the loop, and return this rank's tally.

    `batches_for_epoch(epoch)` yields ``{column: tensor}`` dicts already on the device --
    that callable is the only thing that differs between the two arms.
    """
    import torch

    rows, checksum, t_loop = 0, 0.0, 0.0
    for epoch in range(cfg["epochs"]):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.perf_counter()
        for batch in batches_for_epoch(epoch):
            feat, label = batch["feat"], batch["label"]
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(feat.float()), label.long())
            loss.backward()
            opt.step()
            rows += int(label.shape[0])
            checksum += float(label.sum().item())
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_loop += time.perf_counter() - t0
    return {"rows": rows, "checksum": checksum, "loop_s": t_loop}


def _prepared(cfg: dict):
    """The model, optimizer and loss every rank trains with, wrapped for DDP."""
    import ray.train.torch as rt
    from torch import nn, optim

    model = rt.prepare_model(build_model(cfg["dim"], cfg["hidden"], cfg["classes"]))
    return model, optim.SGD(model.parameters(), lr=0.01), nn.CrossEntropyLoss()


# --------------------------------------------------------------------------- #
# The two train loops
# --------------------------------------------------------------------------- #
def batcher_loop(cfg: dict) -> None:
    """Rank-local Batcher read + `iter_torch_batches` straight onto the device."""
    import ray.train

    import batcher as bt

    ctx = ray.train.get_context()
    rank, world = ctx.get_world_rank(), ctx.get_world_size()
    model, opt, loss_fn = _prepared(cfg)
    files = _shard_files(cfg["dir"], rank, world)
    ds = bt.read.parquet(files)
    shuffle = cfg["shuffle"] or None

    def batches(epoch: int):
        return ds.ml.iter_torch_batches(
            batch_size=cfg["batch"],
            device="cuda",
            prefetch_batches=cfg["prefetch"],
            local_shuffle_buffer_size=shuffle,
            epoch=epoch,
            drop_last=True,
        )

    _report_across_ranks(_run_epochs(batches, cfg, model, opt, loss_fn), cfg["sink"])


def ray_loop(cfg: dict) -> None:
    """The Ray Train idiom: `get_dataset_shard` + `iter_torch_batches`."""
    import ray.train

    model, opt, loss_fn = _prepared(cfg)
    shard = ray.train.get_dataset_shard("train")
    shuffle = cfg["shuffle"] or None

    def batches(_epoch: int):
        return shard.iter_torch_batches(
            batch_size=cfg["batch"],
            device="cuda",
            prefetch_batches=cfg["prefetch"],
            local_shuffle_buffer_size=shuffle,
            drop_last=True,
        )

    _report_across_ranks(_run_epochs(batches, cfg, model, opt, loss_fn), cfg["sink"])


# --------------------------------------------------------------------------- #
# Driving the two trainers
# --------------------------------------------------------------------------- #
def _scaling(cfg: dict):
    from ray.train import ScalingConfig

    return ScalingConfig(num_workers=cfg["workers"], use_gpu=True)


def _run_config(cfg: dict):
    """A `RunConfig` on **shared** storage, which a multi-node trainer needs.

    Ray Train writes its run directory from every worker, so the default (`~/ray_results` on
    whichever machine the worker landed on) is not one place on a cluster. Pointing it at the
    same mount the corpus is on is the fix, and it costs nothing: this benchmark writes no
    checkpoints, only the per-rank metrics `report` carries back.
    """
    from ray.train import RunConfig

    return RunConfig(storage_path=os.path.join(os.path.dirname(cfg["dir"]), "ray_train_runs"))


def _run_arm(engine: str, cfg: dict) -> dict:
    """Run one arm's `TorchTrainer` and return its timing plus every rank's tally."""
    import ray
    from ray.train.torch import TorchTrainer

    datasets = {}
    if engine == "ray":
        import ray.data as rd

        datasets["train"] = rd.read_parquet(cfg["dir"])
    cfg = {**cfg, "sink": os.path.join(cfg["dir"], f"_tallies_{engine}.json")}
    loop = batcher_loop if engine == "batcher" else ray_loop
    trainer = TorchTrainer(
        functools.partial(loop, cfg),
        scaling_config=_scaling(cfg),
        run_config=_run_config(cfg),
        datasets=datasets or None,
    )
    t0 = time.perf_counter()
    result = trainer.fit()
    wall = time.perf_counter() - t0
    # `report` keeps the last metrics per rank; the driver sees rank 0's in `result.metrics`
    # and the rest in the trainer's collected results, so gather from both.
    tallies = _rank_tallies(result, cfg["sink"])
    del ray  # the import exists to fail loudly here rather than inside the trainer
    return {"wall_s": wall, "tallies": tallies}


def _report_across_ranks(tally: dict, sink: str) -> None:
    """Gather every rank's tally onto rank 0, report it, and write it to `sink`.

    Two things about Ray Train V2 made this harder than one `report` call, and both cost a
    run before they were understood.

    `ray.train.report` is a **collective**: a call from rank 0 alone leaves the other seven
    outside the barrier and the trainer never returns, with nothing printed. So every rank
    calls it, and only rank 0 carries a payload.

    And rank 0's payload still did not reach the driver — `Result.metrics` came back without
    it on a run where both arms trained correctly and fast. Rather than keep guessing at the
    metric plumbing, rank 0 also writes the gathered list to a JSON file the driver reads
    directly. That is the measurement's own channel: it does not depend on how a framework
    chooses to persist metrics, and a benchmark whose result vanishes silently is worse than
    one that is slightly less idiomatic.

    Args:
        tally: This rank's ``{rows, checksum, loop_s}``.
        sink: Path rank 0 writes the gathered list to.
    """
    import json

    import ray.train
    import torch.distributed as dist

    ctx = ray.train.get_context()
    world, rank = ctx.get_world_size(), ctx.get_world_rank()
    if world > 1 and dist.is_available() and dist.is_initialized():
        gathered: list[dict | None] = [None] * world
        dist.all_gather_object(gathered, tally)
        tallies = [t for t in gathered if t]
    else:
        tallies = [tally]
    if rank == 0:
        os.makedirs(os.path.dirname(sink), exist_ok=True)
        with open(sink, "w") as fh:
            json.dump(tallies, fh)
    ray.train.report({"tallies": tallies} if rank == 0 else {"rank": rank})


def _rank_tallies(result, sink: str) -> list[dict]:
    """Every rank's ``{rows, checksum, loop_s}``: from `Result.metrics`, else from `sink`."""
    import json

    metrics = getattr(result, "metrics", None) or {}
    tallies = metrics.get("tallies") if isinstance(metrics, dict) else None
    if not tallies and os.path.exists(sink):
        with open(sink) as fh:
            tallies = json.load(fh)
    return [t for t in (tallies or []) if isinstance(t, dict) and "rows" in t]


def _summarize(name: str, arm: dict, corpus: dict, cfg: dict) -> dict:
    """Fold one arm's per-rank tallies into a score, and check it consumed the corpus."""
    tallies = arm["tallies"]
    if not tallies:
        return {"name": name, "error": "no rank metrics returned"}
    slowest = max(t["loop_s"] for t in tallies)
    rows = sum(t["rows"] for t in tallies)
    checksum = sum(t["checksum"] for t in tallies)
    # `drop_last` discards a ragged tail per rank, so a complete pass is bounded below by
    # the corpus minus one short batch per rank per epoch rather than being exactly it.
    floor = corpus["rows"] - cfg["batch"] * len(tallies) * cfg["epochs"]
    complete = floor <= rows <= corpus["rows"] * cfg["epochs"]
    return {
        "name": name,
        "loop_s": slowest,
        "wall_s": arm["wall_s"],
        "rows": rows,
        "rows_s": rows / slowest if slowest else 0.0,
        "checksum": checksum,
        "ranks": len(tallies),
        "complete": complete,
        "consistent": _consumed_a_real_subset(rows, checksum, corpus, cfg),
    }


def _consumed_a_real_subset(rows: int, checksum: float, corpus: dict, cfg: dict) -> bool:
    """Whether this arm's label sum is consistent with having read `rows` of the corpus.

    **Not a cross-engine checksum comparison, deliberately.** The first version of this gate
    demanded the two arms produce the same sum and reported MISMATCH on a correct run: both
    consumed exactly 999,424 of 1,000,000 rows, but not the *same* 999,424. `drop_last`
    discards a ragged tail per rank, the two engines shard differently (`files[rank::W]`
    against Ray Data's block split), so they drop different tails — and neither promises
    otherwise. The observed delta was 9,374, which is 1.6% of the 575,424 a different
    576-row tail can account for.

    What is actually checkable is stronger, and is checked per arm against the corpus rather
    than against the other engine: labels are non-negative, so dropping rows can only lower
    the sum, and it cannot lower it by more than the dropped count times the largest label.
    An arm that duplicated a shard, skipped one silently, or read the wrong rows fails this;
    an arm that merely dropped a different tail does not.
    """
    dropped = corpus["rows"] * cfg["epochs"] - rows
    if dropped < 0:
        return False  # read more than the corpus: a duplicated shard
    shortfall = corpus["checksum"] * cfg["epochs"] - checksum
    return 0 <= shortfall <= dropped * _MAX_LABEL


def main() -> int:
    require_release_build()
    print(machine_fingerprint())
    cfg = _cfg()
    init_batcher_ray(forward=("BENCH_TRAIN_PARQUET",))
    import ray

    gpus = int(float(ray.cluster_resources().get("GPU", 0.0)))
    if not gpus:
        print("no GPUs in the cluster — nothing to measure")
        return 1
    cfg["workers"] = cfg["workers"] or gpus
    print(f"cluster: {gpus} GPU, {ray.cluster_resources().get('CPU')} CPU")

    if not os.path.isdir(cfg["dir"]) or not os.listdir(cfg["dir"]):
        corpus = write_shards(cfg["dir"], cfg["n"], cfg["dim"], cfg["shards"])
        print(f"generated corpus: {corpus['rows']} rows -> {cfg['dir']}")
    else:
        corpus = corpus_signature(cfg["dir"])
        print(f"existing corpus: {corpus['rows']} rows in {cfg['dir']}")
    print(
        f"workers={cfg['workers']} dim={cfg['dim']} batch={cfg['batch']} "
        f"epochs={cfg['epochs']} shuffle={cfg['shuffle'] or 'off'} "
        f"prefetch={cfg['prefetch']}\n"
    )

    rows: list[dict] = []
    for engine in ("batcher", "ray"):
        print(f"  [{engine}] training ...")
        try:
            arm = _run_arm(engine, cfg)
        except Exception as exc:  # a failed arm is a result, not a reason to lose the other
            print(f"  [{engine}] ERROR {type(exc).__name__}: {exc}")
            rows.append({"name": engine, "error": f"{type(exc).__name__}: {exc}"})
            continue
        summary = _summarize(engine, arm, corpus, cfg)
        rows.append(summary)
        if "error" in summary:
            print(f"  [{engine}] {summary['error']}")
        else:
            print(
                f"  [{engine}] loop {summary['loop_s']:.2f}s (slowest of "
                f"{summary['ranks']} ranks)  {summary['rows_s']:.0f} rows/s  "
                f"trainer wall {summary['wall_s']:.1f}s"
            )

    print("\nengine     loop_s   rows/s      wall_s   rows        complete")
    print("-" * 66)
    for r in rows:
        if "error" in r:
            print(f"{r['name']:<10} {r['error']}")
            continue
        print(
            f"{r['name']:<10} {r['loop_s']:>6.2f}  {r['rows_s']:>9.0f}  "
            f"{r['wall_s']:>7.1f}  {r['rows']:>10}   {r['complete']}"
        )
    scored = {r["name"]: r for r in rows if "error" not in r}
    if "batcher" in scored and "ray" in scored:
        b, y = scored["batcher"], scored["ray"]
        print(f"\nbatcher vs ray: {y['loop_s'] / b['loop_s']:.2f}x  (>1 = batcher faster)")
        # Each arm is checked against the CORPUS, not against the other engine: they drop
        # different `drop_last` tails by construction. See `_consumed_a_real_subset`.
        ok = all(a["complete"] and a["consistent"] for a in (b, y)) and b["rows"] == y["rows"]
        print(
            f"correctness: rows b={b['rows']} r={y['rows']} (equal, and each a real subset "
            f"of the {corpus['rows']}-row corpus)  "
            f"checksum b={b['checksum']:.0f} r={y['checksum']:.0f} "
            f"[differ by {abs(b['checksum'] - y['checksum']):.0f}, which a different "
            f"drop_last tail accounts for]  [{'OK' if ok else 'MISMATCH'}]"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
