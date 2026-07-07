"""Batcher vs Ray Data — distributed audio feature extraction (multimodal audio workload).

Waveforms → a mel-spectrogram (torchaudio, CPU) → a CNN classifier (GPU) — the audio branch
of the multimodal-preprocessing workload. A two-stage CPU→GPU chain (the CPU mel transform
feeds the GPU model), so it exercises the same stage-overlap + warm-pool machinery as the
image path, on a different modality. Deterministic seeded model → prediction agreement.

Run:
    python benchmarks/cluster/gpu_audio.py            # 16384 clips x 1s @ 16kHz
"""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import os
import time

import numpy as np
import pyarrow as pa

print = functools.partial(print, flush=True)

_SEED = 1234
_SR = 16000
_SAMPLES = _SR  # 1 second


def _cfg() -> dict:
    return {
        "n": int(os.environ.get("BENCH_AUDIO_N", "16384")),
        "shards": int(os.environ.get("BENCH_AUDIO_SHARDS", "32")),
        "batch": int(os.environ.get("BENCH_AUDIO_BATCH", "128")),
        "num_gpus": float(os.environ.get("BENCH_GPU_NUM_GPUS", "1")),
        "dir": os.environ.get("BENCH_AUDIO_PARQUET", "/mnt/cluster_storage/gpu_audio"),
        "runs": int(os.environ.get("BENCH_RUNS", "2")),
        "timeout": float(os.environ.get("BENCH_ENGINE_TIMEOUT", "400")),
    }


def write_shards(directory: str, n: int, shards: int) -> str:
    import pyarrow.parquet as pq

    os.makedirs(directory, exist_ok=True)
    rng = np.random.default_rng(_SEED)
    per = -(-n // shards)
    written = 0
    for s in range(shards):
        lo, hi = s * per, min((s + 1) * per, n)
        if lo >= hi:
            break
        wav = rng.standard_normal((hi - lo, _SAMPLES), dtype=np.float32)
        tbl = pa.table(
            {
                "id": pa.array(np.arange(lo, hi, dtype=np.int64)),
                "wav": pa.FixedSizeListArray.from_arrays(pa.array(wav.reshape(-1)), _SAMPLES),
            }
        )
        pq.write_table(tbl, os.path.join(directory, f"shard_{s:04d}.parquet"))
        written = hi
    return f"{written} audio clips (1s@16kHz) -> {directory}"


def mel_stage(batch) -> dict:
    """CPU stage: waveform → 64-bin mel spectrogram, resized to a (3,64,64) uint8 image."""
    import torch
    import torchaudio

    col = batch.column("wav")
    col = col.combine_chunks() if isinstance(col, pa.ChunkedArray) else col
    wav = torch.from_numpy(col.flatten().to_numpy(zero_copy_only=False).astype(np.float32)).reshape(
        len(batch), _SAMPLES
    )
    melspec = torchaudio.transforms.MelSpectrogram(
        sample_rate=_SR, n_mels=64, n_fft=1024, hop_length=256
    )
    mel = melspec(wav)  # (B, 64, T)
    mel = torch.nn.functional.interpolate(
        mel.unsqueeze(1), size=(64, 64), mode="bilinear", align_corners=False
    )
    mel = mel.clamp(0, 50) / 50.0  # normalize to 0..1
    img = (mel.repeat(1, 3, 1, 1) * 255).to(torch.uint8).numpy()  # (B,3,64,64)
    ids = batch.column("id")
    ids = ids.combine_chunks() if isinstance(ids, pa.ChunkedArray) else ids
    return {"id": ids.to_numpy(zero_copy_only=False), "mel": img}


class AudioModel:
    """GPU stage: mel image → ResNet-18 → class label (model loaded once)."""

    def __init__(self) -> None:
        import torch
        import torchvision

        torch.manual_seed(_SEED)
        self._dev = "cuda" if torch.cuda.is_available() else "cpu"
        if self._dev == "cuda":
            torch.set_float32_matmul_precision("high")
        self._m = torchvision.models.resnet18(weights=None).to(self._dev).eval()

    def __call__(self, batch: dict) -> dict:
        import torch

        mel = np.ascontiguousarray(batch["mel"]).reshape(-1, 3, 64, 64)
        x = torch.from_numpy(mel).to(self._dev).float().div_(255.0)
        with torch.inference_mode():
            pred = self._m(x).argmax(1).to("cpu").numpy()
        return {"id": np.asarray(batch["id"]), "pred": pred}


def batcher_thunk(cfg: dict):
    import batcher as bt

    ds = (
        bt.read.parquet(f"{cfg['dir']}/*.parquet")
        .map_batches(mel_stage, output_columns=["id", "mel"], batch_format="pyarrow")
        .map_batches(
            AudioModel,
            output_columns=["id", "pred"],
            batch_format="numpy",
            num_gpus=cfg["num_gpus"],
            batch_size=cfg["batch"],
        )
    )

    def run():
        return _sig(ds.collect(distributed=True))

    return run


def ray_thunk(cfg: dict):
    import ray.data as rd

    conc = _auto_gpu_actors(cfg["num_gpus"])

    class _RayAudio:
        def __init__(self) -> None:
            self._m = AudioModel()

        def __call__(self, b):
            return self._m(b) if len(b["id"]) else {"id": [], "pred": []}

    ds = (
        rd.read_parquet(cfg["dir"])
        .map_batches(mel_stage, num_cpus=1, batch_format="pyarrow", batch_size=cfg["batch"])
        .map_batches(
            _RayAudio,
            concurrency=conc,
            num_gpus=cfg["num_gpus"],
            num_cpus=0,
            batch_size=cfg["batch"],
            batch_format="numpy",
        )
    )

    def run():
        rows = [b for b in ds.iter_batches(batch_format="pyarrow") if b.num_rows]
        return _sig(pa.Table.from_batches([b.combine_chunks().to_batches()[0] for b in rows]))

    return run


def _auto_gpu_actors(num_gpus: float) -> int:
    import ray

    return max(1, int(float(ray.cluster_resources().get("GPU", 1.0)) / max(num_gpus, 0.01)))


def _sig(tbl: pa.Table) -> dict:
    d = tbl.to_pydict()
    return {"rows": tbl.num_rows, "preds": dict(zip(d["id"], d["pred"], strict=False))}


def _agreement(a: dict, b: dict) -> float:
    pa_, pb = a["preds"], b["preds"]
    common = set(pa_) & set(pb)
    return sum(1 for k in common if pa_[k] == pb[k]) / len(common) if common else 0.0


def _with_timeout(fn, t):
    import threading

    def wrapped():
        box: dict = {}

        def run():
            try:
                box["v"] = fn()
            except BaseException as e:
                box["e"] = e

        th = threading.Thread(target=run, daemon=True)
        th.start()
        th.join(t)
        if th.is_alive():
            raise TimeoutError
        if "e" in box:
            raise box["e"]
        return box.get("v")

    return wrapped


def bench(cfg: dict, n: int) -> dict:
    out: dict = {}
    for eng, builder in (("batcher", batcher_thunk), ("ray", ray_thunk)):
        try:
            run = _with_timeout(builder(cfg), cfg["timeout"])
            print(f"  [{eng}] warmup ...")
            warm = run()
            best = float("inf")
            for _ in range(cfg["runs"]):
                t0 = time.perf_counter()
                run()
                best = min(best, time.perf_counter() - t0)
            out[eng] = {"s": best, "sig": warm}
            print(f"  [{eng}] {best:.2f}s  {n / best:.1f} clip/s")
        except TimeoutError:
            out[eng] = {"error": "TIMEOUT"}
        except Exception as e:
            out[eng] = {"error": f"{type(e).__name__}: {e}"}
            print(f"  [{eng}] ERROR {type(e).__name__}: {e}")
        finally:
            if eng == "batcher":
                with contextlib.suppress(Exception):
                    from batcher.dist.executors.map import release_inference_pools

                    release_inference_pools()
                with contextlib.suppress(Exception):
                    from batcher.dist.fleet import release_session_fleet

                    release_session_fleet()
    return out


def _init() -> None:
    import importlib.util

    for var in ("RAY_RUNTIME_ENV_HOOK", "RAY_RUNTIME_ENV_PLUGINS"):
        v = os.environ.get(var)
        if v:
            head = v.lstrip("[{\"' ").split(".")[0].split("[")[0]
            if head and importlib.util.find_spec(head) is None:
                os.environ.pop(var, None)
    import batcher
    from batcher.config import active_config, set_config

    pkg = os.path.dirname(os.path.abspath(batcher.__file__))
    renv = {"py_modules": [pkg], "pip": None, "env_vars": {}}
    base = active_config()
    set_config(
        base.replace(
            distributed=dataclasses.replace(base.distributed, ray_address="auto", runtime_env=renv)
        )
    )
    import ray

    if not ray.is_initialized():
        ray.init(address="auto", runtime_env=renv, logging_level="ERROR", log_to_driver=False)


def main() -> int:
    cfg = _cfg()
    _init()
    import pyarrow.parquet as pq
    import ray

    if not os.path.isdir(cfg["dir"]) or not os.listdir(cfg["dir"]):
        print("generating:", write_shards(cfg["dir"], cfg["n"], cfg["shards"]))
    n = sum(
        pq.read_metadata(os.path.join(cfg["dir"], f)).num_rows
        for f in os.listdir(cfg["dir"])
        if f.endswith(".parquet")
    )
    print(f"cluster: {ray.cluster_resources().get('GPU')} GPU, {len(ray.nodes())} nodes")
    print(f"model=resnet18 audio_clips={n} batch={cfg['batch']} best-of-{cfg['runs']}\n")
    res = bench(cfg, n)
    bm, rm = res.get("batcher", {}).get("s"), res.get("ray", {}).get("s")
    print("\nengine     time_s    clip/s")
    print("-" * 32)
    for eng in ("batcher", "ray"):
        r = res.get(eng, {})
        print(
            f"{eng:<10} {r['error']}"
            if "error" in r
            else f"{eng:<10} {r['s']:>6.2f}  {n / r['s']:>7.1f}"
        )
    if bm and rm:
        print(f"\nbatcher vs ray: {rm / bm:.2f}x  (>1 = batcher faster)")
    sb, sr = res.get("batcher", {}).get("sig"), res.get("ray", {}).get("sig")
    if sb and sr:
        agree = _agreement(sb, sr)
        print(f"correctness: agreement={agree:.3%}  [{'OK' if agree >= 0.999 else 'MISMATCH'}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
