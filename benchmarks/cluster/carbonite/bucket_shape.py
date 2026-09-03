"""Cross-node shuffle throughput against the *shape* of the shuffle, on a live Ray cluster.

A hash shuffle cuts every mapper's output into one bucket per reducer, so a cluster of `W`
workers produces `W^2` buckets and each one shrinks as the cluster grows. This benchmark
holds the bytes moved constant and varies only how finely they are cut, which is the axis a
growing cluster travels along -- so a curve that sags at either end is a transfer that gets
slower the bigger the cluster gets, however fast the link is.

The number to read is the **flatness**: max/min MiB/s across the sweep. A transport whose
rate depends on the bucket count is one whose rate depends on the cluster's width.

Run (needs a cluster with at least two worker nodes)::

    python benchmarks/cluster/carbonite/bucket_shape.py
    python benchmarks/cluster/carbonite/bucket_shape.py --buckets 4,64,1024,4096
    python benchmarks/cluster/carbonite/bucket_shape.py --codec 0,1,2 --reps 3

Every rung is correctness-gated against the producer's own checksum before its timing is
quoted, and the producer actor is killed between rungs so one rung's heap never charges the
next one's measurement.
"""

from __future__ import annotations

import functools
import shutil
import sys
import time
from pathlib import Path

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

# `envinfo` lives in `benchmarks/`, two levels up -- these scripts are invoked as
# `python benchmarks/cluster/<dir>/<name>.py`, so only their own directory is on
# `sys.path` and the import below cannot resolve without this.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from envinfo import machine_fingerprint, require_release_build

print = functools.partial(print, flush=True)

ROWS_PER_BATCH = 16_384
#: Batches moved per rung, held constant so only the bucket *count* varies.
TOTAL_BATCHES = 4096
CODECS = {0: "none", 1: "lz4", 2: "zstd"}

_PKG_SRC = Path(__file__).resolve().parents[3] / "python" / "batcher"
_NFS_PKG_PARENT = Path("/mnt/cluster_storage/carb_bucket_shape/pybatcher")


def _stage_package() -> str:
    """Copy the built batcher package to NFS; return the sys.path entry to prepend."""
    dst = _NFS_PKG_PARENT / "batcher"
    if _NFS_PKG_PARENT.exists():
        shutil.rmtree(_NFS_PKG_PARENT, ignore_errors=True)
    _NFS_PKG_PARENT.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        _PKG_SRC, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"), dirs_exist_ok=True
    )
    return str(_NFS_PKG_PARENT)


@ray.remote(num_cpus=1)
class Producer:
    """Holds the payload on its node and publishes it as `n_buckets` shuffle buckets."""

    def __init__(self, pkg_path: str, n_buckets: int) -> None:
        sys.path.insert(0, pkg_path)
        import numpy as np
        import pyarrow as pa

        per = max(1, TOTAL_BATCHES // n_buckets)
        cats = np.array([f"category_{i:02d}" for i in range(32)])

        def batch(seed: int):
            rng = np.random.default_rng(seed)
            return pa.record_batch(
                {
                    # A sorted low-cardinality key beside an inline low-cardinality string:
                    # the shape a grouped or sorted analytical shuffle actually carries, and
                    # the one whose compressibility the wire codec exists for.
                    "k": np.sort(rng.integers(0, 64, ROWS_PER_BATCH)).astype("int64"),
                    "cat": pa.array(cats[rng.integers(0, 32, ROWS_PER_BATCH)]),
                }
            )

        self._buckets = [[batch(b * 1000 + j) for j in range(per)] for b in range(n_buckets)]
        self._session = None

    def node_ip(self) -> str:
        return ray.util.get_node_ip_address()

    def nbytes(self) -> int:
        return sum(b.get_total_buffer_size() for bucket in self._buckets for b in bucket)

    def checksum(self) -> int:
        return sum(int(b.column("k").to_numpy().sum()) for bucket in self._buckets for b in bucket)

    def set_codec(self, code: int) -> None:
        import batcher._native as nat

        nat.set_flight_transport_config(0, 0, 0, code)

    def publish(self) -> tuple[str, list[str]]:
        from batcher.carbonite.transfer import ShuffleSession, ShuffleTicket

        if self._session is None:
            self._session = ShuffleSession(advertise_host=self.node_ip())
        tickets = []
        for i, bucket in enumerate(self._buckets):
            ticket = ShuffleTicket(1, 0, i, 0)
            self._session.publish(ticket, bucket)
            tickets.append(str(ticket))
        return self._session.addr, tickets

    def ready(self) -> bool:
        return True


@ray.remote(num_cpus=1)
class Consumer:
    """Runs on another node and gathers the whole payload, as a reduce task would."""

    def __init__(self, pkg_path: str) -> None:
        sys.path.insert(0, pkg_path)
        import numpy as np  # noqa: F401
        import pyarrow as pa  # noqa: F401

        self._session = None

    def node_ip(self) -> str:
        return ray.util.get_node_ip_address()

    def _get_session(self):
        from batcher.carbonite.transfer import ShuffleSession

        if self._session is None:
            self._session = ShuffleSession(advertise_host=self.node_ip())
        return self._session

    def gather(self, addr: str, tickets: list[str], reps: int) -> tuple[float, int]:
        """Best-of-`reps` wall time for the whole gather, with its checksum.

        Nothing is cached on the consumer, so every repetition is a fresh cross-node
        transfer rather than a re-read.
        """
        from batcher.carbonite.transfer import ShuffleTicket

        session = self._get_session()
        sources = [(addr, ShuffleTicket(*(int(x) for x in t.split("/")))) for t in tickets]

        def once():
            rows, unreachable = session.gather_concat(sources)
            if unreachable:
                raise RuntimeError(f"unreachable sources: {unreachable}")
            return rows

        once()  # warm the pooled gRPC channels; connection setup is not transfer
        best, checksum = None, None
        for _ in range(reps):
            started = time.perf_counter()
            out = once()
            elapsed = time.perf_counter() - started
            checksum = sum(int(b.column("k").to_numpy().sum()) for b in out)
            del out
            best = elapsed if best is None else min(best, elapsed)
        return best, checksum


def _arg(flag: str, default: str) -> str:
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


def _two_nodes(consumer, timeout_s: float = 2400.0) -> str:
    """Block until the consumer actor lands on a real worker node, and return its IP."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            return ray.get(consumer.node_ip.remote(), timeout=10)
        except ray.exceptions.GetTimeoutError:
            alive = sum(1 for n in ray.nodes() if n["Alive"])
            print(f"  waiting for a worker node... (alive nodes: {alive})")
    raise TimeoutError("no worker node became available within the timeout")


def main() -> None:
    require_release_build()
    print(machine_fingerprint())
    buckets = [int(x) for x in _arg("--buckets", "4,16,64,256,1024,4096").split(",")]
    codecs = [int(x) for x in _arg("--codec", "1").split(",")]
    reps = int(_arg("--reps", "2"))

    print("staging the built batcher package to NFS...")
    pkg = _stage_package()
    ray.init(address="auto", logging_level="ERROR", runtime_env={"pip": None})
    try:
        consumer = Consumer.remote(pkg)
        consumer_ip = _two_nodes(consumer)
        producer_node = next(
            n
            for n in ray.nodes()
            if n["Alive"]
            and n["NodeManagerAddress"] != consumer_ip
            and n["Resources"].get("CPU", 0) > 0
        )
        strategy = NodeAffinitySchedulingStrategy(producer_node["NodeID"], soft=False)
        print(
            f"\n{producer_node['NodeManagerAddress']} -> {consumer_ip}, best-of-{reps}, "
            f"{TOTAL_BATCHES} batches held constant while the bucket count varies.\n"
            "Throughput is effective (logical MiB / wire time), so a codec that shrinks the "
            "wire reads higher.\n"
        )
        print(f"  {'buckets':>8} {'bucket MiB':>11} {'codec':>6} {'MiB/s':>9}")
        print("  " + "-" * 38)
        rates: dict[int, list[float]] = {code: [] for code in codecs}
        for n_buckets in buckets:
            producer = Producer.options(scheduling_strategy=strategy).remote(pkg, n_buckets)
            ray.get(producer.ready.remote())
            if ray.get(producer.node_ip.remote()) == consumer_ip:
                raise RuntimeError("producer and consumer co-located -- not a cross-node run")
            total = ray.get(producer.nbytes.remote())
            expected = ray.get(producer.checksum.remote())
            mib = total / (1 << 20)
            for code in codecs:
                ray.get(producer.set_codec.remote(code))
                addr, tickets = ray.get(producer.publish.remote())
                elapsed, checksum = ray.get(consumer.gather.remote(addr, tickets, reps))
                # Correctness before timing: a rate quoted on a transfer that lost or
                # duplicated a bucket measures nothing.
                if checksum != expected:
                    raise AssertionError(
                        f"{n_buckets} buckets, codec {CODECS[code]}: delivered {checksum}, "
                        f"expected {expected}"
                    )
                rates[code].append(mib / elapsed)
                print(
                    f"  {n_buckets:>8} {mib / n_buckets:>11.2f} {CODECS[code]:>6} "
                    f"{mib / elapsed:>9.0f}"
                )
            ray.kill(producer)  # free its heap before the next rung is built
        print()
        for code in codecs:
            got = rates[code]
            print(
                f"  {CODECS[code]}: {min(got):.0f}-{max(got):.0f} MiB/s across the sweep, "
                f"flatness {max(got) / min(got):.2f}x (1.00x = the rate does not depend on "
                "how finely the shuffle is cut)"
            )
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
