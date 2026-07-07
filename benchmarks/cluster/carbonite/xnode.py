"""Carbonite cross-node transfer vs Ray's object store, on a live Ray cluster.

Ray Data moves shuffle blocks *between nodes* through the Ray object store: the
producer ``ray.put``s a block on its node and the consumer ``ray.get``s it, which
pulls the object over the network via Ray's object manager. Carbonite instead moves
the same Arrow batches directly over credit-bounded Arrow Flight (gRPC), never
touching the object store. This benchmark places a **producer** actor and a
**consumer** actor on *different* nodes and moves the identical partition set both
ways, checks the bytes delivered match, and reports throughput and the Carbonite/Ray
ratio — the true cross-node comparison the single-process benchmark
(``shuffle_vs_object_store.py``) cannot make (on one node Ray uses shared memory, so
there is no network transfer to compare against).

It reports two numbers. **transfer** is the cross-node move only — both are NIC-bound,
and the fix here (striping a peer's fetches across several TCP connections, since a
single flow is capped below a cloud NIC's line rate) lifts Carbonite from ~0.6x to
~1.7x Ray. **end-to-end** also charges each side for making a block transferable: Ray
Data must ``ray.put`` every block (serialize/copy it into plasma) before it can move,
which Carbonite avoids by serving batches zero-copy from the producer heap — so the
end-to-end gap is ~3x and widens as blocks shrink.

Run (needs an autoscaling cluster; forces one worker node to launch)::

    .venv/bin/python benchmarks/cluster/carbonite/xnode.py               # 128 MiB, one config
    .venv/bin/python benchmarks/cluster/carbonite/xnode.py 8 64          # 8 parts x 64 batches
    .venv/bin/python benchmarks/cluster/carbonite/xnode.py --sweep 256,1024,4096  # curve

The consumer actor holds ``num_cpus=1`` so it pins a real worker node for the run's
duration (the head is a 0-CPU control node), which both provides the second node and
keeps the autoscaler from scaling it away mid-measurement.
"""

from __future__ import annotations

import functools
import shutil
import sys
import time
from pathlib import Path

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

print = functools.partial(print, flush=True)

ROWS_PER_BATCH = 16_384

# Ship the *freshly built* package to NFS so every node loads identical bytes (the
# workspace copy on a just-launched worker can lag the local build). Actors prepend
# this to sys.path before importing batcher.
_PKG_SRC = Path(__file__).resolve().parents[3] / "python" / "batcher"
_NFS_ROOT = Path("/mnt/cluster_storage/carb_xnode")
_NFS_PKG_PARENT = _NFS_ROOT / "pybatcher"


def _stage_package() -> str:
    """Copy the built batcher package to NFS; return the sys.path entry to prepend."""
    dst = _NFS_PKG_PARENT / "batcher"
    if _NFS_PKG_PARENT.exists():
        shutil.rmtree(_NFS_PKG_PARENT, ignore_errors=True)
    _NFS_PKG_PARENT.mkdir(parents=True, exist_ok=True)
    # Copy the package but skip caches; the compiled _native.abi3.so comes along.
    shutil.copytree(
        _PKG_SRC, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"), dirs_exist_ok=True
    )
    return str(_NFS_PKG_PARENT)


@ray.remote(num_cpus=0)
class Producer:
    """Owns the data on its node; can expose it via Flight or the object store."""

    def __init__(
        self,
        pkg_path: str,
        n_partitions: int,
        n_batches: int,
        seed0: int,
        compressible: bool = False,
        rows: int = ROWS_PER_BATCH,
    ) -> None:
        sys.path.insert(0, pkg_path)
        import os

        # Batcher runs one worker per node (owns all cores), and its compression-aware
        # runtime sizes threads to the cores. Pin it here so the benchmark reflects that
        # (the OnceLock runtime is built on first Flight op, before which production sets
        # the codec — replicated deterministically by this).
        os.environ.setdefault("BATCHER_SHUFFLE_RT_THREADS", str(min(32, os.cpu_count() or 8)))
        import numpy as np
        import pyarrow as pa

        self._pa = pa
        self._np = np
        self._compressible = compressible
        self._rows = rows
        self._partitions = [self._make(n_batches, seed0 + i) for i in range(n_partitions)]
        self._sessions_pub: list = []

    def _make(self, n_batches: int, seed: int):
        import numpy as np
        import pyarrow as pa

        rng = np.random.default_rng(seed)
        if self._compressible:
            # Realistic dimension-heavy analytical shuffle: an inline low-cardinality string
            # column (32 categories — status/country/type, ubiquitous in real data and
            # commonly shuffled inline, not dict-encoded) plus a sorted low-cardinality key
            # (long runs, e.g. grouped/sorted by a dimension). This is a common shuffle shape
            # and it compresses ~10x — unlike Ray's uncompressed object-store blocks. (Random
            # data, below, is the worst case: incompressible, LZ4 gives up fast, effective ~=
            # raw throughput.)
            cats = np.array([f"category_{i:02d}" for i in range(32)])
            return [
                pa.record_batch(
                    {
                        "k": np.sort(rng.integers(0, 64, self._rows)).astype("int64"),
                        "cat": pa.array(cats[rng.integers(0, 32, self._rows)]),
                    }
                )
                for _ in range(n_batches)
            ]
        return [
            pa.record_batch(
                {
                    "k": rng.integers(0, 1_000_000, self._rows).astype("int64"),
                    "v": rng.standard_normal(self._rows),
                }
            )
            for _ in range(n_batches)
        ]

    def node_ip(self) -> str:
        return ray.util.get_node_ip_address()

    def set_compression(self, code: int) -> None:
        """Set this producer's Flight wire codec (0 none / 1 lz4 / 2 zstd). Compression
        happens in the producer's Flight server, so it is controlled here."""
        import batcher._native as nat

        nat.set_flight_transport_config(0, 0, 0, code)

    def nbytes(self) -> int:
        return sum(b.get_total_buffer_size() for part in self._partitions for b in part)

    def checksum(self) -> int:
        return sum(int(b.column("k").to_numpy().sum()) for part in self._partitions for b in part)

    # --- Carbonite Flight path ------------------------------------------------
    def publish_flight(self, n_servers: int = 1) -> list[tuple[str, list[str]]]:
        """Publish the partitions across `n_servers` Flight servers on this node.

        Each server has its own address, so a consumer's pooled client opens one TCP
        connection per server — i.e. `n_servers` parallel flows. With `n_servers=1` this
        is the normal single-endpoint path; raising it tests whether throughput is capped
        by a single TCP flow (the AWS single-flow limit) rather than the link or the CPU.
        """
        from batcher.carbonite.transfer import ShuffleSession, ShuffleTicket

        self._sessions_pub = [
            ShuffleSession(advertise_host=self.node_ip()) for _ in range(n_servers)
        ]
        out: list[tuple[str, list[str]]] = []
        per = [[] for _ in range(n_servers)]
        for i, part in enumerate(self._partitions):
            s = i % n_servers
            t = ShuffleTicket(1, 0, i, 0)
            self._sessions_pub[s].publish(t, part)
            per[s].append(str(t))
        for s in range(n_servers):
            out.append((self._sessions_pub[s].addr, per[s]))
        return out

    def time_publish(self, n_servers: int = 1) -> tuple[float, list]:
        """Publish to Flight and return `(make_available_seconds, endpoints)`.

        Carbonite serves batches straight from this producer's heap, so "publishing" is
        registering a zero-copy view (an Arc bump) in the local store — near-free. This
        is the make-available cost on Carbonite's side of an apples-to-apples transfer.
        """
        t0 = time.perf_counter()
        endpoints = self.publish_flight(n_servers)
        return time.perf_counter() - t0, endpoints

    # --- Ray object store path ------------------------------------------------
    def put_object_store(self) -> list:
        # FRESH refs each call, pinned in this node's object store. Fresh refs matter:
        # once a consumer ray.gets an object it is cached on the consumer's node, so
        # re-getting the same ref reads local shared memory, not the network. New refs
        # per rep force a genuine cross-node object-manager pull every time.
        return [ray.put(part) for part in self._partitions]

    def time_put_object_store(self) -> tuple[float, list]:
        """`ray.put` every partition and return `(make_available_seconds, refs)`.

        A block is not transferable until it is *in* the object store, so Ray Data pays
        to serialize/copy every block from the producer heap into plasma shared memory
        before any cross-node move. That copy is the make-available cost on Ray's side —
        pure overhead that Carbonite's zero-copy publish avoids — and it is part of the
        transfer Ray Data actually performs.
        """
        t0 = time.perf_counter()
        refs = self.put_object_store()
        return time.perf_counter() - t0, refs

    def ready(self) -> bool:
        return True


@ray.remote(num_cpus=1)
class Consumer:
    """Runs on a *different* node; fetches the partitions both ways and times it."""

    def __init__(self, pkg_path: str) -> None:
        sys.path.insert(0, pkg_path)
        import os

        os.environ.setdefault("BATCHER_SHUFFLE_RT_THREADS", str(min(32, os.cpu_count() or 8)))
        import numpy as np  # noqa: F401
        import pyarrow as pa  # noqa: F401

        self._sessions: dict[int, object] = {}

    def node_ip(self) -> str:
        return ray.util.get_node_ip_address()

    def batcher_path(self) -> str:
        import batcher

        return batcher.__file__

    def set_conns(self, n: int) -> None:
        """Set the process-wide per-peer TCP-connection striping bound (takes effect for
        pools built afterward, i.e. for peer addresses fetched from for the first time)."""
        import batcher._native as nat

        nat.set_flight_transport_config(0, 0, n)

    def _session_for(self, credits: int):
        from batcher.carbonite.transfer import ShuffleSession

        s = self._sessions.get(credits)
        if s is None:
            s = ShuffleSession(credits=credits, advertise_host=self.node_ip())
            self._sessions[credits] = s
        return s

    @staticmethod
    def _checksum(partitions) -> int:
        return sum(int(b.column("k").to_numpy().sum()) for part in partitions for b in part)

    def flight_gather(self, endpoints: list, credits: int, reps: int) -> tuple:
        """Concurrent reducer gather (the real path) over credit-bounded Flight.

        `endpoints` is a list of `(addr, [ticket_str, ...])`; sources are pooled across
        all of them, so multiple producer addresses become multiple TCP flows. Carbonite
        does not cache on the consumer, so each gather is a fresh cross-node transfer —
        best-of-`reps` is a clean throughput read. `fan_in` = total sources so every
        upstream streams at once, as a real reduce task would.
        """
        from batcher.carbonite.transfer import ShuffleTicket

        session = self._session_for(credits)
        sources = [
            (addr, ShuffleTicket(*(int(x) for x in t.split("/"))))
            for addr, tickets in endpoints
            for t in tickets
        ]
        fan_in = len(sources)

        def once():
            rows, unreachable = session.gather_concat(sources, fan_in=fan_in)
            if unreachable:
                raise RuntimeError(f"unreachable sources: {unreachable}")
            return rows

        once()  # warm the pooled gRPC channel (connection setup is not transfer)
        best, chk = None, None
        for _ in range(reps):
            t0 = time.perf_counter()
            out = once()
            dt = time.perf_counter() - t0
            chk = sum(int(b.column("k").to_numpy().sum()) for b in out)
            best = dt if best is None else min(best, dt)
        return best, chk

    def get_cold(self, refs: list) -> tuple[float, int]:
        """One COLD cross-node ray.get of freshly-put refs (never fetched here before).

        Timed once per call because after the first get Ray caches the object on this
        node; the driver hands fresh refs each rep so every measurement is a real pull.
        """
        t0 = time.perf_counter()
        out = ray.get(refs)
        dt = time.perf_counter() - t0
        return dt, self._checksum(out)


def _wait_two_nodes(consumer: Consumer, timeout_s: float = 2400.0) -> str:
    """Block until the consumer actor is scheduled on a real (non-head) worker node."""
    deadline = time.time() + timeout_s
    last = None
    while time.time() < deadline:
        try:
            ip = ray.get(consumer.node_ip.remote(), timeout=10)
            return ip
        except ray.exceptions.GetTimeoutError:
            alive = sum(1 for n in ray.nodes() if n["Alive"])
            if alive != last:
                print(f"  waiting for a worker node to launch... (alive nodes: {alive})")
                last = alive
            time.sleep(5)
    raise TimeoutError("no worker node became available within the timeout")


_CODECS = {0: "none", 1: "lz4", 2: "zstd"}


def _measure(consumer, producer, expected: int, reps: int) -> dict:
    """Measure Ray vs Carbonite cross-node transfer for one already-built producer.

    Ray transfer is a cold `ray.get`; Carbonite transfer is the production gather at
    conns=4, swept over the wire codec (none/lz4/zstd) so the compression win shows.
    Returns best-of-`reps` seconds for each. Throughput is computed on *logical* (uncom-
    pressed) bytes by the caller, so a compressed codec reports effective throughput.
    """
    ray_xfer = ray_put = None
    for _ in range(reps):
        put_t, refs = ray.get(producer.time_put_object_store.remote())
        t, chk = ray.get(consumer.get_cold.remote(refs))
        assert chk == expected, f"ray delivered wrong data: {chk} != {expected}"
        ray_xfer = t if ray_xfer is None else min(ray_xfer, t)
        ray_put = put_t if ray_put is None else min(ray_put, put_t)

    ray.get(consumer.set_conns.remote(4))
    car_by_codec: dict[int, float] = {}
    car_pub = None
    for code in (0, 1, 2):
        ray.get(producer.set_compression.remote(code))
        pub_t, endpoints = ray.get(producer.time_publish.remote(1))
        t, chk = ray.get(consumer.flight_gather.remote(endpoints, 32, reps))
        assert chk == expected, f"carbonite(codec={code}) wrong data: {chk} != {expected}"
        car_by_codec[code] = t
        car_pub = pub_t if car_pub is None else min(car_pub, pub_t)

    return {
        "ray_xfer": ray_xfer,
        "ray_put": ray_put,
        "car_by_codec": car_by_codec,
        "car_xfer": min(car_by_codec.values()),
        "car_pub": car_pub,
    }


def main() -> None:
    # Forms (append --compressible for realistic post-sort/grouped shuffle data):
    #   xnode.py [n_partitions] [n_batches] [reps]      — one config
    #   xnode.py --sweep 256,1024,4096 [reps]           — block-count curve
    flags = {"--compressible", "--big-consumer"}
    # `--rows N` sets the per-block row count (default 16384); smaller blocks expose the
    # make-available asymmetry (Ray's per-block ray.put serialize vs Carbonite's zero-copy
    # publish) — the regime a large shuffle's W^2 fanout produces.
    rows = ROWS_PER_BATCH
    raw = sys.argv[1:]
    if "--rows" in raw:
        i = raw.index("--rows")
        rows = int(raw[i + 1])
        raw = raw[:i] + raw[i + 2 :]
    argv = [a for a in raw if a not in flags]
    compressible = "--compressible" in sys.argv
    # Place the consumer (decompress side) on the many-core head instead of the 16-core
    # worker, to test whether ZSTD's ratio is realized with more consumer parallelism.
    big_consumer = "--big-consumer" in sys.argv
    sweep = len(argv) > 0 and argv[0] == "--sweep"
    if sweep:
        counts = [int(x) for x in argv[1].split(",")]
        n_batches = 1
        reps = int(argv[2]) if len(argv) > 2 else 5
    else:
        counts = [int(argv[0]) if len(argv) > 0 else 8]
        n_batches = int(argv[1]) if len(argv) > 1 else 64
        reps = int(argv[2]) if len(argv) > 2 else 5

    print("staging freshly-built batcher package to NFS...")
    pkg_path = _stage_package()

    # Drop the workspace's inherited pip runtime env (it pins an unresolvable
    # `batcher-engine[delta]` that fails to install on a fresh worker); the base image
    # already carries pyarrow/numpy, and batcher itself loads from the NFS sys.path entry.
    head_ip = ray.util.get_node_ip_address()
    ray.init(address="auto", logging_level="ERROR", runtime_env={"pip": None})
    try:
        if big_consumer:
            # Producer forces a worker (num_cpus=1); consumer pinned to the many-core head.
            print("launching producer on a worker (num_cpus=1); consumer on the head...")
            probe = Consumer.options(num_cpus=1).remote(pkg_path)
            worker_ip = _wait_two_nodes(probe)
            head_node = next(
                n for n in ray.nodes() if n["Alive"] and n["NodeManagerAddress"] == head_ip
            )
            consumer = Consumer.options(
                num_cpus=0,  # the head is a 0-CPU control node
                scheduling_strategy=NodeAffinitySchedulingStrategy(head_node["NodeID"], soft=False),
            ).remote(pkg_path)
            consumer_ip = ray.get(consumer.node_ip.remote())
            prod_node = next(
                n for n in ray.nodes() if n["Alive"] and n["NodeManagerAddress"] == worker_ip
            )
            prod_strategy = NodeAffinitySchedulingStrategy(prod_node["NodeID"], soft=False)
            prod_extra = {"num_cpus": 1}
            print(f"  consumer(decompress) on head {consumer_ip}; producer on worker {worker_ip}")
        else:
            print("launching consumer actor (num_cpus=1 -> forces a worker node)...")
            consumer = Consumer.remote(pkg_path)
            consumer_ip = _wait_two_nodes(consumer)
            alive = [n for n in ray.nodes() if n["Alive"]]
            prod_node = next((n for n in alive if n["NodeManagerAddress"] != consumer_ip), None)
            if prod_node is None:
                raise RuntimeError("could not find a second node for the producer")
            prod_strategy = NodeAffinitySchedulingStrategy(prod_node["NodeID"], soft=False)
            prod_extra = {}
            print(f"  consumer on node {consumer_ip}")
        print(f"  consumer loaded batcher from: {ray.get(consumer.batcher_path.remote())}")

        shape = (
            "compressible (sorted low-cardinality — post-sort/grouped shuffle)"
            if compressible
            else "random (incompressible worst case)"
        )
        print(
            f"\ncross-node {prod_node['NodeManagerAddress']} -> {consumer_ip}, best-of-{reps}, "
            f"data = {shape}.\n  Throughput is effective (logical MiB / wire time), so a "
            "codec that shrinks the wire shows a higher number. Ray moves object-store "
            "blocks UNCOMPRESSED.\n"
        )
        hdr = (
            f"  {'blocks':>7} {'blkKiB':>7} │ {'ray xfer':>9} {'car xfer':>9} {'xfer x':>7} │ "
            f"{'ray e2e':>9} {'car e2e':>9} {'e2e x':>7}"
        )
        print(hdr)
        print("  " + "─" * (len(hdr) - 2))

        for n_partitions in counts:
            producer = Producer.options(scheduling_strategy=prod_strategy, **prod_extra).remote(
                pkg_path, n_partitions, n_batches, 0, compressible, rows
            )
            ray.get(producer.ready.remote())
            producer_ip = ray.get(producer.node_ip.remote())
            if producer_ip == consumer_ip:
                raise RuntimeError("producer and consumer co-located — not cross-node")
            total_bytes = ray.get(producer.nbytes.remote())
            expected = ray.get(producer.checksum.remote())
            mb = total_bytes / (1 << 20)
            blk_kib = total_bytes / n_partitions / 1024

            m = _measure(consumer, producer, expected, reps)
            # Per-codec effective throughput (diagnostic: is ZSTD realizing its ratio?).
            percodec = "  ".join(
                f"{_CODECS[c]}={mb / t:.0f}" for c, t in sorted(m["car_by_codec"].items())
            )
            print(f"    [codec MiB/s: {percodec}]")
            # Transfer: cross-node move (best codec). End-to-end also charges make-available
            # (Ray: ray.put serialize-into-plasma; Carbonite: zero-copy publish) — both real
            # costs of moving a block. On compressible data the three wins stack: compression
            # + zero-copy publish + NIC-saturating flows.
            car_xfer = min(m["car_by_codec"].values())
            ray_xfer_bw, car_xfer_bw = mb / m["ray_xfer"], mb / car_xfer
            ray_e2e_bw = mb / (m["ray_put"] + m["ray_xfer"])
            car_e2e_bw = mb / (m["car_pub"] + car_xfer)
            print(
                f"  {n_partitions:>7} {blk_kib:>7.0f} │ "
                f"{ray_xfer_bw:>8.0f}↑ {car_xfer_bw:>8.0f}↑ {car_xfer_bw / ray_xfer_bw:>6.2f}x │ "
                f"{ray_e2e_bw:>8.0f}↑ {car_e2e_bw:>8.0f}↑ {car_e2e_bw / ray_e2e_bw:>6.2f}x"
            )
            ray.kill(producer)  # free the producer's heap + object store before the next config

        print(
            "\n  MiB/s of LOGICAL data. xfer = cross-node move only (best codec); e2e also "
            "charges make-available. Carbonite's three wins stack over Ray's object store: "
            "wire compression (Ray moves blocks uncompressed) + zero-copy publish (Ray "
            "serializes into plasma) + NIC-saturating multi-flow."
        )
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
