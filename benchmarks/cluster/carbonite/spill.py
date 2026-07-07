"""Carbonite vs a SPILLING Ray object store — the regime Ray Data actually hits at scale.

Ray Data moves shuffle blocks through the object store, which is a bounded slice of
each node's RAM (~30%). A real shuffle's blocks exceed it, so Ray **spills** blocks to
disk and must restore them (read from disk back into the store, evicting others — a
thrash when the store stays full) before it can serve a cross-node fetch. Carbonite
never touches the object store: it streams batches from the producer's heap over Arrow
Flight, so it is unaffected by object-store pressure. This benchmark overflows a real
worker's object store and measures a cross-node fetch both ways.

To make the producer's store the one that overflows, the **producer** runs on the
constrained worker (``num_cpus=1`` forces a real 32 GB node) and the **consumer** on
the head. The producer fills the store incrementally (generate → ``ray.put`` → drop the
heap copy), so its own heap stays small while the object store fills past its cap and
spills. The Carbonite side then serves the *same* data from the Flight store (the Ray
refs are released first so both never sit resident at once).

Run::

    .venv/bin/python benchmarks/cluster/carbonite/spill.py               # 1.6x the store
    .venv/bin/python benchmarks/cluster/carbonite/spill.py 2.0 16        # 2x store, 4 MiB blocks
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
_BATCH_BYTES = ROWS_PER_BATCH * 16  # int64 k + float64 v

_PKG_SRC = Path(__file__).resolve().parents[3] / "python" / "batcher"
_NFS_PKG_PARENT = Path("/mnt/cluster_storage/carb_xnode/pybatcher")


def _stage_package() -> str:
    dst = _NFS_PKG_PARENT / "batcher"
    if _NFS_PKG_PARENT.exists():
        shutil.rmtree(_NFS_PKG_PARENT, ignore_errors=True)
    _NFS_PKG_PARENT.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        _PKG_SRC, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"), dirs_exist_ok=True
    )
    return str(_NFS_PKG_PARENT)


def _make_partition(n_batches: int, seed: int):
    import numpy as np
    import pyarrow as pa

    rng = np.random.default_rng(seed)
    return [
        pa.record_batch(
            {
                "k": rng.integers(0, 1_000_000, ROWS_PER_BATCH).astype("int64"),
                "v": rng.standard_normal(ROWS_PER_BATCH),
            }
        )
        for _ in range(n_batches)
    ]


def _part_checksum(part) -> int:
    return sum(int(b.column("k").to_numpy().sum()) for b in part)


@ray.remote(num_cpus=1)
class Producer:
    """On the constrained worker; overflows its object store, and serves the same data
    from an in-memory Flight store for the Carbonite comparison."""

    def __init__(self, pkg_path: str, n_batches: int) -> None:
        sys.path.insert(0, pkg_path)
        self._n_batches = n_batches
        self._n_parts = 0
        self._expected = 0
        self._refs: list = []
        self._session = None

    def node_ip(self) -> str:
        return ray.util.get_node_ip_address()

    def fill_object_store(self, target_bytes: int) -> tuple[int, float]:
        """Generate blocks and ``ray.put`` them until ~`target_bytes`, dropping each heap
        copy right after so the producer's own heap stays small while the store overflows.
        Returns `(n_parts, put_seconds)`. The producer holds the refs so they stay pinned
        (and spilled to disk once the store is full)."""
        part_bytes = self._n_batches * _BATCH_BYTES
        n_parts = max(1, target_bytes // part_bytes)
        t0 = time.perf_counter()
        for i in range(n_parts):
            part = _make_partition(self._n_batches, seed=i)
            self._expected += _part_checksum(part)
            self._refs.append(ray.put(part))  # into the object store; spills when full
            del part
        self._n_parts = int(n_parts)
        return self._n_parts, time.perf_counter() - t0

    def refs(self) -> list:
        return self._refs

    def expected(self) -> int:
        return self._expected

    def total_bytes(self) -> int:
        return self._n_parts * self._n_batches * _BATCH_BYTES

    def drop_object_store(self) -> None:
        """Release the Ray refs so the object store (and its disk spill) frees before the
        Carbonite side allocates the same data in the Flight store — the 32 GB node can't
        hold both at once."""
        self._refs = []
        import gc

        gc.collect()

    def publish_flight(self) -> tuple[str, list[str], float]:
        """Regenerate the same data into an in-memory Flight store and advertise it. This
        is Carbonite's make-available: served zero-copy from the producer heap, no object
        store, no spill. Returns `(addr, tickets, publish_seconds)`."""
        from batcher.carbonite.transfer import ShuffleSession, ShuffleTicket

        self._session = ShuffleSession(advertise_host=self.node_ip())
        tickets = []
        t0 = time.perf_counter()
        for i in range(self._n_parts):
            part = _make_partition(self._n_batches, seed=i)
            t = ShuffleTicket(1, 0, i, 0)
            self._session.publish(t, part)
            tickets.append(str(t))
            del part
        return self._session.addr, tickets, time.perf_counter() - t0


@ray.remote(num_cpus=0)
class Consumer:
    """On the head; fetches the whole dataset cross-node both ways and times it."""

    def __init__(self, pkg_path: str) -> None:
        sys.path.insert(0, pkg_path)
        self._session = None

    def node_ip(self) -> str:
        return ray.util.get_node_ip_address()

    def set_conns(self, n: int) -> None:
        import batcher._native as nat

        nat.set_flight_transport_config(0, 0, n)

    def get_object_store(self, refs: list) -> tuple[float, int]:
        """Cold cross-node ``ray.get`` of every ref. Spilled blocks must be restored from
        the producer's disk (and evict others when the store is full) before transfer."""
        t0 = time.perf_counter()
        out = ray.get(refs)
        dt = time.perf_counter() - t0
        chk = sum(int(b.column("k").to_numpy().sum()) for part in out for b in part)
        return dt, chk

    def gather_flight(self, addr: str, tickets: list[str], fan_in: int) -> tuple[float, int]:
        from batcher.carbonite.transfer import ShuffleSession, ShuffleTicket

        if self._session is None:
            self._session = ShuffleSession(credits=32, advertise_host=self.node_ip())
        sources = [(addr, ShuffleTicket(*(int(x) for x in t.split("/")))) for t in tickets]
        t0 = time.perf_counter()
        rows, unreachable = self._session.gather_concat(sources, fan_in=fan_in)
        dt = time.perf_counter() - t0
        if unreachable:
            raise RuntimeError(f"unreachable: {unreachable}")
        chk = sum(int(b.column("k").to_numpy().sum()) for b in rows)
        return dt, chk


def _wait_for_worker(actor, timeout_s: float = 2400.0) -> str:
    deadline = time.time() + timeout_s
    last = None
    while time.time() < deadline:
        try:
            return ray.get(actor.node_ip.remote(), timeout=10)
        except ray.exceptions.GetTimeoutError:
            alive = sum(1 for n in ray.nodes() if n["Alive"])
            if alive != last:
                print(f"  waiting for a worker node... (alive nodes: {alive})")
                last = alive
            time.sleep(5)
    raise TimeoutError("no worker node became available in time")


def main() -> None:
    overflow = float(sys.argv[1]) if len(sys.argv) > 1 else 1.6
    n_batches = int(sys.argv[2]) if len(sys.argv) > 2 else 16  # 16 x 256 KiB = 4 MiB blocks

    print("staging freshly-built batcher package to NFS...")
    pkg_path = _stage_package()

    ray.init(address="auto", logging_level="ERROR", runtime_env={"pip": None})
    try:
        head_ip = ray.util.get_node_ip_address()
        print("launching producer on a worker (num_cpus=1 forces a 32 GB node)...")
        producer = Producer.remote(pkg_path, n_batches)
        producer_ip = _wait_for_worker(producer)
        print(f"  producer on worker {producer_ip}")
        if producer_ip == head_ip:
            raise RuntimeError("producer landed on the head — need it on a constrained worker")

        # Consumer pinned to the head (a different node than the producer).
        head_node = next(
            n for n in ray.nodes() if n["Alive"] and n["NodeManagerAddress"] == head_ip
        )
        consumer = Consumer.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(head_node["NodeID"], soft=False)
        ).remote(pkg_path)
        consumer_ip = ray.get(consumer.node_ip.remote())
        print(f"  consumer on {consumer_ip}")

        prod_node = next(
            n for n in ray.nodes() if n["Alive"] and n["NodeManagerAddress"] == producer_ip
        )
        store = float(prod_node.get("Resources", {}).get("object_store_memory", 8 << 30))
        target = int(store * overflow)
        store_gib, target_gib = store / (1 << 30), target / (1 << 30)
        print(
            f"\n  producer object store ~{store_gib:.1f} GiB; filling to {target_gib:.1f} GiB "
            f"({overflow:.1f}x -> ~{(overflow - 1) * 100:.0f}% spills to disk)"
        )

        n_parts, put_t = ray.get(producer.fill_object_store.remote(target))
        expected = ray.get(producer.expected.remote())
        total_bytes = ray.get(producer.total_bytes.remote())
        mb = total_bytes / (1 << 20)
        print(
            f"  put {n_parts} blocks x {n_batches * 256} KiB = {mb / 1024:.1f} GiB into the "
            f"store in {put_t:.1f}s (spilled beyond the cap)\n"
        )

        # Ray: cold cross-node get of the (partly spilled) dataset.
        refs = ray.get(producer.refs.remote())
        ray_t, ray_chk = ray.get(consumer.get_object_store.remote(refs))
        assert ray_chk == expected, f"ray wrong data: {ray_chk} != {expected}"
        del refs
        ray.get(producer.drop_object_store.remote())  # free the store+spill before Carbonite

        # Carbonite: same data served from the Flight store (memory), fetched cross-node.
        ray.get(consumer.set_conns.remote(4))
        addr, tickets, pub_t = ray.get(producer.publish_flight.remote())
        car_t, car_chk = ray.get(
            consumer.gather_flight.remote(addr, tickets, min(64, len(tickets)))
        )
        assert car_chk == expected, f"carbonite wrong data: {car_chk} != {expected}"

        print("cross-node fetch of the full dataset (correctness-checked):")
        print(
            f"  {'ray object store (spilling)':<30} {ray_t * 1e3:9.0f} ms   {mb / ray_t:8.1f} MiB/s"
        )
        print(
            f"  {'carbonite flight (memory)':<30} {car_t * 1e3:9.0f} ms   {mb / car_t:8.1f} MiB/s"
        )
        print(f"\n  transfer:   carbonite {ray_t / car_t:.2f}x faster than ray")
        ray_e2e, car_e2e = put_t + ray_t, pub_t + car_t
        print(
            f"  end-to-end: carbonite {ray_e2e / car_e2e:.2f}x faster"
            f"  (ray put {put_t:.1f}s + get {ray_t:.1f}s  vs  carbonite publish {pub_t:.1f}s "
            f"+ fetch {car_t:.1f}s)"
        )
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
