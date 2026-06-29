# RFC: Multi-GPU placement, GPU-resident transport, and fleet persistence

**Status:** Proposed (not implemented). Requires maintainer sign-off and validation on a
real multi-GPU / mixed CPU+GPU cluster before any code lands.

This RFC collects the distributed-execution items that are *not* safe to implement
blind: they either bend a hard invariant, add public API surface, or depend on Ray
features that cannot be validated single-node in CI. The companion work that *was*
safe (PG-leak fix, GPU-aware autoscaling, PACK/SPREAD selection, the heterogeneity
model, object-store knobs) has already shipped. The scheduling contract
(`plan/resource.py::SchedulingEnvelope`) already carries the seams these proposals
need: `gpu_collective`, `placement_strategy`, `prefer_cpu_only_nodes`.

## Background: the invariants this must respect

- **Arrow is the only columnar contract** (CLAUDE.md invariant #3): every operator
  boundary speaks Arrow `RecordBatch`. A `torch.Tensor` passed between Batcher-
  orchestrated stages is a non-Arrow boundary.
- **Ray is scheduling-only; the data plane bypasses the object store** via Arrow
  Flight. Bulk data never moves as Ray objects.
- **`kyber`/`carbonite`/`core` independence; Carbonite has no live cluster topology**
  (its envelope is built before Ray initializes). Topology-dependent resolution lives
  in `dist`.

## Proposal 1 (3a): collective-aware STRICT_PACK placement for multi-GPU UDFs

**Problem.** A model-parallel / pipeline-parallel inference UDF (sharded transformer,
vLLM) runs its *own* NCCL collectives across several GPUs. Today Batcher's GPU actor
pool (`dist/executors/map.py::_drive_actor_pool` / `_run_resident_pool`) spawns actors
with only `_gpu_options` (`num_gpus`, `accelerator_type`) and **no placement group**,
so on a multi-node cluster the actors scatter and the intra-model collective crosses
slow inter-node links instead of NVLink.

**Proposal.** When the stage is flagged as a collective, gang-schedule its GPU actors
**co-located** with a `STRICT_PACK` placement group of GPU bundles (one node / NVLink
island), so the UDF's NCCL group is fast. **Batcher never touches a tensor** — the
Arrow contract at the operator boundary is unchanged; only *placement* changes. This is
the invariant-safe way to "work very well for multi-GPU."

**Wiring (the part that needs sign-off):**
1. Public API: add `map_batches(..., gpu_collective: bool = False)` (a commitment —
   needs a docstring, an `Examples` doctest, and a differential/equivalence test). This
   is the only user-visible signal that a UDF does its own multi-GPU collective.
2. Plumb it onto the `MapBatches` plan node → onto `SchedulingEnvelope.gpu_collective`
   (the field already exists) → set `placement_strategy="STRICT_PACK"` for that stage.
3. `dist`: a `scheduling.py` helper creates a `STRICT_PACK` placement group of
   `{"GPU": n, "CPU": m}` bundles and binds the actor pool to it via
   `PlacementGroupSchedulingStrategy`. The pool path in `map.py` is **over the 500-line
   limit**, so the placement helper lives in `scheduling.py`; `map.py` only calls it.
4. `ml/gpu.py`: a helper reporting same-node / NVLink GPU count so the collective group
   is sized to a single island (reuse `detect_backend`/`gpu_vram_gb`).

**Why deferred.** Correct STRICT_PACK behavior (and the failure mode when an island is
too small) can only be validated on real multi-GPU hardware; an unvalidated gang
reservation can hang in `PENDING`. Ship behind the existing `gpu_collective` seam after
cluster validation.

## Proposal 2 (3b): GPU-resident transport (RDT / Compiled Graphs) between GPU stages

**Problem.** Two consecutive GPU stages (e.g. embed → rerank) currently hand their
intermediate across Flight as Arrow, i.e. GPU→CPU(Arrow)→Flight→CPU→GPU. Ray Direct
Transport (NCCL/NIXL) or Ray Compiled Graphs would keep the tensor on-device.

**Why this needs an RFC, not a patch.** Passing a `torch.Tensor` edge between two
Batcher-orchestrated stages **violates invariant #3** (Arrow-only columnar contract).
It is a genuine win but a genuine architecture change. Additional constraints:
- RDT is **alpha** and needs **Ray ≥ 2.44** (current floor `>=2.9`). Must be
  runtime-capability-gated, not a hard floor bump.
- RDT objects are **mutable** (a reference, not a copy); a producer mutating in place
  can corrupt a consumer. Requires `ray.experimental.wait_tensor_freed` discipline.
- It introduces a non-Arrow intermediate that must be **opt-in** and produce results
  **identical** to the Arrow path (an equivalence test, GPU-gated).

**Proposed shape (for discussion):**
- Amend invariant #3 with a documented, narrow, opt-in exception: *a non-Arrow
  (`torch.Tensor`) intermediate is permitted strictly between two adjacent GPU UDF
  stages of a single `map_batches` pipeline, behind `gpu_tensor_transport`, with an
  Arrow fallback.* The relational shuffle stays Arrow/Flight, always.
- Config: `DistributedConfig.gpu_tensor_transport: str = "off"` (`off|auto|nccl|nixl`),
  mirroring the existing opt-in `stream_inference`.
- Backend selection in `ml/gpu.py::detect_backend`: NVIDIA→`nccl`, RDMA/cross-vendor→
  `nixl`, CPU-torch→`gloo`; surfaced as `SchedulingEnvelope.gpu_tensor_transport`.
- `dist/streaming/pipeline.py`: when enabled AND both stages are GPU AND a backend is
  available, route the hand-off through an RDT collective group instead of
  `session.publish`/`run_split`; otherwise fall back to today's Flight path unchanged.
- Compiled Graphs (optional): for a *fixed* resident multi-GPU inference DAG
  (`resident_inference_pools`), wrap with `ray.dag` `.experimental_compile()` to cut the
  ~1 ms/task launch overhead to <50 µs and get deadlock-free NCCL overlap.

**Recommendation.** Treat as a spike: prototype on a 2-GPU node, measure the eliminated
GPU↔CPU round-trip vs the Flight path, confirm bit-identical results, then decide on the
invariant amendment. Do **not** land the tensor path without that.

## Proposal 3 (5d): detached / named session fleet for cross-driver reuse

**Observation.** The warm session fleet (`dist/fleet/_fleet.py`) uses plain
driver-tied actors (no `lifetime="detached"` / `name=` / `get_if_exists`), so it cannot
survive a driver restart or be shared across drivers.

**Trade-off (why deferred).** Going detached enables cross-driver reuse but:
- adds **leak risk** — detached actors outlive the driver and need explicit, reliable
  teardown (a crashed driver leaves a running fleet + placement group);
- conflicts with the operational constraint that the Anyscale head must not be
  `ray stop`-ed (recovery is via `restart_ray.sh`), so an orphaned fleet is costly;
- needs get-or-create + version-fencing so two drivers don't borrow a fleet spawned
  with an incompatible engine config.

**Recommendation.** Keep the current per-driver warm fleet (already a ~3× win on warm
queries) unless cross-driver sharing becomes a real requirement. If pursued, it needs a
detached-actor lifecycle owner with a guaranteed cleanup path and config-version
fencing — its own design pass.

## Summary

| Item | Invariant-safe? | Blocker | Ship when |
|------|-----------------|---------|-----------|
| 3a STRICT_PACK collective placement | Yes (placement only) | public API + multi-GPU validation | after cluster test |
| 3b RDT/aDAG tensor transport | **No** — bends #3 | architecture amendment + Ray ≥2.44 + mutability | after RFC accepted + spike |
| 5d detached fleet | Yes | leak-lifecycle + ops constraint | only if cross-driver reuse needed |
