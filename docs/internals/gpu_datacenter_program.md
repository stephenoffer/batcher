# The GPU-datacenter alignment program

A working record of the work that made Batcher schedulable inside a GPU datacenter rather than
merely capable of using a GPU. It is kept in-tree for contributors and is not a published page:
it names gaps and unverified areas that a user-facing page should not carry.

## The gap this closed

Batcher already ran work on accelerators. What it could not do was reason about the *fleet*
those accelerators sit in. Ray reports two facts about a device, its count and its model name,
and everything else a datacenter schedules on was unavailable to the control plane:

| Question | Before | Now |
|---|---|---|
| How much power does this device draw? | unavailable | `_internal.device_specs`, plus live NVML draw |
| What is this device doing right now? | unavailable | `_internal.hardware.nvml` |
| How wide is its coherent interconnect? | unavailable | `device_specs.nvlink_domain`, `fabric.topology` |
| Can it be partitioned? | unavailable | `carbonite.accel.mig` |
| How many watts may this job draw? | no such concept | `config.accelerator.energy`, `carbonite.accel.power` |
| What did this run cost in joules? | no such concept | `plan.energy`, `core.energy`, `observe.energy` |
| How many sequences fit on this device? | inferred from weights | `carbonite.accel.kv_cache` |
| Is this device healthy? | assumed yes | `carbonite.accel.health` |
| Where may this dataset be computed? | storage only | `governance.residency`, `fabric.residency` |

## The five design rules everything here follows

1. **Unknown stays unknown.** Every accessor reports `None`, `0`, or `-1` ("no opinion") for a
   device model, a grid, or a topology it cannot answer for. A fleet this build does not
   recognize keeps the behavior it had rather than being scheduled against fabricated figures.
2. **Every default is inert.** A power budget of zero is unbounded, a carbon intensity of zero
   reports no emissions, health checking is opt-in, and placement switches preserve the prior
   scheduler behavior. Each control can be adopted one at a time.
3. **Measured and modelled are distinguishable.** `StageEnergy.measured` and the report's
   "measured / modelled" line exist because an estimate presented as a measurement is the one
   error a cost report cannot recover from.
4. **The lanes hold.** Kyber decides (device class, fan-out, whether a device is worth the
   watts), Carbonite protects (VRAM pool, power admission, health verdicts), Core measures
   (the stage meter), `dist` schedules (fabric and residency placement), `observe` reports.
   The import-linter independence contract stayed green throughout, including one refactor
   where a health helper had to move from `ml` into Carbonite to keep it that way.
5. **Nothing is fabricated.** Device figures are vendor nameplate numbers on a stated basis
   (dense tensor path, no sparsity multiplier). There is deliberately no table of regional
   carbon intensities, no inferred region for a bucket name, and no NIC bandwidth constant.

## What landed

Nineteen new modules, 4,167 lines, 158 public functions, classes, and properties:

| Layer | Module | What it answers |
|---|---|---|
| 0 | `_internal/device_specs.py` | 31 device models x power, bandwidth, TFLOPS, fabric width, MIG |
| 0 | `_internal/hardware/nvml.py` | live draw, utilization, memory, ECC, throttle reasons |
| 1 | `plan/energy/{power,carbon,accounting}.py` | the power model, `GridProfile`, the mergeable `EnergyLedger` |
| 0 | `config/accelerator.py` | 23 tunables: budget, price, intensity, PUE, health, KV cache |
| 3 | `carbonite/accel/{vram,mig,kv_cache,health,power}.py` | device memory as a pool and its spill tier, partitioning, cache budgets, verdicts, admission |
| 3 | `kyber/gpu/energy.py` | device class, power-bounded fan-out, the roofline verdict |
| 3 | `governance/residency.py` | where a dataset may be computed, with advisory and strict modes |
| 3 | `core/energy.py` | the stage meter that records what was drawn |
| 2 | `observe/energy.py` | the per-stage table, per-device efficiency, `energy.*` metrics, the live device table |
| 4 | `dist/.../fabric/{topology,placement,residency}.py` | NVLink domains, racks, power zones, gang bundles, region filtering |
| 5 | `api/session/accelerators.py` | `bt.accelerators()` and `bt.show_accelerators()` |

Integration points, where the new facts change an existing decision:

- `carbonite/policies/scheduling.py` clamps the GPU grant to the power budget.
- `dist/.../scheduling.py` reports a collective wider than the fleet's fabric domain.
- `ml/llm/sizing.py::kv_cache_concurrency` sizes inference by cache rather than weights.
- `ml/devices.py::device_feed_advice` separates a starved pipeline from a saturated device.
- `_internal/accelerators.py` now reads device memory from the one table rather than a second.

Coverage: 13 test modules, 190 tests, none requiring a GPU or the compiled engine. The
properties they pin are the conservative directions — unknown devices, absent telemetry,
unlabelled fleets, unregistered datasets — because those are the paths that fail silently.
`tests/integration/test_gpu_datacenter_loop.py` covers what the unit tests cannot: that the
device class Kyber picks is one Carbonite can price, that the fan-out admission allows is the
one the grant hands out, and that a distributed run's merged energy equals the single-node
figure.

Documentation: `docs/user-guide/gpu-fleets.md` (the walkthrough),
`docs/configuration/accelerator.md` (the field reference), a data-residency section in
`docs/api/governance.md`, and a GPU-fleet section in the `run-a-distributed-job` skill.

## What this program did **not** do

Named explicitly, because the absence of each is a real limit and not an oversight:

- **No Rust data-plane work.** Nothing in `crates/` changed. GPUDirect/RDMA transport for the
  Flight shuffle, device-buffer zero-copy, and NCCL integration remain unaddressed; they are
  data-plane changes and belong with a measurement on real hardware.
- **No measurement on a GPU.** This work was written and tested on a CPU-only host, so every
  energy figure the tests exercise is modelled rather than measured, and no throughput,
  tokens-per-joule, or power claim appears anywhere in the tree. The benchmark script
  (`benchmarks/gpu_backend/energy_efficiency.py`) exists to produce those numbers on a real
  fleet; its results belong in `benchmarks/BENCHMARK_RESULTS.md` with the hardware named.
- **No learned energy statistics.** Kyber consumes device figures but does not yet record
  measured tokens-per-joule across runs the way it records cardinalities. The `EnergyLedger` is
  the shape that would feed it.
- **No admission-path integration for power.** `validate_fleet_power` returns a verdict, but
  `CarboniteManager.validate` still budgets memory only: the physical plan does not carry the
  device model and count a power check needs, and adding that is a contract change.
- **Device figures are unverified against hardware.** They are datasheet values. A wrong row
  produces a wrong ratio, not a wrong result, but it is worth checking a model against
  `nvidia-smi` before trusting a placement decision that turns on it.
