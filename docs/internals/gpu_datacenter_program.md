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

Nineteen new modules, 4,521 lines, 165 public functions, classes, and properties:

| Layer | Module | What it answers |
|---|---|---|
| 0 | `_internal/device_specs/` | 31 device models x power, bandwidth, TFLOPS, fabric width, MIG, host link |
| 0 | `_internal/hardware/mig.py` | the published MIG profile families, per slice count |
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
- `core/gpu_transform.py::gpu_groupby_agg` is bracketed by the stage meter, which is what
  makes the measurement path load-bearing rather than documentation: it is the one point
  every GPU relational stage passes through, local dispatch and Ray worker alike.
- `api/session/accelerators.py::measure_energy` folds each *measured* stage into the hub on
  the way out, which is the conductor's half of the learning loop.

The learning loop, closed the way the architecture describes it: Core measures the stage, the
conductor folds it into the `MetadataHub`, and Kyber's `select_device_class` prefers the device
this fleet measured best over the one the datasheet rates highest. Three refusals keep it from
being worse than no loop — a modelled figure is never learned from (it is the datasheet
restated), an under-sampled bucket reads as unmeasured rather than as slow, and a partially
measured fleet falls back to the datasheet ordering entirely rather than ranking a measured
device against an unmeasured one.

Device names are resolved rather than matched: NVML reports `"NVIDIA H100 80GB HBM3"`, the
driver reports `"NVIDIA A100-SXM4-80GB"`, Ray reports a label, and none of them is a table key.
`resolve_device_name` anchors on the part token, which is what stops an H100 resolving to an
A100 through their shared `80G` — a 2x error in bandwidth, power, and tensor rate that nothing
downstream could have caught.

Coverage: 14 test modules, 220 tests, none requiring a GPU or the compiled engine. The
properties they pin are the conservative directions — unknown devices, absent telemetry,
unlabelled fleets, unregistered datasets — because those are the paths that fail silently.
`tests/integration/test_gpu_datacenter_loop.py` covers what the unit tests cannot: that the
device class Kyber picks is one Carbonite can price, that the fan-out admission allows is the
one the grant hands out, and that a distributed run's merged energy equals the single-node
figure.

Documentation: `docs/user-guide/gpu-fleets.md` (the walkthrough),
`docs/configuration/accelerator.md` (the field reference), a data-residency section in
`docs/api/governance.md`, and a GPU-fleet section in the `run-a-distributed-job` skill.

## The integration audit, and what it found

The program's own defect, found by auditing it rather than by a gate: **fourteen of eighteen
new entry points and four configuration flags had no production caller.** A device table
nothing reads is documentation, and `python/batcher/CLAUDE.md` names that failure explicitly
("no config flag with no current caller"). The pass that closed it:

| Was orphaned | Now called by |
|---|---|
| `select_device_class` | `dist`'s `recommend_accelerator_type`, which duplicated it and now delegates |
| MIG profiles | Kyber's packing fraction (`prefer_mig`), via a layer-0 move so no boundary is crossed |
| `schedulable_device_count` | Carbonite's GPU grant (`health.enabled`) |
| `validate_fleet_power` | the grant clamp itself, so the number and the counter-offer are one figure |
| `device_energy_advice` | `decide_gpu_backend`, which now refuses a device the copy would lose |
| `VramPool` | inference sizing, against free rather than nominal device memory |
| `rank_nodes_by_efficiency`, `devices_within_power_budget`, `permitted_nodes` | `plan_collective`, where the constraints compose |
| `power_zone_load` | `bt.accelerators()`, per zone, because a breaker is a zone's not a fleet's |
| `EnergyLedger.merge` | nested energy scopes |
| `carbon_intensity`, `pue`, `region`, `renewable_fraction` | `configured_grid`, one source for cost and carbon |
| `telemetry_interval_s` | the meter, which reuses a reading rather than hammering NVML per batch |
| `spill_tier` | **deleted** — no honest caller, so it is gone rather than kept as decoration |

Two findings worth naming separately. `select_device_class` and `recommend_accelerator_type`
were the *same decision implemented twice* across a layer boundary, which is the failure mode
the independence contract makes most likely; the live topology now supplies candidates and the
policy makes the choice. And `device_energy_advice` had a real bug: it charged the CPU path
only its memory bandwidth, so a compute-heavy row looked free there and the verdict said an
inference stage was not worth a GPU — exactly backwards.

The renderers (`format_energy_report`, `format_fleet_efficiency`, `energy_metrics`,
`format_device_table`) have no internal caller by design: their caller is the user, and the bar
they are held to instead is documented, rendered by Sphinx, and taught with an executed
example. `plan_collective`, `residency_report`, and `merge_ledgers` meet that bar too, but
their *scheduler-side* call sites are not wired — see the register below.

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
- **The learned loop is per device model, not per workload shape.** Efficiency is bucketed by
  device and by work kind; two very different pipelines on the same device share a bucket and
  average toward each other. Keying by plan signature as the cardinality learner does is the
  obvious next step.
- **No admission-path integration for power.** `validate_fleet_power` now computes the
  scheduling grant's clamp, but `CarboniteManager.validate` still budgets memory only: the
  physical plan does not carry the device model and count a plan-level power check needs.
- **Three scheduler call sites are unwired**, and blocked rather than forgotten:
  `plan_collective` (bundles), `residency_report`/`permitted_nodes` (node filtering), and
  `merge_ledgers` (folding worker ledgers) all belong in
  `dist/executors/ray_runtime/scheduling.py` and the `dist/gpu/` task path, which another
  session held under active edit throughout. Each is reachable, tested, and taught; none is
  yet consulted by the distributed executor itself.
- **The cluster hardware profile does not yet carry the device model.** `HardwareProfile`
  gained the field and `local()` populates it, but `cluster_hardware_profile()` lives in the
  same contested file, so on a distributed run the model-specific decisions (MIG packing, the
  transfer veto) see `""` and fall back to their prior behavior.
- **Device figures are unverified against hardware.** They are datasheet values. A wrong row
  produces a wrong ratio, not a wrong result, but it is worth checking a model against
  `nvidia-smi` before trusting a placement decision that turns on it.
