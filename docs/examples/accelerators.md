# Accelerators

This page covers the scripts that use a GPU when one is present and the CPU engine when it is
not, and the ones that verify the two agree.

## Why these scripts run without a GPU

The device tier is the one tier that cannot share the Rust `Expr` the other tiers consume,
because cuDF has no Rust binding. It is a translator: a second statement of the engine's
semantics against another library. That is what makes verifying it against the CPU oracle
necessary rather than optional.

CI has no GPU, so a suite whose accelerator scripts skipped themselves would check nothing.
Every script here takes a device flag instead and falls back:

```bash
python examples/gpu/device_selection.py               # auto
python examples/gpu/device_selection.py --device cpu  # forced
python examples/gpu/device_selection.py --device gpu  # errors with no accelerator
```

Asking for `--device gpu` where there is none is an error rather than a silent downgrade,
because that is the one time you typed it deliberately.

## The contract

The device changes where a plan runs, never what it computes. Same rows, same column names,
same column types. Anything outside the translated subset is declined with a reason and the
stage runs on the CPU engine, so `backend="gpu"` stays safe.

```python
import batcher as bt
from batcher import col

lineitem = bt.from_pydict(
    {"l_shipmode": ["AIR", "SHIP", "AIR"], "l_quantity": [17, 36, 8]}
)

query = (
    lineitem.filter(col("l_quantity") > 10)
    .group_by("l_shipmode")
    .agg(lines=bt.count())
    .sort("l_shipmode")
)

# `backend="auto"` uses a device when one is visible. The comparison is the contract.
on_device = query.collect(backend="auto")
on_cpu = query.collect(backend="cpu")

assert on_device.schema == on_cpu.schema
assert on_device.to_pydict() == on_cpu.to_pydict()
```

Compare the schema before the values. The two defects this tier has actually shipped were
both *type* bugs with correct values: a DATE column returning a timestamp on a real device,
and an integer `abs` widening to double. A value-only comparison would have passed both, which
is why `examples/gpu/shadow_verification.py` and `examples/gpu/cpu_gpu_parity_matrix.py`
check names and types first.

A green run of these scripts on a CPU-only machine says the harness works. It does not say
the device agrees, and only a recorded run with `distributed.gpu_shadow_verify=True` on real
hardware does.

## Every script on this page

The table below lists the accelerator scripts in path order.

<!-- library-table: gpu -->
| Script | Shows |
| --- | --- |
| `examples/gpu/backend_fallback.py` | What happens when the device tier cannot run part of a plan |
| `examples/gpu/batch_sizing.py` | Sizing a batch for the device you actually have |
| `examples/gpu/cpu_gpu_parity_matrix.py` | A parity matrix: every operator shape, on both tiers |
| `examples/gpu/device_selection.py` | Choosing a device, and running the same query either way |
| `examples/gpu/device_sizing.py` | What the engine can see about the accelerators on this machine |
| `examples/gpu/mixed_device_pipeline.py` | A pipeline where some stages run on a device and some do not |
| `examples/gpu/shadow_verification.py` | Verifying a device result against the CPU engine |
| `examples/gpu/torch_inference.py` | Batch inference with a torch model, on whatever device is available |
<!-- /library-table -->
