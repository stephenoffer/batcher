# Diagnose a slow GPU stage

This page covers finding out why a GPU stage was slower than it should have been, when the
answer is not in the plan. It assumes the query is correct and the plan is the one you wanted.

The hard part is that a GPU fails slowly rather than loudly. A device clamped by its own
thermals still returns the right answer. So does one whose host link trained at half width, one
decoding video on its shader cores instead of its decoder, and one waiting on the stage in
front of it. Every one of those leaves the query correct and the node a fraction as fast, and
none of them appears in a stack trace or a query profile.

## Why one utilization number is not enough

Almost every GPU tool reports the same figure, and it answers less than it appears to.
`sm_utilization` is a *duty cycle*: the fraction of the sampling period during which at least
one kernel was resident. A kernel occupying one streaming multiprocessor of 132 reads as 100
percent busy. So does a kernel occupying all of them.

That means a device at 95 percent might be saturated, or might be running a badly shaped kernel
that leaves most of the hardware idle. Both look identical, and they differ by two orders of
magnitude in throughput.

There is a second problem, and it catches more people than the first. A stage sampled once when
it finishes is sampled at the moment its work drained, so it reports an idle device. Sampled
once at the start, it reports the moment before the work began. Neither describes the stage.

Batcher's answer to both is to sample devices across a window and classify the distribution.

## Turn on sampling

Sampling runs a daemon thread that reads each device on an interval and folds the readings into
a bounded rolling summary. It is off by default, because a driver round trip per device per
interval is real cost in a short-lived worker, and because the figure it produces is only worth
that cost to somebody reading it.

To sample a run and read the diagnosis, complete the following steps:

1. Enable `accelerator.telemetry_sampling`, or start the sampler directly.
1. Run the query you want to characterize.
1. Read the report.

```python
from batcher.observe.accelerators import start_device_series, stop_device_series

start_device_series()
# ... run your query here ...
stop_device_series()
```

The accumulated window survives the stop, so you read it afterwards rather than racing the last
sample. Memory is bounded by the number of devices, not by how long the sampler ran: it keeps
running aggregates rather than samples, so leaving it on for a twelve-hour job costs what one
second costs.

```python
from batcher.observe.accelerators import format_bottleneck_report

print(isinstance(format_bottleneck_report(), str))
# True
```

On a host with no devices, or with sampling never started, the report says so in one line rather
than describing a healthy fleet. That distinction is the point: a confident wrong diagnosis
costs more than no diagnosis, because somebody acts on it.

## Read the verdict

Each device gets one verdict and one thing to change. The verdicts and what each one means:

| Verdict | What the window showed | What to change |
|---|---|---|
| `compute_bound` | SMs busy, memory and bus quiet | Nothing here. Add devices, or reduce work per row |
| `memory_bound` | Memory interface busy, SMs waiting on it | A smaller working set, or fewer passes over it |
| `transfer_bound` | The host link saturated while the SMs were not | Keep data device-resident, or stage from pinned memory |
| `starved` | Everything quiet, or swinging between busy and idle | Deepen the prefetch, or raise the in-flight batch count |
| `throttled` | The driver clamped the clocks | Cooling, or the enforced power limit |
| `contended` | Another process was doing work on the device | Size fractionally, or place the work elsewhere |
| `occupancy_limited` | SMs busy while holding few warps | The kernel's register or shared-memory footprint |
| `unknown` | Not enough signal to say | Sample for longer, or check driver visibility |

Two of these deserve attention because they are commonly misread.

`starved` is reported both for a device that was quiet throughout and for one whose utilization
swung between saturated and idle. The second case has a perfectly ordinary-looking mean, and a
tool reporting only the mean shows it as a device half-loaded. It is not half-loaded. It is
alternating between doing everything and doing nothing, and the fix is upstream.

`compute_bound` is the only verdict with nothing to fix, so the report never leads with it. A
fleet where seven devices are compute bound and one is throttled leads with the throttled one:
the seven are working as intended, and the eighth is quietly costing a third of a node.

## The signals behind the verdicts

Each verdict comes from readings you can also inspect directly. They are worth knowing about,
because each names a failure that nothing else reports.

**The host link.** A slot that trained to x8 on a x16 part, or to Gen3 on a Gen5 board, halves
or quarters every transfer without failing anything. This is the most common silent capacity
loss on rented GPU capacity.

```python
from batcher._internal.hardware.telemetry.throughput import device_throughput

print(isinstance(device_throughput(), tuple))
# True
```

Each record carries the negotiated generation and width against the maximum both ends support,
live transmit and receive rates, and the fraction of the link's capacity in use.

**Intermittent clamping.** Asking whether a device is throttled *right now* only finds a clamp
if you happen to ask during one. A device clamped for 30 percent of a stage looks unclamped on
70 percent of samples. The driver publishes cumulative counters instead, so two readings and a
subtraction give the fraction of an interval the device actually spent clamped. That is a
measurement rather than a sample.

**The fixed-function engines.** A datacenter GPU is not one processor. Beside the SMs sit
dedicated video decode, video encode, and JPEG decode blocks, each with its own utilization
counter, and none of them contributes to `sm_utilization`. A pipeline decoding H.264 on the
shader cores shows a busy GPU while the decoder that would have done the same work for free
sits at zero.

**Who else is on the device.** On a shared device, every device-level utilization figure is the
sum across tenants. Autobatching that reads it sees a neighbour's load as its own, backs off,
gets less of the device, and reads the same high number again. The loop is stable at the wrong
answer.

## Label the timeline for a profiler

When the verdict points at the kernels themselves, the next step is an external profiler. A
Nsight Systems or `rocprof` capture of a Batcher run is otherwise a wall of anonymous kernels:
the profiler sees the CUDA calls and has no idea which operator issued them, and the gap in the
middle of the timeline, which is the reason you opened the capture, is exactly the part it
cannot label.

Setting `accelerator.profiling` emits NVTX ranges around operators, so the capture reads as the
plan. On a ROCm build the same call emits ROCTX, so AMD needs no separate setting.

```python
from batcher._internal.instrument import profiling_enabled

print(profiling_enabled())
# False
```

The same setting switches stage timing from wall-clock to CUDA events. This matters more than
it sounds: a CUDA launch is asynchronous, so a `perf_counter` bracket around a kernel measures
the *launch*, usually a few microseconds, whatever the kernel then does for the next half
second. A profile built from such brackets shows every operator as fast and the total as slow,
with the missing time landing on whichever call happens to synchronize first.

Both are off by default. They are free when nothing is capturing, and a CUDA event pair per
range is not free, so the cost is only paid by somebody reading a profile.

## Two ways a device path silently is not one

Some slow stages are slow because an optimization that was requested did not happen. These are
worth checking explicitly, because in both cases the fallback is *slower* than the plain host
path it replaced and reports success either way.

**Compression the device cannot undo.** A device Parquet read decodes where the compute is,
which is the whole argument for it. That argument has an unstated precondition: the
decompression has to happen there too. Handed a codec it has no kernel for, the device reader
copies the pages to the host, decompresses them there, and copies them back. It crosses PCIe
twice instead of once and uses the host cores anyway.

Batcher checks the footer before choosing the device reader. Snappy, Zstd, and uncompressed
qualify; a codec outside that set keeps the host reader, which is both faster in that case and
better tested. The check reads through the same cache the row-group splitter uses, so on the
usual path it costs nothing, and a footer it cannot read leaves the decision alone rather than
disabling the device path on a guess.

**GPUDirect Storage that is not direct.** KvikIO has a fallback called compat mode, in which
every read is an ordinary host read into a bounce buffer followed by a copy to the device. It
engages when the `nvidia-fs` kernel module is missing, which is the normal state of a container
built without it. Nothing raises, and the read is slower than the plain host read because it
does the same work plus an extra buffer.

```python
from batcher.io.splits.kvikio import kvikio_status

print(isinstance(kvikio_status().direct, bool))
# True
```

`direct` is the only state in which a device-direct read is worth preferring. When it is false,
`reason` names why, which is what an operator can act on.

## Decode on the device to shrink the transfer

For a multimodal pipeline, the largest single lever is usually not the model. A 1440x1440 JPEG
is around 500 KB; decoded to RGB it is 6.2 MB. Decoding on the host and copying the result
moves the 6.2 MB. Copying the JPEG and decoding on the device moves the 500 KB.

```python
from batcher.ml.decode import transfer_saving_ratio

print(round(transfer_saving_ratio(1440, 1440, 500_000), 1))
# 12.4
```

Twelve times less traffic for the same pixels, and on a node where eight devices share one host
link that ratio frequently decides whether the stage is transfer-bound. The decode also stops
competing for host cores that are feeding the other seven devices.

Asking for a device decode and getting one are different events, though. A build of
`torchvision` without nvJPEG falls back to the CPU decoder silently, and identical pixels arrive
the slow way with nothing to say so.

```python
from batcher.ml.decode import hardware_decode_confirmed

print(hardware_decode_confirmed() in (True, False, None))
# True
```

`True` means the device's own decode counters registered work. `False` means they were readable
and idle, so the decode happened somewhere else. `None` means the part publishes no such
counters, which is not evidence either way.

## Alert on it across a fleet

Everything above is also exported as Prometheus series from the existing metrics endpoint, so a
fleet finds these conditions by scraping rather than by somebody opening a report. Link
throughput and derate, clock headroom, the codec engines, the driver's memory reserve, the
host-mappable aperture, and the integrated energy counter each get a series labelled by device.

Node-level conditions are exported as counts an alert can be written against directly:
`batcher_node_throttled_devices`, `batcher_node_transfer_bound_devices`,
`batcher_node_power_capped_devices`, `batcher_node_bar1_pressured_devices`, and
`batcher_node_clock_limited_devices`.

```python
from batcher.observe import prometheus_text

print("batcher_node_throttled_devices" in prometheus_text())
# True
```

A host that cannot read a condition exports it at zero rather than omitting it, so an alert
does not silently stop evaluating the day a container loses its driver mount.

## Requirements and limitations

- Live telemetry needs `pynvml` and a mounted driver. Without it the sampler collects nothing,
  the report says so, and every verdict is `unknown`.
- Real occupancy and tensor-core activity come from DCGM, which ships as a separate daemon and
  separate Python bindings that are not on PyPI. Without it, `occupancy_limited` is never
  reported and a badly shaped kernel reads as `compute_bound`.
- Per-process attribution needs the driver to see the process, which it cannot across a PID
  namespace boundary. Inside most containers, "who else is on this device" is unanswerable, and
  Batcher reports that rather than guessing.
- The integrated energy counter is Volta and later. Older parts fall back to sampled power, and
  the energy ledger records which it used.
- Verdicts describe the sampling window, not a specific query. Sampling around one query is the
  way to attribute them to it.

## See also

- {doc}`/user-guide/operate/gpu-fleets`: sizing, power budgets, health, and placement.
- {doc}`/user-guide/operate/observability`: the metrics endpoint these series join.
- {doc}`/user-guide/operate/performance`: the levers when the answer *is* in the plan.
- {doc}`/ml/inference/gpu`: choosing devices and batch sizes from the pipeline side.
- {doc}`/configuration/options`: every accelerator field with its default and unit.
