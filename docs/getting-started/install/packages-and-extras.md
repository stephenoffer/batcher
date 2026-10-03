# Installation

This page describes how to install Batcher, add the optional extras for clusters, cloud storage, table formats, and ML, and build the engine from source.

Batcher installs from one wheel. The distribution is named `batcher-engine` and imports as `batcher` (the bare `batcher` name on PyPI is an unrelated project). The wheel carries the Python control plane and the compiled Rust engine, so DataFrames, SQL, and Parquet work with nothing else installed.

Everything else is an optional extra layered on top of that wheel:

![A two-layer stack. The bottom layer is the core wheel, installed with pip install batcher-engine: the Python control plane with Dataset, SQL, and the optimizer, the precompiled Rust engine that needs no toolchain, and the required dependencies pyarrow, numpy, sqlglot, and psutil, on Python 3.11 or newer. An arrow labeled each plugs into the same API points up to the optional extras layer, grouped as clusters (ray), object stores (cloud), lakehouse (delta, iceberg, hudi), streaming (kafka, kinesis, pubsub, pulsar, eventhubs), media decode (image, audio, video), ML frameworks (torch, tensorflow, jax), LLMs and embeddings (st, vllm, sglang), and dataframes (pandas, polars), with the bundles lakehouse, streaming, multimodal, and all. A feature whose extra is missing raises MissingDependencyError.](/_static/diagrams/install_extras_stack.svg)

:::{important}
No `batcher-engine` release is on PyPI yet, so `pip install batcher-engine` reports "No matching distribution found". Install from the repository, which compiles the engine and needs a Rust toolchain 1.89 or newer. The `pip install batcher-engine` commands in these docs work once a release is tagged.
:::

## Requirements

You need Python 3.11 or newer on a 64-bit platform. The release workflow, `.github/workflows/release.yml`, builds wheels for Linux on x86_64 and aarch64, macOS on Intel and Apple silicon, and Windows. A released wheel ships the engine precompiled, so installing a release needs no Rust toolchain. Installing from the repository does.

{doc}`index` lists every supported platform, including musl-based Linux such as Alpine, and helps you choose between pip, uv, conda, containers, Ray clusters, and offline installs.

## Install the core engine

Install from the repository with pip. This compiles the Rust engine, so install a [Rust toolchain](https://rustup.rs) first:

```bash
pip install "git+https://github.com/stephenoffer/batcher.git"
```

Once a release is published, the prebuilt wheel installs without a toolchain:

```bash
pip install batcher-engine
```

Then check that the import and a small query work:

```python
import batcher as bt

ds = bt.from_pydict({"x": [1, 2, 3]})
print(ds.select(doubled=bt.col("x") * 2).to_pydict())
# {'doubled': [2, 4, 6]}
```


## Confirm what you installed

{py:func}`bt.versions() <batcher.versions>` returns the Batcher version, the compiled engine's version and build profile, and one key per optional backend:

```python
info = bt.versions()
print("batcher" in info, "engine" in info, "engine_profile" in info)
# True True True
print(isinstance(bt.__version__, str))
# True
```

Each optional backend key, such as `ray`, `torch`, `polars`, or `deltalake`, holds the installed version, or the string `"not installed"` when that package is absent. {py:func}`bt.show_versions() <batcher.show_versions>` prints the same information as a block, which is what to paste into a bug report:

```console
$ python -c "import batcher as bt; bt.show_versions()"
batcher        : 0.1.0
engine         : 0.1.0
engine_profile : release
python         : 3.13.15
...
```

## Optional extras

Extras add integrations without changing the core API. When a feature needs a package you haven't installed, Batcher raises `MissingDependencyError`, an `ImportError` whose message carries the exact install command:

```python
print(issubclass(bt.MissingDependencyError, ImportError))
# True
```

Install extras with the usual bracket syntax:

```bash
pip install "batcher-engine[ray,cloud,delta]"
```

Before a release exists, name the repository as the source of the same extras:

```bash
pip install "batcher-engine[ray,cloud,delta] @ git+https://github.com/stephenoffer/batcher.git"
```

The following table lists the extras most people reach for first. `pyproject.toml` declares the full set, and bundles such as `lakehouse`, `streaming`, `multimodal`, and `all` install a whole group at once.

| Extra | What it turns on | Read more |
|---|---|---|
| `ray` | Distributed execution on a Ray cluster | {doc}`/integrations/compute/ray` |
| `cloud` | Object stores: `s3://`, `gs://`, `az://`, and HTTP(S) URLs | {doc}`/user-guide/moving-data/cloud-storage` |
| `delta`, `iceberg`, `hudi` | Lakehouse table formats | {doc}`/integrations/lakehouse/index` |
| `kafka`, `kinesis`, `pubsub`, `pulsar`, `eventhubs` | Streaming connectors | {doc}`/integrations/streams/index` |
| `image`, `audio`, `video` | Decoding images, audio, and video | {doc}`/ml/index` |
| `torch`, `tensorflow`, `jax` | Handing batches to ML frameworks | {doc}`/integrations/compute/pytorch` |
| `st`, `vllm`, `sglang` | Text embeddings and LLM batch inference | {doc}`/ml/retrieval/index` |
| `pandas`, `polars`, `numpy` | Converting results with `to_pandas`, `to_polars`, and `to_numpy` | {doc}`/getting-started/migration/index` |

## Build from source

To work on the engine itself, clone the repository and compile the crates into your virtual environment. You need a Rust toolchain, [maturin](https://www.maturin.rs), and [just](https://just.systems). The `just` recipes wrap the build commands:

```bash
just build          # maturin develop: compile the engine into the venv
just build-release  # maturin develop --release: the optimized engine
```

Once built, `import batcher` resolves the locally compiled engine, and `bt.versions()["engine_profile"]` tells you whether you're running the debug or release build. Benchmark only a release build.

## See also

- {doc}`/getting-started/quickstart`: run a complete pipeline in five minutes.
- {doc}`/getting-started/concepts/index`: the lazy, immutable dataset model that shapes the API.
- {doc}`/configuration/index`: tuning memory, spill, and parallelism once you're running.
- {doc}`/user-guide/operate/running/troubleshooting`: what to check when an install or a first query fails.
