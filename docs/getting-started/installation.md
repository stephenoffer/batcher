# Installation

This page describes how to install Batcher, which extras you need for cloud storage,
distribution, and ML backends, and how to build the engine from source.

On PyPI the package is `batcher-engine`, and in your code it imports as `batcher`. The
bare `batcher` name on PyPI belongs to an unrelated project. Everything the core engine
needs ships in that one wheel: the Python control plane and the compiled Rust data
plane. Distribution, file formats, and ML backends arrive through extras.

## Requirements

- Python 3.10 or newer.
- A 64-bit platform (Linux, macOS, or Windows).

Releases ship the compiled engine as a native extension module, so you don't need a
Rust toolchain to install one.

## Install the core engine

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

## Optional extras

Extras install with the usual `pip install "batcher-engine[extra]"` syntax. None of
them change the core API.

- `ray`: distributed execution and scheduling. Ray schedules tasks and actors; bulk
  data moves over Arrow Flight, never through the Ray object store.
- `cloud`: object-store filesystems (`s3://`, `gs://`, and similar) via fsspec.
- `torch`, `tensorflow`: ML data-plane backends for the `.ml` accessor.
- `duckdb`, `polars`: the reference backends the benchmark and differential test
  suites compare against. They are not part of the core engine.

Install several at once:

```bash
pip install "batcher-engine[ray,cloud]"
```

## Install an unreleased revision

Installing straight from the git repository builds the Rust engine, so you need a
[Rust toolchain](https://rustup.rs):

```bash
pip install "git+https://github.com/stephenoffer/batcher.git"
```

## Build from source

Working on the engine itself means building the Rust crates into your virtual
environment. The `just` recipes wrap the exact build commands:

```bash
just build          # maturin develop - compile the engine into the venv
just build-release  # optimized build
```

Once built, `import batcher` resolves the locally compiled engine.

## Next steps

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {octicon}`rocket;1.1em` Quickstart
:link: quickstart
:link-type: doc
Run a query end to end in a couple of minutes.
:::

:::{grid-item-card} {octicon}`light-bulb;1.1em` Core concepts
:link: concepts/index
:link-type: doc
The lazy, immutable Dataset model that shapes the API.
:::
::::
