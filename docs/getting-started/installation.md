# Installation

Batcher is distributed on PyPI as **`batcher-engine`** and imported as **`batcher`**
(the bare `batcher` name on PyPI belongs to an unrelated project). The core engine
(the Python control plane plus the compiled Rust data plane) is self-contained.
Optional features such as distribution, file formats, and ML backends are pulled in
through extras.

## Requirements

- Python 3.10 or newer.
- A 64-bit platform (Linux, macOS, or Windows).

The compiled engine is distributed as a native extension module
(`batcher._native`); no separate Rust toolchain is needed to install a release.

## Install the core engine

```
pip install batcher-engine
```

After it finishes, the import and a small query should work:

```python
import batcher as bt

ds = bt.from_pydict({"x": [1, 2, 3]})
print(ds.select(doubled=bt.col("x") * 2).to_pydict())
# {'doubled': [2, 4, 6]}
```

## Optional extras

Extras are installed with the usual `pip install "batcher-engine[extra]"` syntax.
They add capabilities without changing the core API.

- `ray` - distributed execution and scheduling. Ray is used for task and actor
  scheduling only; bulk data moves over Arrow Flight, not the Ray object store.
- `cloud` - object-store filesystems (`s3://`, `gs://`, and similar) via fsspec.
- `torch`, `tensorflow` - ML data-plane backends for the `.ml` accessor.
- `duckdb`, `polars` - reference backends used by the benchmark and differential
  test suites; not part of the core engine.

Install several at once:

```
pip install "batcher-engine[ray,cloud]"
```

## Install from GitHub

To install an unreleased revision straight from source (this builds the Rust engine,
so a [Rust toolchain](https://rustup.rs) is required):

```
pip install "git+https://github.com/stephenoffer/batcher.git"
```

## Building from source

If you are working on the engine itself, build the Rust crates into your virtual
environment. The project uses `just` recipes that wrap the exact build commands:

```
just build          # maturin develop - compile the engine into the venv
just build-release  # optimized build
```

Once built, `import batcher` resolves the local `batcher._native` extension.

## Next steps

- The [quickstart](quickstart.md) runs an end-to-end query.
- The [concepts](concepts/index.md) page explains the lazy, immutable Dataset model.
