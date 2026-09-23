# Install into a Python environment

This page shows how to install Batcher with the Python tools you already use on a laptop, a workstation, or a notebook server: pip, uv, Poetry, conda, mamba, pixi, and Jupyter.

Each tool installs the same `batcher-engine` wheel from PyPI, and none of them needs a Rust toolchain on a {doc}`supported platform <index>`. You need Python 3.11 or newer. Extras use the same bracket syntax everywhere, such as `batcher-engine[ray,cloud]`, and {doc}`packages-and-extras` lists what each extra turns on.

## pip

pip is the tool that ships with Python. Install into a virtual environment so Batcher's dependencies don't collide with system packages:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install batcher-engine
```

On Windows, activate the environment with `.venv\Scripts\activate` instead of the `source` line.

## uv

uv creates environments and resolves dependencies faster than pip, and it can download a suitable Python for you. In a uv project, add Batcher as a dependency:

```bash
uv add batcher-engine
```

To install into an environment outside a project, use uv's pip interface:

```bash
uv venv --python 3.12
uv pip install batcher-engine
```

## Poetry

In a Poetry project, add Batcher the same way. Quote the name when you add extras, so the shell doesn't expand the brackets:

```bash
poetry add "batcher-engine[cloud]"
```

## conda, mamba, and pixi

Batcher isn't published on conda-forge, so a conda-family environment installs it from PyPI. Install Python and Batcher's compiled dependencies from conda-forge first, then install Batcher with pip. pip then finds PyArrow and NumPy already satisfied and leaves the conda builds in place, rather than replacing them with its own:

```bash
conda create -n batcher -c conda-forge "python>=3.11" "pyarrow>=16" "numpy>=1.24"
conda activate batcher
pip install batcher-engine
```

mamba takes the same arguments, so replace `conda` with `mamba` in the first two commands.

pixi tracks PyPI dependencies alongside conda ones in the same manifest, so add Python from conda-forge and Batcher with `--pypi`:

```bash
pixi init batcher-project
cd batcher-project
pixi add "python>=3.11" "pyarrow>=16"
pixi add --pypi batcher-engine
```

## Jupyter and other notebooks

In a notebook, install into the kernel the notebook is running with the `%pip` magic, which targets that kernel's Python even when it differs from the one on your `PATH`:

```text
%pip install batcher-engine
```

Restart the kernel afterward, so the next `import batcher` loads the newly installed package. A hosted notebook service works the same way if its kernel runs Python 3.11 or newer on a supported platform.

## Check the install

Whichever tool you used, confirm the engine loads and runs a query in the same environment:

```python
import batcher as bt

print(bt.from_pydict({"x": [1, 2, 3]}).agg(total=bt.col("x").sum()).to_pydict())
# {'total': [6]}
```

`bt.versions()["engine_profile"]` should read `release` for an installed wheel. {doc}`packages-and-extras` describes the rest of what {py:func}`bt.versions() <batcher.versions>` reports.

## Troubleshooting

The following table lists the install failures these tools report most often, with the cause and the fix:

| Symptom | Cause | Fix |
|---|---|---|
| `No matching distribution found for batcher-engine` | Python older than 3.11, or a platform with no wheel | Check with the snippet in {doc}`index`, then use Python 3.11 or newer |
| pip starts compiling Rust, or asks for `cargo` | No wheel matched, so pip fell back to the source package | Same as above. On an unsupported platform, follow the "Build from source" steps in {doc}`packages-and-extras` |
| `pip install batcher` installs something unrelated | The bare `batcher` name on PyPI belongs to another project | Install `batcher-engine` |
| `MissingDependencyError` when you call a feature | That feature's extra isn't installed | Install the extra named in the error message |

## See also

- {doc}`index`: supported platforms and the other install methods.
- {doc}`containers`: running Batcher in Docker and Kubernetes.
- {doc}`../quickstart`: a complete pipeline in five minutes.
