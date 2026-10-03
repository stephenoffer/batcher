# Install

Batcher is one Python package, `batcher-engine`, with the compiled Rust engine inside it. The pip wheel, the container images, and an offline wheelhouse all carry the same wheel, so a pipeline behaves the same however it arrived.

```python
import batcher as bt

print(bt.from_pydict({"x": [1, 2, 3]}).agg(total=bt.col("x").sum()).to_pydict())
# {'total': [6]}
```

:::{important}
No release has been published to PyPI or the container registry yet. Install from the repository, as {doc}`packages-and-extras` describes.
:::

## Find your situation

The following table maps where Batcher runs to the install method and the page that walks through it.

| Where Batcher runs | Install method | Page |
|---|---|---|
| A laptop or workstation on macOS, Linux, or Windows | `pip` or `uv` into a virtual environment | {doc}`python-environments` |
| A notebook such as Jupyter or VS Code | `%pip install` into the running kernel | {doc}`python-environments` |
| A conda, mamba, or pixi environment | Install from PyPI into that environment | {doc}`python-environments` |
| Docker, Kubernetes, or an Alpine-based image | The prebuilt image, or `pip` in your own image | {doc}`containers` |
| A cloud VM on x86 or Arm | `pip`, the same as a workstation | {doc}`clusters-and-servers` |
| A Ray cluster, including KubeRay | The `-ray` image on every node | {doc}`clusters-and-servers` |
| An on-premises or air-gapped network | A wheelhouse or an image archive | {doc}`clusters-and-servers` |
| A platform the table below doesn't list | Build the engine from source | {doc}`packages-and-extras` |

## Supported platforms

Batcher ships a prebuilt wheel for every platform PyArrow publishes one for:

| Operating system | Architecture | Covers |
|---|---|---|
| Linux with glibc | x86_64 | Most distributions, cloud VMs, and Debian- or Red Hat-based images |
| Linux with glibc | aarch64 | Arm servers such as AWS Graviton, Google Axion, and Azure Cobalt, and Arm laptops running Linux |
| Linux with musl | x86_64, aarch64 | Alpine and other musl-based container images |
| macOS | arm64 | Apple silicon |
| macOS | x86_64 | Intel Macs |
| Windows | x86_64 | 64-bit Windows |

Every platform needs Python 3.11 or newer, and one wheel serves every Python version from 3.11 up.

:::{dropdown} CPU instruction sets and unlisted platforms
On Linux x86_64 the engine targets `x86-64-v2` (SSE4.2 and POPCNT), which every x86_64 cloud instance supports. It detects AVX2 and AVX-512 at run time and uses them where they exist, so one wheel runs on a mixed fleet.

Windows on Arm, ppc64le, s390x, and 32-bit platforms have no PyArrow wheel and so no Batcher wheel. On another Linux platform, try the "Build from source" steps in {doc}`packages-and-extras`.
:::

## Check your platform

Run the following in the Python you plan to use:

```python
import platform
import sys
import sysconfig

print(sysconfig.get_platform(), platform.machine())
print(sys.version_info >= (3, 11), platform.libc_ver()[0] or "not glibc")
```

The first line names the platform, such as `linux-x86_64` or `macosx-14.0-arm64`. The second prints `True` on a supported Python. On Apple silicon, `platform.machine()` should print `arm64`; `x86_64` means that Python runs under Rosetta 2.

```{toctree}
:hidden:

packages-and-extras
python-environments
containers
clusters-and-servers
```

## See also

- {doc}`packages-and-extras`: the package, its optional extras, and confirming what you installed.
- {doc}`../quickstart`: a complete pipeline once Batcher is installed.
- {doc}`/user-guide/operate/running/troubleshooting`: what to check when an install or a first query fails.
