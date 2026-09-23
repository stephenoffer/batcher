# Install

This section covers installing Batcher: the package and its optional extras, and how to pick the install method that fits where Batcher will run, from a laptop to a container fleet to a cluster with no internet access.

Every method installs the same thing. Batcher is one Python package, `batcher-engine` on PyPI, with the compiled Rust engine inside it. The pip wheel, the container images, and a wheelhouse carried into an offline network all hold the same wheel, so a pipeline behaves the same whichever way it arrived. {doc}`packages-and-extras` covers the package itself, its optional extras, and how to confirm what you installed.

## Find your situation

The following table maps where Batcher runs to the method that suits it and the page that walks through it. Rows run from the most common case to the most specialized.

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

Batcher publishes a prebuilt wheel for every platform that PyArrow, its columnar foundation, publishes one for. Batcher can't install where PyArrow can't, so this list and PyArrow's are the same. The following table lists them by operating system:

| Operating system | Architecture | Covers |
|---|---|---|
| Linux with glibc | x86_64 | Most distributions, cloud VMs, and Debian- or Red Hat-based images |
| Linux with glibc | aarch64 | Arm servers such as AWS Graviton, Google Axion, and Azure Cobalt, and Arm laptops running Linux |
| Linux with musl | x86_64, aarch64 | Alpine and other musl-based container images |
| macOS | arm64 | Apple silicon |
| macOS | x86_64 | Intel Macs |
| Windows | x86_64 | 64-bit Windows |

Every platform needs Python 3.11 or newer, and one wheel serves every Python version from 3.11 up. On an older Python, pip refuses the install rather than installing a package that can't import.

There is no wheel for Windows on Arm, ppc64le, s390x, or any 32-bit platform, because PyArrow publishes none. On a Linux platform outside the table you can still try the "Build from source" steps in {doc}`packages-and-extras`, provided PyArrow itself builds there.

On Linux x86_64 the engine is compiled for the `x86-64-v2` instruction set, which includes SSE4.2 and POPCNT. Every x86_64 cloud instance type supports it. Wider vector instructions such as AVX2 and AVX-512 aren't required: the engine detects them on each machine at run time and uses them where they exist, so one wheel runs correctly on a mixed fleet.

## Check your platform

To see which row of the table applies to a machine, run the following in the Python you plan to use:

```python
import platform
import sys
import sysconfig

print(sysconfig.get_platform(), platform.machine())
print(sys.version_info >= (3, 11), platform.libc_ver()[0] or "not glibc")
```

The first line names the operating system and architecture, such as `linux-x86_64` or `macosx-14.0-arm64`. The second line prints `True` when the Python version is supported, then `glibc` on a glibc Linux. On an Apple silicon Mac, `platform.machine()` should print `arm64`. If it prints `x86_64`, that Python runs under Rosetta 2 and installs the slower Intel wheel.

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
