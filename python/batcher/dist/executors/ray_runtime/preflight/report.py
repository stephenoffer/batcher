"""The facts a worker must share with the driver to load its engine, and the comparison.

Pure: nothing here talks to Ray. `facts_on_this_node` is the probe body, written to run on a
worker with the standard library alone, and `compare` turns the driver's facts and each
worker's into findings. `run` is the half that schedules the probe and enforces the verdict.

What is compared, and why each one is (or is not) a reason to refuse:

* **Operating system, architecture and C library family.** The shipped `_native` extension is
  the driver's machine code. A worker on another OS, another ISA or musl instead of glibc
  cannot `dlopen` it, and the import fails inside the first task, or crashes the worker.
* **The glibc version, against the engine's own floor.** Not against the driver's glibc: a
  driver on a newer distribution than its workers is fine as long as the extension does not
  reference a symbol version the workers lack. The floor is read from the extension itself
  (`engine_glibc_floor`), so the check is the loader's own rule rather than a guess.
* **Python major.minor and implementation.** The extension is abi3, but the shipped pure-Python
  package and every pickled function are not portable across interpreter minors.
* **Dependencies.** The upload carries Batcher and not what it imports, so a worker missing
  PyArrow or NumPy fails on its first task (`REQUIRED_ON_WORKER`). A *different version* of one
  is reported but never refused: the Arrow C Data Interface the engine crosses is ABI-stable
  across releases.
"""

from __future__ import annotations

import mmap
import pathlib
import re
from collections.abc import Mapping
from dataclasses import dataclass, field

__all__ = [
    "CompatibilityReport",
    "Finding",
    "PlatformFacts",
    "compare",
    "engine_glibc_floor",
    "facts_on_this_node",
]

#: The distribution the driver's own package is installed as. Compared only when the workers
#: run their own build (a trusted image); when the driver ships its package, a worker's
#: installed copy is shadowed and its version is irrelevant.
ENGINE_DIST = "batcher-engine"

#: The dependencies whose absence on a worker is refused rather than reported. `import
#: batcher` reaches both at module level, so a worker without either fails its first task.
#: The other declared requirements are reported when missing: `psutil` degrades to `sysconf`,
#: and the rest are reached only by features a worker may never run.
REQUIRED_ON_WORKER = ("pyarrow", "numpy")


def facts_on_this_node(packages: tuple[str, ...]) -> dict:
    """The probe body: this interpreter's platform facts, from the standard library alone.

    Runs on a worker, so it MUST NOT import Batcher, the engine, or any dependency: the point
    is to learn whether the engine can be loaded *before* anything tries. Package versions are
    read from installed metadata, which imports nothing. Every import is local so a by-value
    copy of this function carries no reference to the module it was written in.

    Args:
        packages: Distribution names whose installed version to report.

    Returns:
        A plain dict, so the answer crosses any Python version without a shared class.
    """
    import platform
    import sys
    from importlib import metadata

    versions = {}
    for name in packages:
        try:
            versions[name] = metadata.version(name)
        except Exception:  # not installed, or unreadable metadata: both mean "absent"
            versions[name] = ""
    libc, libc_version = platform.libc_ver()
    return {
        "os": sys.platform,
        "machine": platform.machine(),
        "libc": libc,
        "libc_version": libc_version,
        "python": f"{sys.version_info[0]}.{sys.version_info[1]}",
        "implementation": sys.implementation.name,
        "packages": versions,
    }


#: Spellings of one ISA that differ only by vendor convention (`uname`, Windows, Apple).
_MACHINE_ALIASES = {"amd64": "x86_64", "x64": "x86_64", "arm64": "aarch64"}


@dataclass(frozen=True)
class PlatformFacts:
    """One interpreter's binary-compatibility facts, normalized for comparison."""

    os: str
    machine: str
    libc: str
    libc_version: str
    python: str
    implementation: str
    packages: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def from_probe(cls, raw: Mapping) -> PlatformFacts:
        """Build from `facts_on_this_node`'s dict, normalizing the architecture's spelling.

        Args:
            raw: The probe's answer.

        Returns:
            The normalized facts.
        """
        machine = str(raw.get("machine", "")).lower()
        return cls(
            os=str(raw.get("os", "")),
            machine=_MACHINE_ALIASES.get(machine, machine),
            libc=str(raw.get("libc", "")),
            libc_version=str(raw.get("libc_version", "")),
            python=str(raw.get("python", "")),
            implementation=str(raw.get("implementation", "")),
            packages=dict(raw.get("packages") or {}),
        )


@dataclass(frozen=True)
class Finding:
    """One field on which a worker node differs from the driver."""

    node_id: str
    node: str
    field: str
    driver: str
    worker: str
    blocking: bool

    def render(self) -> str:
        """One line naming the node, the field, and both sides."""
        return f"{self.node}: {self.field} is {self.worker!r} (driver: {self.driver!r})"


@dataclass(frozen=True)
class CompatibilityReport:
    """What the preflight learned about the workers, against the driver.

    `ships_driver_build` records which contract applied: when True the driver's own build was
    being shipped and any blocking finding refuses the query; when False the workers run their
    own image and every finding is advisory.
    """

    driver: PlatformFacts
    ships_driver_build: bool
    findings: tuple[Finding, ...] = ()
    probed: tuple[str, ...] = ()
    unanswered: tuple[str, ...] = ()

    @property
    def blocking(self) -> tuple[Finding, ...]:
        """The findings that make the driver's build unloadable on a worker."""
        return tuple(f for f in self.findings if f.blocking)

    def render(self) -> str:
        """The report as text: every finding, then the nodes that did not answer."""
        lines = [f.render() for f in self.findings]
        if self.unanswered:
            lines.append(f"not verified (no answer): {', '.join(self.unanswered)}")
        return "\n".join(f"  - {line}" for line in lines)


def _version_tuple(text: str) -> tuple[int, ...]:
    return tuple(int(p) for p in re.findall(r"\d+", text))


def _glibc_finding(
    driver: PlatformFacts, worker: PlatformFacts, floor: tuple[int, ...] | None, enforce: bool
) -> tuple[str, str, str, bool] | None:
    """The glibc-version finding, judged against the engine's floor when it is known."""
    if driver.libc != "glibc" or worker.libc != "glibc":
        return None
    if floor:
        have = _version_tuple(worker.libc_version)
        if have and have < floor:
            need = ">= " + ".".join(map(str, floor))
            return ("glibc version", f"{need} (the engine's floor)", worker.libc_version, enforce)
        return None
    if worker.libc_version != driver.libc_version:
        return ("glibc version", driver.libc_version, worker.libc_version, False)
    return None


def _node_findings(
    driver: PlatformFacts,
    worker: PlatformFacts,
    *,
    glibc_floor: tuple[int, ...] | None,
    enforce: bool,
) -> list[tuple[str, str, str, bool]]:
    """`(field, driver, worker, blocking)` for each difference between two nodes."""
    out: list[tuple[str, str, str, bool]] = []
    for name, ours, theirs in (
        ("operating system", driver.os, worker.os),
        ("architecture", driver.machine, worker.machine),
        ("C library", driver.libc or "not glibc", worker.libc or "not glibc"),
        ("Python version", driver.python, worker.python),
        ("Python implementation", driver.implementation, worker.implementation),
    ):
        if ours != theirs:
            out.append((name, ours, theirs, enforce))
    glibc = _glibc_finding(driver, worker, glibc_floor, enforce)
    if glibc is not None:
        out.append(glibc)
    for dist, ours in driver.packages.items():
        if dist == ENGINE_DIST and enforce:
            continue  # the shipped package shadows whatever the worker has installed
        theirs = worker.packages.get(dist, "")
        if not ours or ours == theirs:
            continue
        refused = enforce and not theirs and dist in REQUIRED_ON_WORKER
        out.append((f"package {dist}", ours, theirs or "not installed", refused))
    return out


def compare(
    driver: PlatformFacts,
    workers: Mapping[str, tuple[str, PlatformFacts]],
    *,
    ships_driver_build: bool,
    glibc_floor: tuple[int, ...] | None = None,
) -> tuple[Finding, ...]:
    """Every field on which a worker differs from the driver, marked blocking or advisory.

    Examples:
        .. doctest::

            >>> d = PlatformFacts("linux", "x86_64", "glibc", "2.35", "3.12", "cpython")
            >>> w = PlatformFacts("linux", "aarch64", "glibc", "2.35", "3.12", "cpython")
            >>> [f.field for f in compare(d, {"n1": ("arm-1", w)}, ships_driver_build=True)]
            ['architecture']

    Args:
        driver: The driver's facts.
        workers: Node id to `(label, facts)` for every node that answered.
        ships_driver_build: Whether the driver's build is what the workers will load. When
            False every finding is advisory: the workers bring their own build.
        glibc_floor: The highest glibc symbol version the engine references, if known.

    Returns:
        The findings, in node order.
    """
    return tuple(
        Finding(node_id, label, name, ours, theirs, blocking)
        for node_id, (label, facts) in workers.items()
        for name, ours, theirs, blocking in _node_findings(
            driver, facts, glibc_floor=glibc_floor, enforce=ships_driver_build
        )
    )


_GLIBC_SYMBOL = re.compile(rb"GLIBC_(\d+)\.(\d+)(?:\.(\d+))?")


def engine_glibc_floor(path: pathlib.Path) -> tuple[int, ...] | None:
    """The highest `GLIBC_x.y` symbol version an ELF object references, or `None`.

    That is the oldest glibc the dynamic loader will accept the object on: a worker below it
    fails the `dlopen` with "version `GLIBC_2.xx' not found". Scanned through `mmap`, so a
    large extension costs a page-cache pass rather than its size in heap.

    Args:
        path: The compiled extension.

    Returns:
        The version as a tuple, or `None` for a non-ELF file or one naming no glibc version.
    """
    try:
        with path.open("rb") as fh, mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            if mm[:4] != b"\x7fELF":
                return None
            found = {
                tuple(int(g) for g in m.groups() if g is not None)
                for m in _GLIBC_SYMBOL.finditer(mm)
            }
    except (OSError, ValueError):
        return None
    return max(found) if found else None
