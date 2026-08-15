"""Whether a device read actually bypasses the host, or only reports that it did.

`gds` answers whether GPUDirect Storage *could* serve a path: the library is present, the file
is local, the filesystem supports the DMA path. All three can hold and the read still go through
the host, because the library that performs it has a fallback and takes it silently.

KvikIO calls that fallback **compat mode**. In it, every `CuFile` read is a POSIX read into a
host bounce buffer followed by a copy to the device — the exact path GDS exists to remove, with
the GDS API in front of it. It engages when cuFile fails to initialize, when the environment
asks for it, and, most often, when the container was built without the `nvidia-fs` kernel module
the DMA path needs. Nothing raises. The reader is slower than the plain host reader it replaced,
because it does the same work plus an extra buffer, and every log line says it used GDS.

That is the failure this module exists to catch. The distinction it draws:

* **Available** — KvikIO imports and can open files. Says nothing about how it will read them.
* **Compat mode** — it will read through the host. The device-direct argument does not apply,
  and a caller that took on a second Parquet implementation for it has taken on the edge cases
  without the benefit.
* **Direct** — reads reach the device by DMA. This is the only state in which a GDS read is
  worth preferring over the host reader.

**Reported, never worked around.** A compat-mode deployment is a deployment decision — a base
image, a mounted module, a filesystem — and silently routing around it would hide the thing an
operator needs to fix. So the answer here is a fact for a caller to act on, and the caller's
action is to use the host path, which works.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

from batcher.config.env import env_flag

__all__ = [
    "KvikioStatus",
    "kvikio_status",
    "reset_kvikio_probe",
]

#: The environment variable KvikIO reads to force compat mode. Checked directly as well as
#: through the library, because a caller wanting to explain *why* a fleet is in compat mode is
#: better served by "someone set this" than by "the library says so".
_COMPAT_ENV = "KVIKIO_COMPAT_MODE"

#: The predicates that report compat mode, across the spellings KvikIO has used. It became a
#: function, then a property, then `is_compat_mode_preferred`; a build carrying only one of them
#: would otherwise report *direct* by default, which is the wrong way round to be wrong.
_COMPAT_PREDICATES = ("is_compat_mode_preferred", "compat_mode")


@dataclass(frozen=True, slots=True)
class KvikioStatus:
    """What a device read on this host would actually do.

    Attributes:
        available: Whether KvikIO imported.
        compat_mode: Whether reads go through a host bounce buffer. `True` with `available`
            False as well, because a library that is not there certainly is not doing DMA —
            a caller checking one field gets the conservative answer.
        threads: KvikIO's configured reader thread count, `0` when unreported.
        reason: A short phrase naming why compat mode is on, `""` when it is not or when the
            library did not say. The one field an operator can act on.
    """

    available: bool = False
    compat_mode: bool = True
    threads: int = 0
    reason: str = ""

    @property
    def direct(self) -> bool:
        """Whether reads reach the device by DMA, with no host bounce buffer.

        The only state in which preferring a device Parquet read over the host reader is
        justified by the transfer argument. Everything else is a second implementation of
        Parquet taken on for nothing.
        """
        return self.available and not self.compat_mode


def _compat(kvikio) -> tuple[bool, str]:
    """`(in compat mode, why)` from the library, defaulting to compat when it will not say.

    Defaulting to compat rather than to direct is the whole safety property: being wrong toward
    compat costs a caller the host path it would have used anyway, and being wrong toward direct
    costs it a slower read it believes is faster.
    """
    if env_flag(_COMPAT_ENV):
        return (True, f"{_COMPAT_ENV} is set in the environment")
    defaults = getattr(kvikio, "defaults", None)
    if defaults is None:
        return (True, "the installed kvikio publishes no defaults module")
    for name in _COMPAT_PREDICATES:
        probe = getattr(defaults, name, None)
        if probe is None:
            continue
        try:
            value = probe() if callable(probe) else probe
        except Exception:
            return (True, f"kvikio.defaults.{name} could not be read")
        return (bool(value), "cuFile could not open the DMA path" if value else "")
    return (True, "the installed kvikio reports no compat-mode predicate")


def _threads(kvikio) -> int:
    """KvikIO's configured reader thread count, `0` when unreported."""
    defaults = getattr(kvikio, "defaults", None)
    probe = getattr(defaults, "num_threads", None) if defaults is not None else None
    if probe is None:
        return 0
    try:
        return int(probe() if callable(probe) else probe)
    except Exception:
        return 0


@functools.lru_cache(maxsize=1)
def kvikio_status() -> KvikioStatus:
    """What a device-direct read would do on this host.

    Memoized: it imports a library and reads its configuration, neither of which changes within
    a process, and the question sits in front of a per-split decision.

    Returns:
        The status. An unavailable library reports `compat_mode=True` and `direct=False`, so a
        caller testing one field never concludes it has DMA it does not have.
    """
    try:
        import kvikio
    except Exception:
        return KvikioStatus(available=False, compat_mode=True, reason="kvikio is not installed")
    compat, reason = _compat(kvikio)
    return KvikioStatus(
        available=True,
        compat_mode=compat,
        threads=_threads(kvikio),
        reason=reason,
    )


def reset_kvikio_probe() -> None:
    """Forget the memoized status so the next call re-reads the library and environment."""
    kvikio_status.cache_clear()
