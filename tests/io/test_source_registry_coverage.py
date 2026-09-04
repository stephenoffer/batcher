"""Every registered source format is reached by some test.

The sinks already have this. `test_format_fidelity_matrix.py` holds each of the 26 registry
sinks in `LOCAL_FILE_SINKS` or in `NOT_LOCALLY_WRITABLE` with a reason, and fails on a sink
in neither -- so a new sink cannot be added without either being round-tripped or being
classified. The 69 *sources* had no equivalent, and that asymmetry is exactly how `delta_cdf`
came to be registered with no test reaching it at all: it worked, and nothing said otherwise.

This is the sources' half, and it is deliberately the weaker mechanical form. It asks whether
each registry name is *referenced* -- by its registry string or by its Source class -- not
whether it is exercised. A test that imports `KafkaSource` and never calls it satisfies this.
That is a real limit and it is written down rather than implied: the guard catches "nobody
added a test for this format", which is the failure that happened, and not "the test that
exists is shallow", which needs a reader.

Reaching by *class* and not only by registry string is the part that matters. Format tests
overwhelmingly import the Source class and construct it directly -- the `delta_cdf` test this
guard exists to have caught uses `DeltaChangeFeedSource` and never writes the string
`"delta_cdf"` -- so a string-only scan reports two thirds of the covered formats as missing
and is useless.
"""

from __future__ import annotations

import pathlib
import re

import pytest

from batcher.io.formats.base import SOURCES

pytestmark = pytest.mark.unit

_TESTS = pathlib.Path(__file__).resolve().parents[1]

#: Registered sources no test reaches, and why. Empty, and it should stay that way: a source
#: needing a live server is still *referenced* by the suite that skips against it, which is
#: what this guard asks for. An entry here means nobody has written even that.
NOT_REACHED: dict[str, str] = {}


#: This file is excluded from its own corpus, and that is not fastidiousness. Every registry
#: name it mentions -- in a `NOT_REACHED` entry, in the vacuity guard's probe string -- would
#: otherwise count as "reached", so the guard would report a format as covered on the strength
#: of being *named in the guard*. The vacuity guard found this itself: its invented name
#: `NotARegisteredFormatName` matched, because the line asserting it does not appear is in the
#: corpus being searched.
_SELF = pathlib.Path(__file__).name


def _test_corpus() -> str:
    return "\n".join(
        path.read_text()
        for path in sorted(_TESTS.rglob("*.py"))
        if path.name not in {"conftest.py", _SELF}
    )


def _reaches(corpus: str, name: str, class_name: str) -> bool:
    """Whether the corpus mentions this format by registry name or by its Source class."""
    return bool(re.search(rf'["\']{re.escape(name)}["\']|\b{re.escape(class_name)}\b', corpus))


def _registered() -> dict[str, str]:
    return {
        name: getattr(SOURCES.get(name), "__name__", str(SOURCES.get(name)))
        for name in SOURCES.names()
    }


def test_every_registered_source_is_reached_by_some_test():
    corpus = _test_corpus()
    registered = _registered()
    unreached = sorted(
        name
        for name, class_name in registered.items()
        if name not in NOT_REACHED and not _reaches(corpus, name, class_name)
    )
    assert not unreached, (
        f"{len(unreached)} registered source format(s) are reached by no test: {unreached}. "
        "Add one, or record the name in `NOT_REACHED` with the reason -- a format in the "
        "registry and in no test is one a user can reach and nobody has run"
    )


def test_the_not_reached_table_names_real_sources():
    """A stale exemption is precedent for adding another."""
    registered = _registered()
    unknown = sorted(set(NOT_REACHED) - set(registered))
    assert not unknown, f"`NOT_REACHED` names {unknown}, which are not registered sources"
    for name, reason in NOT_REACHED.items():
        assert reason.strip(), f"`NOT_REACHED[{name!r}]` needs a reason"


def test_the_scan_reads_the_registry_and_the_suite():
    """Guard against a vacuous suite.

    The assertion above is "this list is empty", which an empty registry, an empty corpus, or
    a regex matching everything all satisfy. Pin all three: the registry is large, the corpus
    is large, and a name that genuinely appears nowhere is reported as unreached.
    """
    registered = _registered()
    assert len(registered) >= 60, f"only {len(registered)} sources registered; import missed"
    assert "parquet" in registered and "kafka" in registered

    corpus = _test_corpus()
    assert len(corpus) > 1_000_000, "the test corpus scan is not reading the suite"

    invented = "NotARegisteredFormatName"
    assert not _reaches(corpus, invented, invented), (
        "the matcher claims to find a format name that does not exist, so it would never "
        "report anything as unreached"
    )
    assert _reaches(corpus, "parquet", "ParquetSource"), "the matcher finds nothing real"
