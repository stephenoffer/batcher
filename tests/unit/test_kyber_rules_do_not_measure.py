"""A Kyber rule may read learned state. It may not write a measurement.

`CLAUDE.md`: "**Core measures, Kyber decides, Carbonite protects.** A Kyber pass that
collects runtime metadata, or a Core path that makes an optimization decision, compiles and
passes tests while quietly corrupting the feedback loop that makes plans improve across runs."

That is on the silent-failure list, and the two halves are not equally defended. Core making
an optimization decision would mean `core` importing `kyber`, which `lint-layers` refuses
outright. The other half has no gate: a rule that writes to the `MetadataHub` while deciding
breaks nothing, passes everything, and poisons the loop -- because what Kyber then reads back
on the next run is not what execution measured, it is what optimization guessed. The estimate
becomes self-confirming, and the more the query runs the more confident the wrong number gets.

The line is not "kyber never writes". `kyber/learning.py` and `kyber/learned_tuning/` are the
cross-query learned store -- `CLAUDE.md` calls them the moat -- and writing there is their
job. Measured: their only external caller is `dist/adaptive_sizing/sizing.py`, handing in an
`output_rows` the executor actually observed. The measurer supplies; the store receives; the
rules read. This pins the third of those.

Scoped to `kyber/rules/`, where the passes live, and to the metadata *write* surface only.
Reading a hub, an estimate or a learned parameter inside a rule is the entire point of the
learning loop and is untouched.
"""

from __future__ import annotations

import pathlib
import re

import pytest

pytestmark = pytest.mark.unit

_PACKAGE = pathlib.Path(__file__).resolve().parents[2] / "python" / "batcher"
_RULES = _PACKAGE / "kyber" / "rules"

#: The metadata write surface. Reads (`get_*`, `estimate`, `lookup`) are deliberately absent:
#: a rule reading learned state is the learning loop working.
_WRITES = (
    "put_keyed_param",
    "put_keyed",
    "record_source_io",
    "record_udf_row_seconds",
    "record_smoothed_scalar",
    "record_exec_metrics",
    "record_execution",
    "record_selectivity",
    "record_column_stats",
    "record_column_row_bytes",
)

_CALL = re.compile(r"\b(" + "|".join(_WRITES) + r")\s*\(")


def _writers_under(root: pathlib.Path) -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for path in sorted(root.rglob("*.py")):
        hits = sorted(set(_CALL.findall(path.read_text())))
        if hits:
            found[str(path.relative_to(_PACKAGE))] = hits
    return found


def test_no_optimizer_rule_writes_a_measurement():
    offenders = _writers_under(_RULES)
    assert not offenders, (
        f"these Kyber rules write to the metadata store: {offenders}. A pass that records "
        "what it *decided* makes the next run read optimization's guess as though it were "
        "execution's measurement -- the estimate becomes self-confirming and gets more "
        "confident the more the query runs. Measure in `core`/`dist`; decide here"
    )


def test_the_learned_store_is_still_allowed_to_write():
    """The other side of the line, so this file cannot be read as "kyber never writes".

    If `kyber/learning.py` ever stops writing, the cross-query loop has been disconnected --
    which is a much bigger problem than a rule writing, and would otherwise look like this
    test simply getting greener.
    """
    learning = (_PACKAGE / "kyber" / "learning.py").read_text()
    assert _CALL.search(learning), (
        "`kyber/learning.py` no longer writes to the metadata store, so the cross-query "
        "learned-stats loop has nothing to persist -- plans can no longer improve across runs"
    )


def test_the_scan_would_see_a_writer():
    """Guard against a vacuous suite.

    The assertion above is "this dict is empty", which an unreadable directory, a wrong path,
    or a regex that matches nothing satisfies. Pin that the scan walks a real rule tree and
    that the pattern matches the real spelling, by finding the writes that legitimately exist
    one level up in `kyber/`.
    """
    rule_files = list(_RULES.rglob("*.py"))
    assert len(rule_files) > 50, f"only {len(rule_files)} rule modules found; wrong path?"

    elsewhere_in_kyber = _writers_under(_PACKAGE / "kyber")
    assert elsewhere_in_kyber, (
        "the write-surface pattern matched nothing anywhere in `kyber/`, including "
        "`learning.py` -- the names in `_WRITES` have been renamed and this file is checking "
        "for calls that no longer exist"
    )
