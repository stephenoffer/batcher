"""While a query runs, the line has to say what is running.

The engine executes inside Rust and reports nothing until it returns, and every operator
stage the profile carries is replayed onto the bus *after* the query ends. So between
`QUERY_START` and `QUERY_END` the bus carried nothing at all, and the live progress line
showed the placeholder ``running`` from the first millisecond to the last. Measured on a
six-operator join over three million rows that was 122 ms of a 142 ms query; on a
distributed run the line sat on ``admission`` -- a phase that had taken 0.3 ms -- for the
whole seven seconds.

The control plane knows exactly which phase it is in, because it already times every one of
them for the DEBUG log. `PHASE` carries that to anything watching, which is the only thing
that can be reported *while* the slow part is happening rather than after it.
"""

from __future__ import annotations

import io
import re

import pytest

import batcher as bt
from batcher._internal import events
from batcher.api.orchestration import phases
from batcher.observe import ConsoleReporter
from batcher.observe.console.paint import STAGE_W

pytestmark = pytest.mark.unit


class _Tty(io.StringIO):
    """A stream the reporter will draw a live bar into."""

    encoding = "utf-8"

    def isatty(self) -> bool:
        return True


def _stages_drawn(work) -> list[str]:
    """The distinct stage labels the live line showed while `work` ran, in order."""
    stream = _Tty()
    detach = ConsoleReporter(stream=stream, live=True).attach()
    try:
        work()
    finally:
        detach()
    plain = re.sub(r"\x1b\[[0-9;?]*[a-zA-Z]", "", stream.getvalue())
    seen: list[str] = []
    for frame in plain.split("\r"):
        # The stage is the third column, between the label and the bar's left cap.
        match = re.search(r"^\s*\S\s+\S+\s+(.+?)\s+[▕[]", frame)
        if match and (label := match.group(1).strip()) not in seen:
            seen.append(label)
    return seen


@pytest.fixture
def query():
    """A plan big enough that the control plane spends measurable time in several phases."""
    rows = 200_000
    left = bt.from_pydict(
        {"k": [i % 500 for i in range(rows)], "v": [float(i) for i in range(rows)]}
    )
    right = bt.from_pydict({"k": list(range(500)), "w": [float(i) for i in range(500)]})
    return lambda: (
        left.filter(bt.col("v") > 5)
        .join(right, on="k")
        .group_by("k")
        .agg(t=bt.col("v").sum())
        .collect()
    )


def test_the_line_names_more_than_one_phase_while_the_query_runs(query):
    """The property that was missing: the line changes, and says what changed."""
    drawn = _stages_drawn(query)
    assert len(drawn) > 1, f"the live line never moved off its first stage: {drawn}"


def test_executing_is_one_of_them(query):
    """The phase that dominates has to be nameable, or the rest is decoration."""
    assert "executing" in _stages_drawn(query)


def test_the_placeholder_is_not_the_only_thing_shown(query):
    """`running` is the pre-first-phase default and must not be the whole story.

    The control for the two tests above: both would pass on a line that alternated between
    two equally uninformative words.
    """
    drawn = _stages_drawn(query)
    assert [d for d in drawn if d != "running"], f"only the placeholder was drawn: {drawn}"


def test_a_phase_event_carries_both_vocabularies():
    """The log's name is a field that gets grepped; the label is prose for a status line."""
    seen: list[events.Event] = []
    unsubscribe = events.subscribe(seen.append)
    try:
        phases.begin("kyber.optimize_full")
    finally:
        unsubscribe()
    assert len(seen) == 1
    assert seen[0].kind == events.PHASE
    assert seen[0].name == "optimizing"
    assert seen[0].fields["phase"] == "kyber.optimize_full"


def test_an_unmapped_phase_shows_its_machine_name_rather_than_nothing():
    """Worse than a label, never wrong -- and a new phase starts reporting without an edit."""
    seen: list[events.Event] = []
    unsubscribe = events.subscribe(seen.append)
    try:
        phases.begin("some.new.phase")
    finally:
        unsubscribe()
    assert seen[0].name == "some.new.phase"


@pytest.mark.parametrize("phase", sorted(phases.PHASE_LABELS))
def test_every_label_fits_the_column_it_is_drawn_in(phase):
    """A label read on every frame must not arrive as ``executing (cl…``."""
    label = phases.PHASE_LABELS[phase]
    assert len(label) <= phases.MAX_LABEL
    assert phases.MAX_LABEL <= STAGE_W, "the cap drifted above the column it exists to fit"


def test_publishing_a_phase_costs_nothing_when_nobody_is_watching():
    """This runs on every query, so the guard is the whole reason it is affordable."""
    assert not events.listening()
    phases.begin("core.execute")  # must not raise, must not build an event


def test_a_phase_is_not_an_operator_stage():
    """They share a live line and nothing else.

    A stage is keyed by `op_id` and the dashboard files it into a per-operator timeline; a
    control-plane phase has no operator, so publishing one as a stage would collide with
    operator 0 and overwrite a real operator's row.
    """
    assert events.PHASE != events.STAGE_START
    seen: list[events.Event] = []
    unsubscribe = events.subscribe(seen.append)
    try:
        phases.begin("core.execute")
    finally:
        unsubscribe()
    assert "op_id" not in seen[0].fields
