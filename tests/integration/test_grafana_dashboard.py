"""The shipped Grafana dashboard may only reference metrics the exporter emits.

A dashboard is the one artifact that fails *quietly*: a renamed metric leaves a panel
rendering an empty axis, which looks exactly like a healthy idle system. Nothing in a
Grafana instance checks the queries against the producer, so this does.

The check runs in both directions. Every metric a panel names must exist, or the panel is
already broken; and every metric the exporter emits must appear somewhere in the dashboard,
or a signal the engine pays to produce is one nobody can see.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

import batcher as bt
from batcher.observe import prometheus_text

pytestmark = pytest.mark.integration

DASHBOARD = Path(__file__).resolve().parents[2] / "tools" / "grafana" / "batcher-overview.json"

# Prometheus appends these to a histogram's base name; a panel referencing
# `batcher_query_duration_ms_bucket` is referencing the `batcher_query_duration_ms` family.
_HISTOGRAM_SUFFIXES = ("_bucket", "_sum", "_count")

# Deliberately not on the dashboard: uptime is a liveness detail rather than a signal an
# operator acts on, and it would occupy a panel that a spill rate deserves.
_NOT_CHARTED = {"batcher_uptime_seconds"}


def _exported_metric_families() -> set[str]:
    """Every metric family name the Prometheus exporter renders, read from the exporter."""
    # A query makes the counters non-trivial; the HELP lines are emitted either way, and
    # reading the names from the real exposition is the whole point of the check.
    bt.from_pydict({"x": [1, 2, 3]}).agg(s=bt.col("x").sum()).collect()
    return set(re.findall(r"^# HELP (\S+)", prometheus_text(), re.M))


def _referenced_metrics() -> set[str]:
    """Every `batcher_*` metric named by any panel expression in the dashboard."""
    document = json.loads(DASHBOARD.read_text())
    expressions = [
        t["expr"] for panel in document["panels"] for t in panel.get("targets", []) if "expr" in t
    ]
    expressions += [
        v["query"] for v in document["templating"]["list"] if isinstance(v.get("query"), str)
    ]
    assert expressions, "no panel expressions found — the dashboard shape changed"
    names: set[str] = set()
    for expr in expressions:
        for name in re.findall(r"\bbatcher_[a-z0-9_]+\b", expr):
            for suffix in _HISTOGRAM_SUFFIXES:
                if name.endswith(suffix):
                    name = name[: -len(suffix)]
                    break
            names.add(name)
    return names


def test_every_charted_metric_is_actually_exported():
    """A panel naming a metric the exporter does not emit renders an empty axis forever."""
    referenced = _referenced_metrics()
    assert referenced, "extracted no metric names — the extraction regex is broken"
    missing = sorted(referenced - _exported_metric_families())
    assert not missing, f"dashboard charts metrics the exporter does not emit: {missing}"


def test_every_exported_metric_is_charted_somewhere():
    """A metric the engine pays to produce and no panel shows is a signal nobody reads."""
    unused = sorted(_exported_metric_families() - _referenced_metrics() - _NOT_CHARTED)
    assert not unused, f"exported but not on the dashboard: {unused}"


def test_panels_aggregate_across_instances():
    """Every expression must sum across processes, or the dashboard reads one worker.

    This is the property that makes one dashboard serve a single process and a Ray
    cluster. A panel that dropped the aggregation would still render — showing whichever
    instance Prometheus happened to return first — which is the failure this catches.
    """
    document = json.loads(DASHBOARD.read_text())
    offenders = []
    for panel in document["panels"]:
        for t in panel.get("targets", []):
            expr = t.get("expr", "")
            if "batcher_" not in expr:
                continue
            if not re.search(r"\b(sum|avg|max|min|count)\s*(by\s*\([^)]*\)\s*)?\(", expr):
                offenders.append((panel["title"], expr))
    assert not offenders, f"panels that do not aggregate across instances: {offenders}"


def test_every_panel_selects_on_the_template_variables():
    """A panel that hardcodes no selector ignores the job and instance pickers."""
    document = json.loads(DASHBOARD.read_text())
    offenders = [
        (panel["title"], t["expr"])
        for panel in document["panels"]
        for t in panel.get("targets", [])
        if "batcher_" in t.get("expr", "") and 'job=~"$job"' not in t["expr"]
    ]
    assert not offenders, f"panels ignoring the job selector: {offenders}"
