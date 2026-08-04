"""`drift_report` must not re-read a column once per metric.

It used to cost six executions per column: a mean/null-rate aggregate on each frame, plus a
full binning-and-join for the PSI and *another* for the Jensen-Shannon divergence — two
readings of the same aligned shares, computed twice. On a monitoring job that runs hourly
over a wide table, that is the whole bill.

The counts below are the guard. The equivalence tests beside them are what make the sharing
safe: the report's numbers must still equal the standalone metrics exactly, because those
are the functions a user calls directly and the ones the doctests pin.
"""

from __future__ import annotations

import numpy as np
import pytest

import batcher as bt
from batcher.api.dataset.frame import Dataset
from batcher.ml.stats.drift import drift_report, js_divergence, population_stability_index

pytestmark = pytest.mark.unit


@pytest.fixture
def executions(monkeypatch) -> list[int]:
    """Count executed queries, including the ones that go out through `to_pydict`."""
    tally = [0]
    for name in ("collect", "to_pydict"):
        original = getattr(Dataset, name)

        def counting(self, *args, _original=original, **kwargs):
            tally[0] += 1
            return _original(self, *args, **kwargs)

        monkeypatch.setattr(Dataset, name, counting)
    return tally


def _frames(width: int, rows: int = 300) -> tuple[bt.Dataset, bt.Dataset, list[str]]:
    rng = np.random.default_rng(0)
    reference = bt.from_pydict({f"f{i}": rng.normal(size=rows).tolist() for i in range(width)})
    current = bt.from_pydict(
        {f"f{i}": (rng.normal(size=rows) + 0.4).tolist() for i in range(width)}
    )
    return reference, current, [f"f{i}" for i in range(width)]


@pytest.mark.parametrize("width", [1, 4, 10])
def test_the_summary_half_costs_two_passes_whatever_the_width(width: int, executions) -> None:
    """Two aggregates total for the means and null rates, not two per column."""
    reference, current, columns = _frames(width)
    executions[0] = 0
    drift_report(reference, current, columns)
    # Two summary aggregates, plus the per-column binning that genuinely cannot be shared
    # across columns (each needs its own quantile edges).
    assert executions[0] <= 2 + 3 * width


@pytest.mark.parametrize("width", [4, 10])
def test_psi_and_js_share_one_binning_per_column(width: int, executions) -> None:
    """Computing them separately would double the per-column cost."""
    reference, current, columns = _frames(width)
    executions[0] = 0
    drift_report(reference, current, columns)
    shared = executions[0]

    executions[0] = 0
    for name in columns:
        population_stability_index(reference, current, name)
        js_divergence(reference, current, name)
    separate = executions[0]
    assert shared < separate


def test_the_report_matches_the_standalone_metrics_exactly() -> None:
    reference, current, columns = _frames(5)
    report = drift_report(reference, current, columns).sort("column").to_pydict()
    for index, name in enumerate(report["column"]):
        assert report["psi"][index] == pytest.approx(
            population_stability_index(reference, current, name), rel=1e-12
        )
        assert report["js_divergence"][index] == pytest.approx(
            js_divergence(reference, current, name), rel=1e-12
        )


def test_the_mean_and_null_rate_shifts_are_still_right() -> None:
    reference = bt.from_pydict({"x": [0.0, 2.0, 4.0, None]})
    current = bt.from_pydict({"x": [10.0, 12.0, 14.0, 16.0]})
    row = drift_report(reference, current, ["x"]).to_pydict()
    assert row["mean_shift"][0] == pytest.approx(13.0 - 2.0)
    assert row["null_rate_shift"][0] == pytest.approx(-0.25)


def test_columns_keep_their_own_statistics() -> None:
    """The batched summary must not cross-wire one column's mean onto another."""
    reference = bt.from_pydict({"a": [0.0, 1.0, 2.0, 3.0], "b": [5.0, 6.0, 7.0, 8.0]})
    current = bt.from_pydict({"a": [10.0, 11.0, 12.0, 13.0], "b": [5.0, 6.0, 7.0, 8.0]})
    rows = drift_report(reference, current, ["a", "b"]).sort("column").to_pydict()
    assert rows["column"] == ["a", "b"]
    assert rows["mean_shift"][0] == pytest.approx(10.0)
    assert rows["mean_shift"][1] == pytest.approx(0.0)


def test_a_drifted_column_still_outranks_a_stable_one() -> None:
    rng = np.random.default_rng(5)
    rows = 400
    reference = bt.from_pydict(
        {"stable": rng.normal(size=rows).tolist(), "drifted": rng.normal(size=rows).tolist()}
    )
    current = bt.from_pydict(
        {
            "stable": rng.normal(size=rows).tolist(),
            "drifted": (rng.normal(size=rows) + 4.0).tolist(),
        }
    )
    report = drift_report(reference, current, ["stable", "drifted"]).to_pydict()
    assert report["column"][0] == "drifted"  # sorted by psi, descending
