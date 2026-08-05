"""`EllipticEnvelope` — the outlier a per-column rule cannot see.

`flag_outliers` looks at one column at a time. Against a height/weight sample, 1.79m is about
half a standard deviation tall and 55kg about half a standard deviation light, so neither
value is remarkable and every per-column rule lets the row through - while together they sit
three noise-widths off the height/weight line. Mahalanobis distance measures how far a row is
from the fitted centre in the metric the covariance defines, which is what sees that.

The second thing this file pins is the train-then-apply split.
{py:func}`mahalanobis_distance` relearns the centre and covariance from whatever dataset it
is given, which is wrong for scoring new data: a batch made entirely of outliers relearns
itself as normal and comes back clean. `fit` and `predict` separate those, and a test drives
exactly that failure.
"""

from __future__ import annotations

import numpy as np
import pytest

import batcher as bt
from batcher._internal.errors import ColumnNotFoundError, PlanError
from batcher.ml import EllipticEnvelope

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def bodies() -> bt.Dataset:
    """Height and weight, correlated, with no outlier among them."""
    rng = np.random.default_rng(11)
    height = rng.normal(1.75, 0.08, size=200)
    weight = 60.0 + (height - 1.75) * 120.0 + rng.normal(0.0, 3.0, size=200)
    return bt.from_pydict({"h": height.tolist(), "w": weight.tolist()})


#: Tall and light: 1.79m is about half a standard deviation above the mean height, and 55kg
#: about half below the mean weight, so neither value is remarkable on its own. Together they
#: are three noise-widths off the height-weight line.
ODD_ROW = (1.79, 55.0)


def test_it_catches_a_row_that_is_ordinary_in_every_column_separately(bodies) -> None:
    """The whole reason the class exists."""
    envelope = EllipticEnvelope(["h", "w"], contamination=0.01).fit(bodies)
    odd = bt.from_pydict({"h": [ODD_ROW[0]], "w": [ODD_ROW[1]]})
    assert envelope.predict(odd).to_pydict()["is_outlier"] == [True]


def test_the_same_row_passes_every_per_column_rule(bodies) -> None:
    """Pins the premise, so the test above is not merely asserting a large number.

    A first draft of this used a row that was extreme in both columns, which the per-column
    z-score rule flagged too - proving nothing about what the envelope adds.
    """
    from batcher.ml.outliers import flag_outliers

    table = bodies.to_pydict()
    combined = bt.from_pydict(
        {"h": [*table["h"], ODD_ROW[0]], "w": [*table["w"], ODD_ROW[1]]},
    )
    flagged = flag_outliers(combined, ["h", "w"], method="zscore").to_pydict()
    assert flagged["h_outlier"][-1] is False
    assert flagged["w_outlier"][-1] is False


def test_a_typical_row_is_not_flagged(bodies) -> None:
    envelope = EllipticEnvelope(["h", "w"], contamination=0.01).fit(bodies)
    typical = bt.from_pydict({"h": [1.75], "w": [60.0]})
    assert envelope.predict(typical).to_pydict()["is_outlier"] == [False]


def test_the_envelope_learned_on_training_data_is_applied_unchanged(bodies) -> None:
    """The failure this class exists to prevent, driven directly.

    A batch of nothing but outliers, scored by relearning its own statistics, looks clean.
    Scored against the training envelope it does not.
    """
    from batcher.ml.outliers import mahalanobis_distance

    envelope = EllipticEnvelope(["h", "w"], contamination=0.01).fit(bodies)
    all_odd = bt.from_pydict({"h": [1.79, 1.82, 1.85], "w": [55.0, 56.0, 58.0]})

    relearned = mahalanobis_distance(all_odd, ["h", "w"]).to_pydict()["mahalanobis"]
    assert max(relearned) < 2.0, "relearning on the batch makes its own outliers look normal"

    applied = envelope.predict(all_odd).to_pydict()["is_outlier"]
    assert applied == [True, True, True]


def test_contamination_sets_how_many_training_rows_fall_outside(bodies) -> None:
    """A looser envelope must flag at least as much as a tighter one, never less."""
    tight = EllipticEnvelope(["h", "w"], contamination=0.001).fit(bodies)
    loose = EllipticEnvelope(["h", "w"], contamination=0.2).fit(bodies)
    tight_count = sum(tight.predict(bodies).to_pydict()["is_outlier"])
    loose_count = sum(loose.predict(bodies).to_pydict()["is_outlier"])
    assert loose.threshold_ < tight.threshold_
    assert loose_count >= tight_count


def test_score_samples_ranks_without_thresholding(bodies) -> None:
    envelope = EllipticEnvelope(["h", "w"]).fit(bodies)
    probe = bt.from_pydict({"h": [1.75, ODD_ROW[0]], "w": [60.0, ODD_ROW[1]]})
    scores = envelope.score_samples(probe).to_pydict()["mahalanobis"]
    assert scores[1] > scores[0]
    assert all(v >= 0 for v in scores)


def test_the_fit_is_one_pass_per_statistic_and_reads_nothing_at_predict(bodies) -> None:
    """`predict` must lower to an expression, not another scan."""
    frame = type(bodies)
    calls = {"n": 0}
    originals = {name: getattr(frame, name) for name in ("collect", "to_pydict", "count")}

    def wrap(original):
        def counted(self, *args, **kwargs):
            calls["n"] += 1
            return original(self, *args, **kwargs)

        return counted

    envelope = EllipticEnvelope(["h", "w"]).fit(bodies)
    for name, original in originals.items():
        setattr(frame, name, wrap(original))
    try:
        envelope.predict(bodies)
        envelope.score_samples(bodies)
    finally:
        for name, original in originals.items():
            setattr(frame, name, original)
    assert calls["n"] == 0, "predict and score_samples must build expressions, not execute"


def test_a_union_of_partitions_fits_what_one_partition_does(bodies) -> None:
    table = bodies.to_pydict()
    half = len(table["h"]) // 2
    left = bt.from_pydict({k: v[:half] for k, v in table.items()})
    right = bt.from_pydict({k: v[half:] for k, v in table.items()})
    whole = EllipticEnvelope(["h", "w"]).fit(bodies)
    split = EllipticEnvelope(["h", "w"]).fit(left.union(right))
    assert whole.location_ == pytest.approx(split.location_, abs=1e-9)
    assert whole.threshold_ == pytest.approx(split.threshold_, abs=1e-12)


def test_perfectly_collinear_columns_are_scored_rather_than_refused() -> None:
    """A singular covariance is a shape of data, so the pseudo-inverse handles it."""
    ds = bt.from_pydict({"a": [1.0, 2.0, 3.0, 4.0], "b": [2.0, 4.0, 6.0, 8.0]})
    envelope = EllipticEnvelope(["a", "b"]).fit(ds)
    assert len(envelope.predict(ds).to_pydict()["is_outlier"]) == 4


def test_the_output_column_is_configurable(bodies) -> None:
    envelope = EllipticEnvelope(["h", "w"], output_column="anomalous").fit(bodies)
    assert "anomalous" in envelope.predict(bodies).columns


# --------------------------------------------------------------------------------------
# Rejections
# --------------------------------------------------------------------------------------


def test_no_columns_is_rejected() -> None:
    with pytest.raises(PlanError, match="at least one column"):
        EllipticEnvelope([])


@pytest.mark.parametrize("value", [0.0, 1.0, -0.5, 2.0])
def test_a_contamination_outside_the_unit_interval_is_rejected(value: float) -> None:
    with pytest.raises(PlanError, match="between 0 and 1"):
        EllipticEnvelope(["a"], contamination=value)


def test_a_missing_column_is_named(bodies) -> None:
    with pytest.raises(ColumnNotFoundError):
        EllipticEnvelope(["nope"]).fit(bodies)


def test_a_string_column_is_named(bodies) -> None:
    with pytest.raises(PlanError, match="'label'"):
        EllipticEnvelope(["label"]).fit(bodies.with_columns(label=bt.lit("x")))


def test_predicting_before_fitting_is_rejected(bodies) -> None:
    with pytest.raises(PlanError, match="must be fitted"):
        EllipticEnvelope(["h", "w"]).predict(bodies)
