"""Scoring a model from SQL — ``ML_PREDICT`` and BigQuery's ``ML.PREDICT``.

A warehouse can score a model inside a query; a DataFrame-only engine makes you leave SQL to
do it. These cases pin that a Batcher query can, in both spellings, and that both land on the
same `Dataset.ml.predict` call rather than on a second inference path.

The model here is a fitted `LinearRegression` over one feature, so the predictions are exact
and a wrong feature order or a dropped setting shows up as a number rather than as a shrug.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher._internal.errors import PlanError
from batcher.ml import LinearRegression

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def doubler():
    """``y = 2x``, fitted exactly, so every prediction below is a checkable number."""
    train = bt.from_pydict({"x": [1.0, 2.0, 3.0, 4.0], "y": [2.0, 4.0, 6.0, 8.0]})
    return LinearRegression(features=["x"], target="y").fit(train)


def _session(doubler, dialect: str) -> bt.Session:
    s = bt.Session(dialect=dialect)
    s.register("points", bt.from_pydict({"x": [5.0, 10.0]}))
    s.register_model("doubler", doubler)
    return s


@pytest.fixture
def session(doubler):
    return _session(doubler, "duckdb")


@pytest.fixture
def bigquery(doubler):
    """The same catalog read as BigQuery, which is the only dialect that parses `ML.PREDICT`."""
    return _session(doubler, "bigquery")


def _predictions(ds) -> list[float]:
    return [round(v, 6) for v in ds.to_pydict()["prediction"]]


# --- the neutral spelling, in the default dialect --------------------------------------


def test_a_registered_model_scores_a_registered_table(session):
    out = session.sql("SELECT x, prediction FROM ML_PREDICT(points, doubler) ORDER BY x")
    assert out.to_pydict()["x"] == [5.0, 10.0]
    assert _predictions(out) == [10.0, 20.0]


def test_the_scored_relation_can_be_a_subquery(session):
    out = session.sql(
        "SELECT prediction FROM ML_PREDICT((SELECT x FROM points WHERE x > 6), doubler)"
    )
    assert _predictions(out) == [20.0]


def test_the_result_is_an_ordinary_relation_to_keep_querying(session):
    """The point of scoring *in* SQL: the prediction is a column the rest of the query sees."""
    out = session.sql("SELECT COUNT(*) AS n FROM ML_PREDICT(points, doubler) WHERE prediction > 15")
    assert out.to_pydict() == {"n": [1]}


def test_a_per_call_binding_is_scorable_without_registering_the_table(session):
    out = session.sql(
        "SELECT prediction FROM ML_PREDICT(t, doubler)", t=bt.from_pydict({"x": [7.0]})
    )
    assert _predictions(out) == [14.0]


def test_the_output_column_can_be_renamed(session):
    out = session.sql(
        "SELECT score FROM ML_PREDICT(points, doubler, output_column => 'score') ORDER BY score"
    )
    assert [round(v, 6) for v in out.to_pydict()["score"]] == [10.0, 20.0]


def test_a_native_estimator_scores_a_relation_wider_than_it_was_fitted_on(session, doubler):
    """It reads its own feature columns by name, so extra columns ride along untouched."""
    wide = bt.from_pydict({"x": [3.0], "note": ["ignored"], "extra": [99.0]})
    out = session.sql("SELECT note, prediction FROM ML_PREDICT(t, doubler)", t=wide)
    assert out.to_pydict()["note"] == ["ignored"]
    assert _predictions(out) == [6.0]


def test_features_is_refused_for_a_native_estimator_that_already_has_them(session):
    """Silently ignoring it would be the bad outcome: the query would look configured and not be."""
    with pytest.raises(PlanError, match=r"cannot be set for a batcher\.ml estimator"):
        session.sql("SELECT * FROM ML_PREDICT(points, doubler, features => ['x'])")


# --- BigQuery's spelling ---------------------------------------------------------------


def test_bigquery_ml_predict_runs_as_written(bigquery):
    """A ported BigQuery query should not have to be rewritten to run here."""
    out = bigquery.sql(
        "SELECT x, prediction FROM ML.PREDICT(MODEL doubler, TABLE points) ORDER BY x"
    )
    assert _predictions(out) == [10.0, 20.0]


def test_bigquery_ml_predict_takes_a_subquery_as_its_relation(bigquery):
    out = bigquery.sql(
        "SELECT prediction FROM ML.PREDICT(MODEL doubler, (SELECT x FROM points WHERE x < 6))"
    )
    assert _predictions(out) == [10.0]


def test_bigquery_settings_come_through_the_struct_argument(bigquery):
    out = bigquery.sql(
        "SELECT score FROM ML.PREDICT(MODEL doubler, TABLE points, "
        "STRUCT('score' AS output_column)) ORDER BY score"
    )
    assert [round(v, 6) for v in out.to_pydict()["score"]] == [10.0, 20.0]


def test_both_spellings_produce_the_same_relation(session, bigquery):
    """One implementation, two grammars — the thing that keeps them from drifting."""
    neutral = session.sql("SELECT x, prediction FROM ML_PREDICT(points, doubler) ORDER BY x")
    ported = bigquery.sql(
        "SELECT x, prediction FROM ML.PREDICT(MODEL doubler, TABLE points) ORDER BY x"
    )
    assert neutral.to_pydict() == ported.to_pydict()


# --- what it refuses, and how ----------------------------------------------------------


def test_an_unregistered_model_names_what_is_registered(session):
    with pytest.raises(PlanError, match="unknown model 'missing'"):
        session.sql("SELECT * FROM ML_PREDICT(points, missing)")


def test_an_unknown_table_is_reported_as_a_table_not_as_a_model(session):
    with pytest.raises(PlanError, match="unknown table 'nowhere'"):
        session.sql("SELECT * FROM ML_PREDICT(nowhere, doubler)")


def test_an_unsupported_setting_lists_the_supported_ones(session):
    with pytest.raises(PlanError, match="unknown model-scoring setting 'num_gpus'"):
        session.sql("SELECT * FROM ML_PREDICT(points, doubler, num_gpus => '1')")


def test_a_missing_argument_is_refused_before_anything_runs(session):
    with pytest.raises(PlanError, match="exactly two positional arguments"):
        session.sql("SELECT * FROM ML_PREDICT(points)")


# --- the catalog -----------------------------------------------------------------------


def test_registering_a_model_is_visible_to_list_models(session):
    assert session.list_models() == ["doubler"]


def test_a_model_registered_after_a_query_ran_is_not_served_from_the_plan_cache(session, doubler):
    """Registering bumps the catalog generation, or a prepared statement outlives its catalog."""
    with pytest.raises(PlanError, match="unknown model 'second'"):
        session.sql("SELECT * FROM ML_PREDICT(points, second)")
    session.register_model("second", doubler)
    assert _predictions(session.sql("SELECT prediction FROM ML_PREDICT(points, second)")) == [
        10.0,
        20.0,
    ]


def test_an_empty_model_name_is_refused():
    with pytest.raises(PlanError, match="non-empty string"):
        bt.Session().register_model("", object())
