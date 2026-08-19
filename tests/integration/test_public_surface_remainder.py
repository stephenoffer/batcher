"""The final unexercised public callables, closed against what each one promises.

After the geospatial, string, temporal, connector and inference rounds, the
execution-coverage sweep had thirty-odd names left. They are not a theme, they are a
remainder -- the activations at the bottom of the numeric surface, the ``GroupBy``
whole-frame shorthands, the streaming progress record's serializers, the ML data-loader
exits, and four sinks nobody had called. What they share is that each is the *last* thing
between a working pipeline and its output, which is the worst place to find a defect.

Where a method has an exact definition, that definition is the oracle and the test is
written from it rather than from the engine's answer: ``elu`` against
``x if x > 0 else exp(x) - 1``, ``softsign`` against ``x / (1 + |x|)``, ``expanding_var``
against the running sample variance. Where it does not -- ``format_bytes``, the progress
serializers -- the test asserts the contract the caller depends on: the field names, the
round trip, the units.

Two names stay out and are named here so their absence is a decision rather than an
oversight: ``from_spark`` needs a JVM, and ``to_ray_dataset`` / ``from_ray_dataset`` need a
Ray cluster whose workers share this process's engine build. Both have tests that run where
those exist; neither can run here.
"""

from __future__ import annotations

import json
import math

import pytest

import batcher as bt

pytestmark = pytest.mark.integration

VALUES = [1.0, -2.0, 3.0, -0.5]
GROUPS = {"g": ["a", "a", "b", "b"], "n": [1, 2, 3, 4], "x": [1.0, -2.0, 3.0, -0.5]}


@pytest.fixture
def ds():
    return bt.from_pydict(GROUPS)


#: ``(method, kwargs, reference)`` for the activations, each against its definition.
ACTIVATIONS = [
    ("elu", {}, lambda v: v if v > 0 else math.exp(v) - 1.0),
    ("hardtanh", {}, lambda v: max(-1.0, min(1.0, v))),
    ("leaky_relu", {}, lambda v: v if v > 0 else 0.01 * v),
    ("softsign", {}, lambda v: v / (1.0 + abs(v))),
    ("tanhshrink", {}, lambda v: v - math.tanh(v)),
]


@pytest.mark.parametrize(("method", "kwargs", "reference"), ACTIVATIONS)
def test_activation_matches_its_definition(ds, method, kwargs, reference):
    """Each activation against the formula it is named for, on both signs and a fraction."""
    got = ds.select(v=getattr(bt.col("x"), method)(**kwargs)).to_pydict()["v"]
    want = [reference(v) for v in GROUPS["x"]]
    assert got == pytest.approx(want, rel=1e-12), f"{method}: {got} vs {want}"


def test_the_activations_are_shaped_the_way_their_names_promise(ds):
    """Sign behaviour and saturation, which a formula comparison alone does not read as.

    A transcription error that swapped two of these formulas would satisfy the parametrized
    test for whichever pair it swapped; these assertions are about what each is *for*.
    """
    got = ds.select(
        elu=bt.col("x").elu(),
        hardtanh=bt.col("x").hardtanh(),
        leaky=bt.col("x").leaky_relu(),
        soft=bt.col("x").softsign(),
    ).to_pydict()
    positives = [i for i, v in enumerate(GROUPS["x"]) if v > 0]
    negatives = [i for i, v in enumerate(GROUPS["x"]) if v < 0]
    for i in positives:
        assert got["elu"][i] == GROUPS["x"][i], "ELU is the identity above zero"
        assert got["leaky"][i] == GROUPS["x"][i], "leaky ReLU is the identity above zero"
    for i in negatives:
        assert -1.0 < got["elu"][i] < 0.0, "ELU saturates at -1 below zero"
        assert got["leaky"][i] < 0.0, "leaky ReLU leaks rather than clamping to zero"
    assert all(-1.0 <= v <= 1.0 for v in got["hardtanh"]), "hardtanh is bounded to [-1, 1]"
    assert all(-1.0 < v < 1.0 for v in got["soft"]), "softsign is bounded and never saturates"


def test_fill_nan_replaces_a_nan_and_leaves_a_null_alone():
    """The distinction that makes it a separate method from ``fill_null``.

    A NaN is a float that arrived from a computation; a null is an absent value. Filling one
    must not fill the other, which is the whole reason both methods exist.
    """
    ds = bt.from_pydict({"x": [1.0, float("nan"), None, -2.0]})
    got = ds.select(filled=bt.col("x").fill_nan(0.0), nulls=bt.col("x").fill_null(99.0)).to_pydict()
    assert got["filled"][1] == 0.0, "the NaN was replaced"
    assert got["filled"][2] is None, "the null was not"
    assert got["nulls"][2] == 99.0, "and fill_null does the opposite"
    assert math.isnan(got["nulls"][1]), "fill_null leaves the NaN"


def test_format_bytes_renders_a_human_size(ds):
    """Byte counts as text, with the unit and the singular case both correct."""
    sizes = [0, 1, 2, 1023, 1024, 1536, 1048576, 1073741824]
    got = bt.from_pydict({"n": sizes}).select(v=bt.col("n").format_bytes()).to_pydict()["v"]
    assert got[0].startswith("0")
    # Singular-vs-plural for exactly one byte is left alone: both spellings are in
    # circulation and which one this build renders is not what this test is for. What it
    # pins is that the count and the unit are both there.
    assert got[1].startswith("1 byte"), f"one byte rendered as {got[1]!r}"
    assert got[2] == "2 bytes"
    assert "KiB" in got[4] or "KB" in got[4], f"1024 rendered as {got[4]!r}"
    assert "MiB" in got[6] or "MB" in got[6], f"1 Mi rendered as {got[6]!r}"
    assert "GiB" in got[7] or "GB" in got[7], f"1 Gi rendered as {got[7]!r}"
    assert all(isinstance(v, str) for v in got)


def test_expanding_std_and_var_are_the_running_sample_statistics():
    """Each row's value over every row up to it, with one row having no sample deviation."""
    values = [2.0, 4.0, 4.0, 4.0, 5.0]
    ds = bt.from_pydict({"i": list(range(len(values))), "v": values})
    got = ds.select(
        std=bt.col("v").expanding_std(order_by=[bt.col("i")]),
        var=bt.col("v").expanding_var(order_by=[bt.col("i")]),
    ).to_pydict()

    assert got["std"][0] is None or math.isnan(got["std"][0]), (
        "one observation has no sample deviation"
    )
    for row in range(1, len(values)):
        window = values[: row + 1]
        mean = sum(window) / len(window)
        variance = sum((v - mean) ** 2 for v in window) / (len(window) - 1)
        assert got["var"][row] == pytest.approx(variance, rel=1e-12), f"var at row {row}"
        assert got["std"][row] == pytest.approx(math.sqrt(variance), rel=1e-12), row


def test_kurtosis_pop_is_the_population_form(ds):
    """Population kurtosis, which differs from the sample form on a small column.

    Both are exposed, so the pair has to disagree somewhere or one of them is mislabelled.
    """
    values = [1.0, 2.0, 3.0, 4.0, 10.0]
    frame = bt.from_pydict({"v": values})
    population = frame.agg(k=bt.col("v").kurtosis_pop()).to_pydict()["k"][0]
    sample = frame.agg(k=bt.col("v").kurtosis()).to_pydict()["k"][0]
    assert population is not None
    mean = sum(values) / len(values)
    m2 = sum((v - mean) ** 2 for v in values) / len(values)
    m4 = sum((v - mean) ** 4 for v in values) / len(values)
    assert population == pytest.approx(m4 / m2**2 - 3.0, rel=1e-9), (
        "population excess kurtosis is m4 / m2^2 - 3"
    )
    if sample is not None:
        assert sample != pytest.approx(population), (
            "the sample and population forms must not be the same number"
        )


def test_quantile_disc_returns_a_value_that_is_in_the_column():
    """Discrete quantile: it picks an observation rather than interpolating between two."""
    values = [1.0, 2.0, 3.0, 4.0]
    frame = bt.from_pydict({"v": values})
    discrete = frame.agg(q=bt.col("v").quantile_disc(0.5)).to_pydict()["q"][0]
    continuous = frame.agg(q=bt.col("v").quantile(0.5)).to_pydict()["q"][0]
    assert discrete in values, f"{discrete} is not one of the observed values"
    assert continuous not in values or continuous == discrete, (
        "the fixture must be one where interpolation lands between two observations"
    )


def test_top_k_returns_the_most_frequent_values(ds):
    """``top_k`` is a frequency aggregate, not an ordering one -- the easy misreading."""
    values = ["a", "b", "a", "c", "a", "b"]
    got = bt.from_pydict({"v": values}).agg(t=bt.col("v").top_k(2)).to_pydict()["t"][0]
    assert got[0] == "a", f"'a' occurs three times and must lead, got {got}"
    assert set(got) == {"a", "b"}, f"the two most frequent are a and b, got {got}"
    assert len(got) == 2


#: ``(shorthand, per-column expression)`` for the ``GroupBy`` whole-frame aggregates.
GROUP_SHORTHANDS = [
    ("array_agg", lambda: bt.col("n").array_agg()),
    ("mode", lambda: bt.col("n").mode()),
    ("product", lambda: bt.col("n").product()),
    ("kurtosis", lambda: bt.col("n").kurtosis()),
    ("skewness", lambda: bt.col("n").skewness()),
]


@pytest.mark.parametrize(("shorthand", "expression"), GROUP_SHORTHANDS)
def test_group_by_shorthand_matches_the_explicit_aggregate(ds, shorthand, expression):
    """Each shorthand must equal spelling the same aggregate out per column."""
    short = getattr(ds.group_by("g"), shorthand)("n").to_pydict()
    explicit = ds.group_by("g").agg(n=expression()).to_pydict()
    assert sorted(short) == sorted(explicit), f"{shorthand} produced {sorted(short)}"
    assert dict(zip(short["g"], short["n"], strict=True)) == dict(
        zip(explicit["g"], explicit["n"], strict=True)
    )


def test_group_by_array_agg_and_product_have_the_values_they_should(ds):
    """Two of the shorthands checked against arithmetic, not just against each other."""
    collected = ds.group_by("g").array_agg("n").to_pydict()
    by_group = dict(zip(collected["g"], collected["n"], strict=True))
    assert sorted(by_group["a"]) == [1, 2]
    assert sorted(by_group["b"]) == [3, 4]

    products = ds.group_by("g").product("n").to_pydict()
    product_by_group = dict(zip(products["g"], products["n"], strict=True))
    assert product_by_group["a"] == pytest.approx(2.0)
    assert product_by_group["b"] == pytest.approx(12.0)


def test_dataset_cov_matches_the_sample_covariance(ds):
    """``ds.cov`` against the definition, and against the variance when both columns match."""
    got = ds.cov("n", "n")
    values = GROUPS["n"]
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    assert got == pytest.approx(variance, rel=1e-12), (
        "the covariance of a column with itself is its variance"
    )
    mixed = ds.cov("n", "x")
    assert isinstance(mixed, float)


def test_streaming_progress_serializes_to_the_spark_field_names():
    """``to_dict`` / ``json`` are what a monitoring sink reads, so the keys are the contract."""
    operator = bt.StateOperatorProgress(
        operator_name="agg", num_rows_total=10, num_late_inputs_dropped=3
    )
    progress = bt.StreamingQueryProgress(
        batch_id=1,
        num_input_rows=100,
        num_output_rows=50,
        duration_ms=12.5,
        timestamp=1.0,
        name="q",
        state_operators=(operator,),
        duration_breakdown_ms=(("addBatch", 8.0), ("commit", 4.5)),
    )
    as_dict = progress.to_dict()
    assert as_dict["batchId"] == 1, "the keys are camelCase, as Spark writes them"
    assert as_dict["numInputRows"] == 100
    assert as_dict["name"] == "q"

    decoded = json.loads(progress.json())
    assert decoded == as_dict, "json() must be to_dict() encoded, not a second rendering"


def test_streaming_progress_duration_breakdown_is_a_mapping():
    """``duration_ms_map`` turns the ordered breakdown into the map a dashboard indexes."""
    progress = bt.StreamingQueryProgress(
        batch_id=0,
        num_input_rows=1,
        num_output_rows=1,
        duration_ms=12.5,
        timestamp=0.0,
        duration_breakdown_ms=(("addBatch", 8.0), ("commit", 4.5)),
    )
    assert progress.duration_ms_map == {"addBatch": 8.0, "commit": 4.5}
    assert sum(progress.duration_ms_map.values()) == pytest.approx(12.5), (
        "the phases must account for the batch's own duration"
    )


def test_state_operator_reports_rows_the_watermark_dropped():
    """The number that tells a user their watermark is discarding data."""
    operator = bt.StateOperatorProgress(operator_name="agg", num_late_inputs_dropped=7)
    assert operator.num_rows_dropped_by_watermark == 7
    assert bt.StateOperatorProgress(operator_name="agg").num_rows_dropped_by_watermark == 0


def test_ml_to_torch_and_dataloader_carry_the_rows(ds):
    """``ds.ml.to_torch`` and its DataLoader wrapper, which are the training-loop exits."""
    torch = pytest.importorskip("torch")
    frame = bt.from_pydict({"a": [1.0, 2.0, 3.0, 4.0]})

    seen: list[float] = []
    for batch in frame.ml.to_torch(batch_size=2, columns=["a"]):
        seen.extend(float(v) for v in batch["a"].reshape(-1))
    assert seen == [1.0, 2.0, 3.0, 4.0]

    loader = frame.ml.to_torch_dataloader(batch_size=2, columns=["a"])
    assert isinstance(loader, torch.utils.data.DataLoader)
    from_loader: list[float] = []
    for batch in loader:
        from_loader.extend(float(v) for v in batch["a"].reshape(-1))
    assert sorted(from_loader) == [1.0, 2.0, 3.0, 4.0]


def test_ml_download_reports_a_failure_per_row_rather_than_aborting(tmp_path):
    """``ds.ml.download`` over URLs that cannot resolve, with ``on_error='null'``.

    No network is assumed: every URL here is unroutable on purpose. What is asserted is the
    error contract -- a fetch that fails nulls its row instead of taking the whole job down,
    which is the behaviour a million-row crawl depends on.
    """
    urls = ["http://127.0.0.1:9/one", "http://127.0.0.1:9/two"]
    ds = bt.from_pydict({"url": urls})
    got = ds.ml.download("url", output_column="blob", on_error="null").to_pydict()
    assert len(got["blob"]) == len(urls), "every row must come back"
    assert all(v is None for v in got["blob"]), "an unreachable URL nulls its row"
    assert got["url"] == urls, "the input column is preserved"


def test_read_documents_reports_a_missing_file(tmp_path):
    """The document reader, on a path that is not there."""
    missing = str(tmp_path / "absent.pdf")
    with pytest.raises(Exception) as failure:
        bt.read.documents(missing).to_pydict()
    assert "absent.pdf" in str(failure.value)


def test_write_noop_consumes_the_rows_without_writing_anything(tmp_path):
    """The sink that exists to measure a pipeline without paying for its output."""
    query = bt.from_pydict({"a": [1, 2, 3]}).write.noop()
    assert query is not None
    query.process_all_available()
    query.stop()
    assert not list(tmp_path.iterdir()), "the noop sink must not write a file"


def test_write_for_each_sees_every_row():
    """``write.for_each`` hands each row to a callable, which is the escape hatch sink."""
    seen: list[dict] = []
    query = bt.from_pydict({"a": [1, 2, 3]}).write.for_each(seen.append)
    query.process_all_available()
    query.stop()
    assert len(seen) == 3, f"the sink saw {len(seen)} rows"
    assert sorted(row["a"] for row in seen) == [1, 2, 3]


def test_write_sql_and_mongo_say_what_they_need():
    """Two sinks nothing had called; neither can connect here, so the message is the contract."""
    ds = bt.from_pydict({"a": [1, 2]})
    for name, call in [("sql", lambda: ds.write.sql("t")), ("mongo", lambda: ds.write.mongo("c"))]:
        with pytest.raises(Exception) as failure:
            call()
        message = str(failure.value).lower()
        assert any(
            hint in message for hint in ("requires", "install", "uri", "missing", "argument")
        ), f"ds.write.{name} failed with a message a user cannot act on: {failure.value!r}"


def test_await_any_termination_and_reset_return_promptly_with_no_query_running():
    """The process-wide streaming waits, on a process with nothing streaming.

    ``await_any_termination`` must not block forever when there is nothing to wait for, and
    ``reset_terminated`` must clear the latch so a later call waits again rather than
    returning stale news.
    """
    bt.reset_terminated()
    assert bt.await_any_termination(timeout=0.1) in (True, False)
    assert bt.reset_terminated() is None
    assert bt.await_any_termination(timeout=0.1) in (True, False)
    assert bt.streams() == [] or all(hasattr(q, "name") for q in bt.streams())


@pytest.fixture(scope="module")
def ray_with_capacity():
    """A Ray driver with at least one CPU free, however that has to be arranged.

    ``ray.data.from_arrow_refs`` launches a metadata task that needs one CPU. On a shared
    cluster whose CPUs are all held by other sessions' placement groups there is never one
    free, and the call **blocks forever** rather than failing -- which is what made an
    earlier version of this test hang for seven minutes and die to SIGTERM. It looked like a
    defect in the handoff until ``ray.available_resources()`` showed ``CPU`` absent while
    ``cluster_resources()`` reported 96.

    So: attach first, and if the attached cluster has nothing to spare, disconnect this
    driver and bring up a private local one. Disconnecting does not stop the shared cluster,
    and the private one is torn down here, so a starved box costs a slower test rather than
    a skipped name or a hung suite.
    """
    ray = pytest.importorskip("ray")
    pytest.importorskip("ray.data")
    from tests._ray_cluster import init_test_ray, shutdown_test_ray

    started = init_test_ray(2)
    private = False
    if ray.available_resources().get("CPU", 0) < 1:
        ray.shutdown()
        ray.init(
            address="local",
            num_cpus=2,
            include_dashboard=False,
            log_to_driver=False,
            configure_logging=False,
        )
        private, started = True, False
    if ray.available_resources().get("CPU", 0) < 1:  # pragma: no cover - belt and braces
        pytest.skip("no Ray cluster with a free CPU could be reached or started")
    yield ray
    if private:
        ray.shutdown()
    else:
        shutdown_test_ray(started)


def test_to_and_from_ray_dataset_round_trip(ray_with_capacity):
    """The Ray Data handoff, both directions, over a cluster that can schedule it."""
    rows = {"a": [1, 2, 3], "s": ["x", "y", "z"]}
    handle = bt.from_pydict(rows).to_ray_dataset()
    assert handle is not None, "to_ray_dataset returned nothing"
    assert handle.count() == 3, "the Ray dataset must hold every row"

    back = bt.from_ray_dataset(handle).to_pydict()
    assert sorted(back["a"]) == [1, 2, 3]
    assert sorted(back["s"]) == ["x", "y", "z"]
    assert sorted(back) == ["a", "s"], "the round trip must not add or drop a column"


def test_to_ray_dataset_carries_the_schema_of_an_empty_result(ray_with_capacity):
    """A result with no rows still has columns; a Ray dataset built from no blocks has none.

    The export writes one empty block for exactly this reason, so what is asserted is that
    ``schema()`` survives -- without it every downstream Ray op fails on a column the user
    can see in ``ds.schema``.
    """
    empty = bt.from_pydict({"a": [1, 2], "s": ["x", "y"]}).filter(bt.col("a") > 99)
    handle = empty.to_ray_dataset()
    assert handle.count() == 0, "the result really is empty"
    schema = handle.schema()
    assert schema is not None, "an empty Ray dataset must still carry its columns"
    assert set(schema.names) == {"a", "s"}, f"columns were {schema.names}"
