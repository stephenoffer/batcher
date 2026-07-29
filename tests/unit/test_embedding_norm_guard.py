"""`normalize=False` against an endpoint that does not normalize is a silent wrong answer.

`openai_embedding_encoder` defaults to `normalize=False`, which is correct for OpenAI: its
API returns unit vectors, and re-normalizing one is pure cost. But the same request shape is
spoken by Azure, Together, and **vLLM's embedding server** — and vLLM does not normalize.
Point the encoder at one of those, keep the default, and every cosine similarity downstream
is computed over un-normalized vectors: a wrong ranking, nothing raised, no failing test.

Rather than guess a default per provider, the vectors are measured.
"""

from __future__ import annotations

import math
import warnings

import pyarrow as pa
import pytest

from batcher._internal.errors import PerformanceWarning
from batcher.ml import embed_api

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _reset_once_flag():
    embed_api._NORM_WARNED = False
    yield
    embed_api._NORM_WARNED = False


class _FakeEncoder(embed_api._ApiEncoder):
    """An `_ApiEncoder` whose endpoint returns whatever vectors the test supplies."""

    def __init__(self, vectors, **kw):
        super().__init__(
            text_column="t",
            output_column="embedding",
            output_type="fixed_size_list",
            max_batch=512,
            concurrency=1,
            **kw,
        )
        self._vectors = vectors

    def _embed_chunk(self, texts):
        return [self._vectors for _ in texts]


def _run(vectors, *, normalize):
    batch = pa.record_batch({"t": ["a", "b"]})
    return _FakeEncoder(vectors, normalize=normalize)(batch)


def test_un_normalized_vectors_with_normalize_false_warn():
    with pytest.warns(PerformanceWarning, match="not cosine similarity"):
        _run([3.0, 4.0], normalize=False)  # L2 norm 5.0


def test_the_warning_reports_the_measured_norm():
    """A bare "not normalized" leaves the user unsure whether it is a rounding artifact."""
    with pytest.warns(PerformanceWarning, match="5.000"):
        _run([3.0, 4.0], normalize=False)


def test_unit_vectors_stay_silent():
    """OpenAI's own vectors are unit-norm, and that is the default's whole justification —
    warning there would fire on every correct OpenAI pipeline."""
    unit = [1 / math.sqrt(3)] * 3
    with warnings.catch_warnings():
        warnings.simplefilter("error", PerformanceWarning)
        _run(unit, normalize=False)


def test_float_noise_does_not_trip_it():
    """A norm of 0.999 or 1.001 is float accumulation, not a missing normalize."""
    almost = [0.9999] + [0.0] * 9
    with warnings.catch_warnings():
        warnings.simplefilter("error", PerformanceWarning)
        _run(almost, normalize=False)


def test_normalize_true_never_warns():
    """With `normalize=True` the vectors are made unit here, so the endpoint's scale is
    irrelevant and the advice would be wrong."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", PerformanceWarning)
        _run([3.0, 4.0], normalize=True)


def test_it_warns_once_not_once_per_batch():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        enc = _FakeEncoder([3.0, 4.0], normalize=False)
        for _ in range(3):
            enc(pa.record_batch({"t": ["a"]}))
    assert sum("cosine" in str(c.message) for c in caught) == 1


def test_a_degenerate_response_is_not_judged():
    """An empty or missing vector should not produce a confident claim about its norm.

    Tested against the guard directly: an empty vector also fails later in the Arrow column
    build (a zero-length `fixed_size_list` is invalid), and routing through the encoder would
    make this assert that unrelated failure instead of the guard's own restraint."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", PerformanceWarning)
        embed_api._warn_if_not_unit_norm([])
        embed_api._warn_if_not_unit_norm([[]])
        embed_api._warn_if_not_unit_norm([None])
