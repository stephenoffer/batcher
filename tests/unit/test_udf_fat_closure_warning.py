"""A UDF that carries megabytes of data is a bug the engine can see for free.

The field guides are blunt: "10MB+ function closure means a bug in your code" (`core-024`,
`core-026`) — a lookup table or weight tensor caught by the callable and shipped with every
dispatch. Ray warns at 10 MB and errors at 100 MB; Batcher measured nothing.

The size comes out of a pickle the process path had to do anyway, so the check is free. Its
reach is exactly what that path can carry, and the tests below pin both halves of that:
what it catches, and what it deliberately cannot see.
"""

from __future__ import annotations

import functools
import warnings

import numpy as np
import pytest

from batcher._internal.errors import PerformanceWarning
from batcher.core.udf import processes

pytestmark = pytest.mark.unit

_BIG = np.zeros(15 << 20, dtype="uint8")  # 15 MB, over the 10 MB mark
_SMALL = np.zeros(1 << 10, dtype="uint8")


@pytest.fixture(autouse=True)
def _reset_once_flag():
    processes._FAT_CLOSURE_WARNED = False
    yield
    processes._FAT_CLOSURE_WARNED = False


def _udf(batch, table=None):
    return batch


class _Holder:
    def __init__(self, table):
        self.table = table

    def __call__(self, batch):
        return batch


def test_data_bound_in_with_partial_is_flagged():
    """`map_batches(partial(fn, table=df))` is the common shape, and pickle carries the arg."""
    with pytest.warns(PerformanceWarning, match="serializes to"):
        processes.is_picklable(functools.partial(_udf, table=_BIG))


def test_a_callable_carrying_state_is_flagged():
    with pytest.warns(PerformanceWarning, match="15 MB"):
        processes.is_picklable(_Holder(_BIG))


def test_a_small_payload_is_silent():
    with warnings.catch_warnings():
        warnings.simplefilter("error", PerformanceWarning)
        processes.is_picklable(functools.partial(_udf, table=_SMALL))
        processes.is_picklable(_udf)


def test_a_global_reference_is_not_flagged_and_that_is_documented():
    """`pickle` serializes a module-level function by reference, so a plain `def` reading a
    large global pickles to a few bytes. Ray catches this only because cloudpickle
    serializes globals by value. Pinning the limit so nobody assumes coverage it lacks."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", PerformanceWarning)
        processes.is_picklable(_reads_a_big_global)
    assert "cannot see" in processes.warn_if_closure_is_fat.__doc__


def _reads_a_big_global(batch):
    _ = _BIG.size  # a genuine read of the module-level array
    return batch


def test_an_unpicklable_callable_is_rejected_not_measured():
    """A closure over large data never reaches the size check — it fails to pickle first."""
    table = _BIG

    def closure(batch):
        _ = table.size  # genuinely closes over the array
        return batch

    with warnings.catch_warnings():
        warnings.simplefilter("error", PerformanceWarning)
        # A local closure is picklable-by-reference only if importable; this one is not.
        processes.is_picklable(closure)


def test_it_warns_once_not_per_probe():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(3):
            processes.is_picklable(_Holder(_BIG))
    assert sum("serializes to" in str(c.message) for c in caught) == 1
