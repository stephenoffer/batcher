"""A UDF column Arrow cannot represent must name the column and the fix.

The field guides flag this as a top multimodal footgun (`Arrow Pickled Object Type in
Schema`): a `map_batches` returning PIL Images, torch tensors, or any custom object hands
Arrow a value it cannot type. Ray Data's answer is to *silently* fall back to a pickle-backed
object column — 10-100x slower on every downstream transfer, and the user is expected to
notice by eyeballing `ds.schema()`.

Batcher fails loudly instead, which is right. But it used to fail loudly and *unhelpfully*:
a raw `pyarrow.lib.ArrowInvalid` quoting the offending value's repr, naming neither the
column, nor `map_batches`, nor what to do. These pin the actionable version.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher._internal.errors import PlanError
from batcher.interop.diagnostics import _remedy, _sample_type

pytestmark = pytest.mark.unit


class _Thing:
    """An opaque Python object, standing in for any un-Arrow-able UDF output."""


def test_the_offending_column_is_named() -> None:
    """pyarrow quotes the value's repr and never says which column it came from — which is
    the one thing a user needs to fix it."""
    ds = bt.from_pydict({"x": [1, 2, 3], "y": [4, 5, 6]})
    with pytest.raises(PlanError, match="'obj'"):
        ds.map_batches(
            lambda b: {"obj": [_Thing() for _ in range(b.num_rows)]}, output_columns=["obj"]
        ).collect()


def test_the_error_is_batchers_typed_one_not_pyarrows() -> None:
    """A raw `pyarrow.lib.ArrowInvalid` leaking out breaks the project's error contract, and
    cannot be caught by a user handling Batcher's exceptions."""
    ds = bt.from_pydict({"x": [1]})
    with pytest.raises(PlanError):
        ds.map_batches(lambda b: {"obj": [_Thing()]}, output_columns=["obj"]).collect()


def test_it_says_why_this_matters() -> None:
    """The cost is not the error — it is that an opaque object would otherwise be pickled,
    which is what makes every later stage slow."""
    ds = bt.from_pydict({"x": [1]})
    with pytest.raises(PlanError, match="pickled"):
        ds.map_batches(lambda b: {"obj": [_Thing()]}, output_columns=["obj"]).collect()


def _element(module: str, name: str) -> object:
    """One instance of a class that reports `module`/`name`, without importing either."""
    cls = type(name, (), {})
    cls.__module__ = module
    return cls()


@pytest.mark.parametrize(
    ("module", "name", "expected"),
    [
        ("torch", "Tensor", "cpu().numpy()"),
        ("PIL.Image", "Image", "np.asarray"),
        ("pandas.core.frame", "DataFrame", "columns individually"),
    ],
)
def test_the_named_causes_get_their_own_one_line_fix(module, name, expected) -> None:
    """PIL Images and torch tensors are the two the guides name; each has a one-line answer,
    and a generic "convert to an Arrow type" would make the user go looking for it."""
    assert expected in _remedy([_element(module, name)])


def test_an_unknown_type_still_gets_a_usable_fix() -> None:
    assert "Arrow-native" in _remedy([_Thing()])


def test_the_description_names_the_element_type_not_the_container() -> None:
    """ "a list" is useless; "a sequence of PIL.Image.Image" is the diagnosis."""
    assert _sample_type([_element("PIL.Image", "Image")]) == "a sequence of PIL.Image.Image"


def test_a_convertible_column_is_untouched() -> None:
    """The guard runs only on the failure path — a normal UDF must be unaffected."""
    ds = bt.from_pydict({"x": [1, 2, 3]})
    out = ds.map_batches(
        lambda b: {"doubled": [v.as_py() * 2 for v in b.column("x")]}, output_columns=["doubled"]
    ).to_pydict()
    assert out == {"doubled": [2, 4, 6]}


def test_a_numpy_tensor_column_still_works() -> None:
    """Multi-dim NumPy is the common ML return and must keep taking the tensor path."""
    import numpy as np

    ds = bt.from_pydict({"x": [1, 2]})
    out = ds.map_batches(
        lambda b: {"emb": np.ones((b.num_rows, 4), dtype="float32")}, output_columns=["emb"]
    ).collect()
    assert out.num_rows == 2


# --- the mixed-resolution case ---------------------------------------------------------
# Mixed-resolution images are the common multimodal shape, and Arrow's canonical tensor type
# needs one shape for the whole column. This used to be a hard stop, diagnosed but not
# solved. It is now carried as a variable-shape tensor column
# (`io.formats.ml.ragged`), so the tests below assert it *works*; the message they replaced
# survives in `interop.diagnostics` for the shapes that genuinely still cannot be typed.


def test_mixed_resolution_arrays_are_carried_rather_than_rejected() -> None:
    import numpy as np

    out = (
        bt.from_pydict({"id": [1, 2]})
        .map_batches(
            lambda b: {"img": [np.ones((2, 2), "uint8"), np.ones((3, 3), "uint8")]},
            output_columns=["img"],
        )
        .to_numpy()["img"]
    )
    assert [a.shape for a in out] == [(2, 2), (3, 3)]


def test_the_ragged_message_still_exists_for_a_shape_that_cannot_be_typed() -> None:
    """The advice it replaced told the caller to pass an ndarray, which they had done."""
    import numpy as np

    fix = _remedy([np.ones((2, 2)), np.ones((3, 3))])
    assert "Resize or pad" in fix
    assert "Convert it to an Arrow-native type" not in fix


def test_uniform_arrays_still_become_a_tensor_column() -> None:
    """Same-shaped arrays are the case that *works*, and must be unaffected."""
    import numpy as np

    out = (
        bt.from_pydict({"id": [1, 2]})
        .map_batches(lambda b: {"img": np.ones((2, 4), dtype="uint8")}, output_columns=["img"])
        .collect()
    )
    assert out.num_rows == 2


def test_columns_of_different_lengths_are_named_individually() -> None:
    """pyarrow says "Arrays were not all the same length: 1 vs 2", which for a wide result
    identifies neither column. The short one is usually the one built from a partial result."""
    ds = bt.from_pydict({"id": [1, 2]})
    with pytest.raises(PlanError, match=r"a=2, b=1, c=2"):
        ds.map_batches(
            lambda batch: {"a": [1, 2], "b": [1], "c": [1, 2]},
            output_columns=["a", "b", "c"],
        ).collect()


def test_equal_lengths_are_not_reported_as_a_length_problem() -> None:
    """The guard must not claim a length mismatch for a column that is simply un-typable."""
    ds = bt.from_pydict({"id": [1, 2]})
    with pytest.raises(PlanError, match="Arrow cannot represent"):
        ds.map_batches(
            lambda batch: {"a": [1, 2], "b": [_Thing(), _Thing()]},
            output_columns=["a", "b"],
        ).collect()
