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
from batcher.core.udf.call import _fix_for, _sample_type

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
    assert expected in _fix_for([_element(module, name)])


def test_an_unknown_type_still_gets_a_usable_fix() -> None:
    assert "Arrow-native" in _fix_for([_Thing()])


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
# Arrow has no variable-shape tensor type, so a column of differently-shaped arrays cannot
# be typed at all. The generic "convert it to an ndarray" advice is *actively wrong* there:
# the caller already passed ndarrays. Mixed-resolution images are the common multimodal
# shape, and the guides flag it as `ArrowTypeError` on write.


def test_ragged_arrays_are_diagnosed_as_a_shape_problem() -> None:
    import numpy as np

    ds = bt.from_pydict({"id": [1, 2]})
    with pytest.raises(PlanError, match="different shapes"):
        ds.map_batches(
            lambda b: {"img": [np.ones((2, 2), "uint8"), np.ones((3, 3), "uint8")]},
            output_columns=["img"],
        ).collect()


def test_the_ragged_message_names_both_shapes() -> None:
    """ "they differ" is a restatement; "(2, 2) and (3, 3)" is the diagnosis."""
    import numpy as np

    ds = bt.from_pydict({"id": [1, 2]})
    with pytest.raises(PlanError, match=r"\(2, 2\).*\(3, 3\)"):
        ds.map_batches(
            lambda b: {"img": [np.ones((2, 2), "uint8"), np.ones((3, 3), "uint8")]},
            output_columns=["img"],
        ).collect()


def test_the_ragged_message_does_not_tell_you_to_pass_an_ndarray() -> None:
    """The bug this fixes: the generic advice told the caller to do what they had done."""
    import numpy as np

    from batcher.core.udf.call import _fix_for

    fix = _fix_for([np.ones((2, 2)), np.ones((3, 3))])
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
