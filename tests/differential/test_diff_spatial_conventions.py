"""Euler sequences, matrix repair and canonical signs, against SciPy's `Rotation`.

DuckDB has no rotation functions, so the oracle is `scipy.spatial.transform.Rotation`, an
independent implementation: `as_euler` / `from_euler` for every one of the 24 intrinsic
and extrinsic axis sequences, `from_matrix` (an SVD, where Batcher uses the quaternion
eigenvector method) for the nearest rotation to a drifted matrix, and
``as_quat(canonical=True)`` for the canonical sign. The residual measures are checked
against NumPy and against the strict reader they describe.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pyarrow as pa
import pytest

import batcher as bt
from batcher import PlanError, col

pytestmark = pytest.mark.differential

_Rotation = pytest.importorskip("scipy.spatial.transform").Rotation

_AXES = ["".join(p) for p in itertools.product("xyz", repeat=3) if p[0] != p[1] and p[1] != p[2]]
SEQUENCES = _AXES + [s.upper() for s in _AXES]
assert len(SEQUENCES) == 24

_RNG = np.random.default_rng(20260928)
QUATS = _Rotation.random(40, random_state=7).as_quat()
_Q = ("qx", "qy", "qz", "qw")
_M = tuple(f"m{i}{j}" for i in range(3) for j in range(3))


def _quat_ds(quats: np.ndarray) -> bt.Dataset:
    return bt.from_pydict({n: quats[:, i].tolist() for i, n in enumerate(_Q)})


def _same_rotation(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs(np.sum(a * b, axis=1)) > 1 - 1e-10


@pytest.mark.parametrize("sequence", SEQUENCES)
def test_reading_angles_matches_scipy(sequence):
    got = _quat_ds(QUATS).select(**bt.quat_to_euler_seq(_Q, sequence=sequence)).to_pydict()
    angles = np.column_stack([got["angle_1"], got["angle_2"], got["angle_3"]])
    want = _Rotation.from_quat(QUATS).as_euler(sequence)
    np.testing.assert_allclose(angles, want, atol=1e-9)


@pytest.mark.parametrize("sequence", SEQUENCES)
def test_building_from_angles_matches_scipy(sequence):
    angles = _RNG.uniform(-math.pi, math.pi, size=(40, 3))
    ds = bt.from_pydict({f"a{i}": angles[:, i].tolist() for i in range(3)})
    got = ds.select(**bt.quat_from_euler_seq(("a0", "a1", "a2"), sequence=sequence))
    q = np.column_stack([got.to_pydict()[n] for n in _Q])
    want = _Rotation.from_euler(sequence, angles).as_quat()
    assert _same_rotation(q, want).all()


def test_the_default_sequence_is_the_familys_roll_pitch_yaw():
    ds = _quat_ds(QUATS)
    seq = ds.select(
        yaw=bt.quat_to_euler_seq_first(*_Q),
        pitch=bt.quat_to_euler_seq_second(*_Q),
        roll=bt.quat_to_euler_seq_third(*_Q),
    ).to_pydict()
    zyx = ds.select(**bt.quat_to_euler(_Q)).to_pydict()
    for name in ("yaw", "pitch", "roll"):
        np.testing.assert_allclose(seq[name], zyx[name], atol=1e-9)


@pytest.mark.parametrize("sequence", ["ZYX", "xyz", "ZXZ", "yzy"])
def test_gimbal_lock_zeroes_the_third_angle_like_scipy(sequence):
    middle = 0.0 if sequence[0].lower() == sequence[2].lower() else math.pi / 2
    q = _Rotation.from_euler(sequence, [[0.7, middle, 0.4]]).as_quat()
    got = _quat_ds(q).select(**bt.quat_to_euler_seq(_Q, sequence=sequence)).to_pydict()
    assert got["angle_3"] == [0.0]
    back = _Rotation.from_euler(sequence, [[got["angle_1"][0], got["angle_2"][0], 0.0]])
    assert _same_rotation(back.as_quat(), q).all()


@pytest.mark.parametrize("bad", ["ZY", "XXY", "Zyx", "abc", "ZYXZ"])
def test_a_sequence_that_names_no_sequence_is_refused(bad):
    with pytest.raises(PlanError, match="Euler sequence"):
        bt.quat_to_euler_seq(_Q, sequence=bad)


def _matrix_ds(mats: np.ndarray) -> bt.Dataset:
    flat = mats.reshape(len(mats), 9)
    return bt.from_pydict({n: flat[:, i].tolist() for i, n in enumerate(_M)})


def test_the_nearest_rotation_matches_scipys_svd_repair():
    exact = _Rotation.from_quat(QUATS).as_matrix()
    drifted = exact + _RNG.normal(scale=0.02, size=exact.shape)
    got = _matrix_ds(drifted).select(**bt.quat_from_rotmat_nearest(_M)).to_pydict()
    q = np.column_stack([got[n] for n in _Q])
    want = _Rotation.from_matrix(drifted).as_quat()
    assert _same_rotation(q, want).all()
    assert (q[:, 3] >= 0).all()
    # The strict reader refuses every one of these; the repair is what makes them usable.
    strict = _matrix_ds(drifted).select(w=bt.quat_from_rotmat_w(*_M)).to_pydict()["w"]
    assert strict == [None] * len(drifted)


def test_the_nearest_rotation_refuses_a_reflection_and_nulls_a_null():
    mats = np.stack([np.diag([1.0, 1.0, -1.0]), np.zeros((3, 3)), np.eye(3)])
    ds = _matrix_ds(mats).with_columns(m00=bt.when(col("m22") != 1.0).then(col("m00")))
    got = ds.select(w=bt.quat_from_rotmat_nearest_w(*_M)).to_pydict()["w"]
    # Row 0 is a reflection, row 1 the zero matrix, and row 2's m00 was nulled.
    assert got == [None, None, None]


def test_the_residuals_are_exactly_what_the_strict_reader_checks():
    exact = _Rotation.from_quat(QUATS[:10]).as_matrix()
    scales = np.linspace(0.99995, 1.0003, 10)
    mats = exact * scales[:, None, None]
    got = (
        _matrix_ds(mats)
        .select(
            err=bt.rotmat_orthogonality_error(_M),
            det=bt.rotmat_determinant(_M),
            w=bt.quat_from_rotmat_w(*_M),
        )
        .to_pydict()
    )
    gram = mats @ np.transpose(mats, (0, 2, 1)) - np.eye(3)
    np.testing.assert_allclose(got["err"], np.abs(gram).max(axis=(1, 2)), atol=1e-12)
    np.testing.assert_allclose(got["det"], np.linalg.det(mats), atol=1e-12)
    accepted = [w is not None for w in got["w"]]
    predicted = [
        e <= 1e-4 and abs(d - 1.0) <= 1e-4 for e, d in zip(got["err"], got["det"], strict=True)
    ]
    assert accepted == predicted
    assert any(accepted) and not all(accepted)


def test_canonicalize_matches_scipys_canonical_form():
    signs = _RNG.choice([-1.0, 1.0], size=(len(QUATS), 1))
    # Include every tie-break: w == 0, then x == 0, then y == 0.
    ties = np.array([[0.6, 0.0, 0.8, 0.0], [-0.6, 0.0, 0.8, 0.0], [0.0, -1.0, 0.0, 0.0]])
    quats = np.vstack([QUATS * signs * 3.0, ties, -ties])
    got = _quat_ds(quats).select(**bt.quat_canonicalize(_Q)).to_pydict()
    q = np.column_stack([got[n] for n in _Q])
    want = _Rotation.from_quat(quats).as_quat(canonical=True)
    np.testing.assert_allclose(q, want, atol=1e-12)


def test_canonicalize_makes_a_rotation_group_as_one():
    q = QUATS[:1]
    ds = _quat_ds(np.vstack([q, -q]))
    canon = ds.select(**bt.quat_canonicalize(_Q))
    assert canon.group_by(*_Q).agg(n=col("qw").count()).to_pydict()["n"] == [2]


def test_nulls_empties_and_streaming():
    ds = bt.from_pydict({"qx": [0.0, None], "qy": [0.0, 0.0], "qz": [0.0, 0.0], "qw": [1.0, 1.0]})
    got = ds.select(**bt.quat_to_euler_seq(_Q, sequence="ZXZ")).to_pydict()
    assert got["angle_1"][1] is None and got["angle_1"][0] == 0.0
    schema = pa.schema([(n, pa.float64()) for n in _Q])
    empty = bt.from_pydict({n: [] for n in _Q}, schema=schema)
    assert empty.select(**bt.quat_canonicalize(_Q)).to_pydict()["qw"] == []
    big = _quat_ds(np.vstack([QUATS] * 30))
    query = big.select(**bt.quat_to_euler_seq(_Q, sequence="yxz"))
    streamed = [
        v for b in query.iter_batches(batch_size=97) for v in b.column("angle_2").to_pylist()
    ]
    assert streamed == query.to_pydict()["angle_2"]


def test_sql_names_the_sequence_as_a_string_literal():
    ds = _quat_ds(QUATS[:5])
    sql = bt.sql("SELECT quat_to_euler_seq_second(qx, qy, qz, qw, 'zxz') AS v FROM t", t=ds)
    api = ds.select(v=bt.quat_to_euler_seq_second(*_Q, sequence="zxz"))
    assert sql.to_pydict() == api.to_pydict()
