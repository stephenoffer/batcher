"""The rigid-body functions `test_diff_spatial.py` does not reach, against textbook formulas.

That file covers the rotation, transform, product, Euler and slerp families component by
component. What it leaves unexercised is the matrix-to-quaternion constructor, the inverse
*rotation* (as opposed to the inverse transform), `pose_interpolate`, and the dict-returning
forms of `quat_inverse`, `quat_normalize` and `se3_inverse_transform`.

As there, the oracle is an independent formulation rather than the implementation typed
twice: the rotation matrix is built from the quaternion here and its transpose is the
inverse rotation; `quat_from_rotmat` is checked by feeding it the matrix of a known rotation
and asking for that rotation back, which is exactly the round trip the half-turn cases break
in the naive trace formula; interpolation is linear in translation and checked against the
separately tested `quat_slerp` in rotation.
"""

from __future__ import annotations

import math

import pytest

import batcher as bt

pytestmark = pytest.mark.differential


def _axis_angle(ax: float, ay: float, az: float, angle: float) -> tuple[float, float, float, float]:
    s, c = math.sin(angle / 2), math.cos(angle / 2)
    return (ax * s, ay * s, az * s, c)


_R3 = 1.0 / math.sqrt(3.0)

#: Unit rotations, including both half turns: at a half turn `1 + trace` is zero, so a
#: matrix-to-quaternion conversion that only uses the trace branch divides by zero there.
ROTATIONS = [
    (0.0, 0.0, 0.0, 1.0),
    _axis_angle(1.0, 0.0, 0.0, 0.7),
    _axis_angle(0.0, 1.0, 0.0, -1.2),
    _axis_angle(0.0, 0.0, 1.0, 2.5),
    _axis_angle(1.0, 0.0, 0.0, math.pi),
    _axis_angle(0.0, 1.0, 0.0, math.pi),
    _axis_angle(0.0, 0.0, 1.0, math.pi),
    _axis_angle(_R3, _R3, _R3, 1.9),
    _axis_angle(0.6, 0.8, 0.0, -2.8),
]

POINTS = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 2.0, 3.0), (-4.5, 0.25, 17.0)]

_Q = ("qx", "qy", "qz", "qw")
_P = ("px", "py", "pz")
_M = ("m00", "m01", "m02", "m10", "m11", "m12", "m20", "m21", "m22")


def _matrix(q: tuple[float, ...]) -> list[list[float]]:
    """The rotation matrix of a unit quaternion, from the textbook component formula."""
    x, y, z, w = q
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]


def _collect(columns: dict[str, list], exprs: dict) -> dict[str, list]:
    return bt.from_pydict(columns).select(**exprs).collect(distributed=False).to_pydict()


def test_quat_from_rotmat_recovers_the_rotation_that_built_the_matrix():
    """Matrix of q in, q out -- up to sign, since q and -q are the same rotation."""
    rows = [(*[v for r in _matrix(q) for v in r],) for q in ROTATIONS]
    cols = {name: [r[i] for r in rows] for i, name in enumerate(_M)}
    names = ("x", "y", "z", "w")
    exprs = {c: getattr(bt, f"quat_from_rotmat_{c}")(*_M) for c in names}
    got = _collect(cols, exprs)
    for i, q in enumerate(ROTATIONS):
        back = tuple(got[c][i] for c in names)
        dot = sum(a * b for a, b in zip(q, back, strict=True))
        assert abs(dot) == pytest.approx(1.0, abs=1e-12), (q, back)
        assert math.fsum(v * v for v in back) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("component", ["x", "y", "z"])
def test_quat_inverse_rotate_is_the_transposed_rotation_matrix(component):
    cases = [(q, p) for q in ROTATIONS for p in POINTS]
    cols = {n: [q[i] for q, _ in cases] for i, n in enumerate(_Q)}
    cols |= {n: [p[i] for _, p in cases] for i, n in enumerate(_P)}
    fn = getattr(bt, f"quat_inverse_rotate_{component}")
    got = _collect(cols, {"v": fn(*_Q, *_P)})["v"]
    k = "xyz".index(component)
    for (q, p), g in zip(cases, got, strict=True):
        r = _matrix(q)
        want = sum(r[j][k] * p[j] for j in range(3))  # row k of R^T is column k of R
        assert g == pytest.approx(want, rel=1e-12, abs=1e-12), (q, p)


def test_inverse_rotate_undoes_rotate():
    cases = [(q, p) for q in ROTATIONS for p in POINTS]
    cols = {n: [q[i] for q, _ in cases] for i, n in enumerate(_Q)}
    cols |= {n: [p[i] for _, p in cases] for i, n in enumerate(_P)}
    rotated = {f"r{c}": getattr(bt, f"quat_rotate_{c}")(*_Q, *_P) for c in "xyz"}
    ds = bt.from_pydict(cols).with_columns(**rotated)
    back = {c: getattr(bt, f"quat_inverse_rotate_{c}")(*_Q, "rx", "ry", "rz") for c in "xyz"}
    got = ds.select(**back).collect(distributed=False).to_pydict()
    for i, (_, p) in enumerate(cases):
        for k, c in enumerate("xyz"):
            assert got[c][i] == pytest.approx(p[k], rel=1e-12, abs=1e-12)


_A = ("atx", "aty", "atz", "aqx", "aqy", "aqz", "aqw")
_B = ("btx", "bty", "btz", "bqx", "bqy", "bqz", "bqw")


def _pose_columns(t_values: list[float]) -> dict[str, list]:
    a = (1.0, -2.0, 0.5, *ROTATIONS[1])
    b = (5.0, 2.0, -1.5, *ROTATIONS[7])
    n = len(t_values)
    cols = {name: [a[i]] * n for i, name in enumerate(_A)}
    cols |= {name: [b[i]] * n for i, name in enumerate(_B)}
    cols["t"] = t_values
    return cols


def test_pose_interpolate_is_linear_in_translation_and_slerp_in_rotation():
    ts = [0.0, 0.25, 0.5, 1.0, 1.5]  # 1.5: the docstring promises extrapolation, not a clamp
    cols = _pose_columns(ts)
    exprs = bt.pose_interpolate(_A, _B, "t")
    exprs |= bt.quat_slerp(_A[3:], _B[3:], "t", prefix="s_")
    got = _collect(cols, exprs)
    for i, t in enumerate(ts):
        for k, axis in enumerate(("tx", "ty", "tz")):
            a, b = cols[_A[k]][i], cols[_B[k]][i]
            assert got[axis][i] == pytest.approx(a + t * (b - a), rel=1e-12, abs=1e-12), t
        pose_q = [got[c][i] for c in ("qx", "qy", "qz", "qw")]
        slerp_q = [got[f"s_q{c}"][i] for c in "xyzw"]
        assert pose_q == pytest.approx(slerp_q, rel=1e-12, abs=1e-12), t


def test_pose_interpolate_hits_both_endpoints():
    got = _collect(_pose_columns([0.0, 1.0]), bt.pose_interpolate(_A, _B, "t", prefix="p_"))
    cols = _pose_columns([0.0, 1.0])
    for i, src in enumerate((_A, _B)):
        want = [cols[n][i] for n in src]
        have = [got[f"p_{c}"][i] for c in ("tx", "ty", "tz", "qx", "qy", "qz", "qw")]
        assert have[:3] == pytest.approx(want[:3], abs=1e-12)
        dot = sum(a * b for a, b in zip(have[3:], want[3:], strict=True))
        assert abs(dot) == pytest.approx(1.0, abs=1e-12)


def test_the_dict_forms_equal_their_component_functions():
    """`quat_inverse`/`quat_normalize`/`se3_inverse_transform` are the per-axis functions."""
    q = (0.0, 0.0, 3.0, 4.0)  # non-unit on purpose: both forms must normalize it
    cols = {n: [q[i]] for i, n in enumerate(_Q)} | {
        n: [v] for n, v in zip(_P, POINTS[2], strict=True)
    }
    cols |= {"tx": [1.0], "ty": [2.0], "tz": [3.0]}
    pose = ("tx", "ty", "tz", *_Q)
    exprs = bt.quat_inverse(_Q, prefix="i_") | bt.quat_normalize(_Q, prefix="n_")
    exprs |= bt.se3_inverse_transform(pose, _P, prefix="e_")
    for c in "xyzw":
        exprs[f"ci_{c}"] = getattr(bt, f"quat_inverse_{c}")(*_Q)
        exprs[f"cn_{c}"] = getattr(bt, f"quat_normalize_{c}")(*_Q)
    for c in "xyz":
        exprs[f"ce_{c}"] = getattr(bt, f"se3_inverse_transform_{c}")(*pose, *_P)
    got = _collect(cols, exprs)
    for c in "xyzw":
        assert got[f"i_q{c}"] == got[f"ci_{c}"]
        assert got[f"n_q{c}"] == got[f"cn_{c}"]
    for c in "xyz":
        assert got[f"e_{c}"] == got[f"ce_{c}"]
    # And the normalization is real: a (0, 0, 3, 4) input comes back at unit length.
    assert math.fsum(got[f"n_q{c}"][0] ** 2 for c in "xyzw") == pytest.approx(1.0)


def test_a_null_or_empty_input_propagates():
    # Row 0 is all null; row 1 is the identity matrix, whose rotation has w == 1.
    cols = {n: [None, 0.0] for n in _M}
    cols |= {"m00": [None, 1.0], "m11": [None, 1.0], "m22": [None, 1.0]}
    got = _collect(cols, {"w": bt.quat_from_rotmat_w(*_M)})["w"]
    assert got[0] is None and got[1] == pytest.approx(1.0)
    empty = {n: [] for n in (*_Q, *_P)}
    out = _collect(empty, {"v": bt.quat_inverse_rotate_x(*_Q, *_P)})
    assert out["v"] == []
