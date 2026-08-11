"""The rigid-body surface against DuckDB, used as an independent reimplementation.

DuckDB has no quaternion functions, so unlike the geospatial suite there is no ported
library to compare against. What there is, and what makes this a real differential test
rather than a restatement, is that every function in this family is a closed-form
expression in ordinary arithmetic — so the oracle is that formula, written out as SQL,
from the *textbook* rather than from the Rust.

That independence is not cosmetic. Each SQL reference below is deliberately a different
formulation from the implementation it checks:

* rotation is written as the 3x3 **rotation matrix** built from the quaternion's
  components, while `bc_spatial::Quat::rotate` uses the two-cross-product form;
* `quat_angle` is `2 * acos(|w|)`, while the implementation uses `atan2` of the vector
  and scalar parts;
* `quat_angular_distance` is `2 * acos(|dot|)`, while the implementation uses the
  chord-length `atan2` form.

Two formulations that agree to the last few bits on a hundred inputs agree because the
mathematics is right, not because the same expression was typed twice.

Values near the poles of the `acos` formulations are excluded from the comparisons that
use them and covered by invariants instead, which is stated at each site: `acos` loses
about half its significant digits as its argument approaches one, so agreeing there
would be agreeing to a tolerance wide enough to hide a real error.
"""

from __future__ import annotations

import itertools
import math

import pytest

import batcher as bt

pytestmark = pytest.mark.differential


def _axis_angle(ax: float, ay: float, az: float, angle: float) -> tuple[float, ...]:
    """A quaternion for `angle` radians about the (already unit) axis."""
    s, c = math.sin(angle / 2), math.cos(angle / 2)
    return (ax * s, ay * s, az * s, c)


_ROOT3 = 1.0 / math.sqrt(3.0)

#: Rotations chosen to separate the cases that behave differently: the identity, small
#: angles, each principal axis, a half turn (where the naive matrix-to-quaternion
#: formula divides by zero), a tilted axis, and a deliberately non-unit quaternion that
#: every function must normalize rather than scale by.
ROTATIONS: list[tuple[float, ...]] = [
    (0.0, 0.0, 0.0, 1.0),
    _axis_angle(0.0, 0.0, 1.0, 0.001),
    _axis_angle(1.0, 0.0, 0.0, 0.7),
    _axis_angle(0.0, 1.0, 0.0, -1.2),
    _axis_angle(0.0, 0.0, 1.0, 2.5),
    _axis_angle(1.0, 0.0, 0.0, math.pi),
    _axis_angle(0.0, 0.0, 1.0, math.pi),
    _axis_angle(_ROOT3, _ROOT3, _ROOT3, 1.9),
    _axis_angle(0.6, 0.8, 0.0, -2.8),
    (0.0, 0.0, 3.0, 4.0),
    (-0.2, 0.4, -0.6, 0.8),
]

#: Points chosen to include the origin, each axis, a negative octant, and magnitudes
#: spanning the range a lidar sweep actually contains.
POINTS: list[tuple[float, float, float]] = [
    (0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
    (1.0, 2.0, 3.0),
    (-4.5, 0.25, 17.0),
    (120.0, -80.0, -1.5),
]

#: Translations, including none at all.
TRANSLATIONS: list[tuple[float, float, float]] = [
    (0.0, 0.0, 0.0),
    (10.0, 0.0, 0.0),
    (-3.0, 7.5, 0.5),
    (1000.0, 2000.0, -30.0),
]

_POSE_CASES = [(t, q, p) for t in TRANSLATIONS for q in ROTATIONS for p in POINTS]

_ROT_CASES = [(q, p) for q in ROTATIONS for p in POINTS]

# --- The SQL oracle -----------------------------------------------------------------
#
# `u*` are the normalized components; `m<r><c>` is the rotation matrix built from them.
# Written as the matrix rather than the cross-product form the engine uses, so the two
# are independent statements of the same rotation.
_UNIT = """
    WITH n AS (SELECT sqrt(qx*qx + qy*qy + qz*qz + qw*qw) AS len),
    u AS (
        SELECT qx/len AS ux, qy/len AS uy, qz/len AS uz, qw/len AS uw FROM n
    )
"""

_MATRIX = """
    m AS (
        SELECT
            1 - 2*(uy*uy + uz*uz) AS m00, 2*(ux*uy - uz*uw) AS m01, 2*(ux*uz + uy*uw) AS m02,
            2*(ux*uy + uz*uw) AS m10, 1 - 2*(ux*ux + uz*uz) AS m11, 2*(uy*uz - ux*uw) AS m12,
            2*(ux*uz - uy*uw) AS m20, 2*(uy*uz + ux*uw) AS m21, 1 - 2*(ux*ux + uy*uy) AS m22
        FROM u
    )
"""


_ROWS = {"x": ("m00", "m01", "m02"), "y": ("m10", "m11", "m12"), "z": ("m20", "m21", "m22")}
_COLS = {"x": ("m00", "m10", "m20"), "y": ("m01", "m11", "m21"), "z": ("m02", "m12", "m22")}


def _rotate_sql(component: str, *, offset: str = "") -> str:
    """SQL for one component of `p` rotated by `q`, via the rotation matrix."""
    a, b, c = _ROWS[component]
    return f"{_UNIT}, {_MATRIX} SELECT {a}*px + {b}*py + {c}*pz{offset} FROM m"


def _inverse_rotate_sql(component: str) -> str:
    """SQL for one component of the *translated* point under the transposed matrix."""
    a, b, c = _COLS[component]
    return f"{_UNIT}, {_MATRIX} SELECT {a}*(px-tx) + {b}*(py-ty) + {c}*(pz-tz) FROM m"


def _scalar(duck, sql: str, binds: dict[str, float]) -> float:
    """Evaluate one reference expression with its named inputs substituted in.

    Substitution rather than parameter binding because the references are CTEs that
    name each input several times, and DuckDB binds positionally: a seven-way repeat
    would mean seven copies of every value in the right order, which is precisely the
    kind of bookkeeping this suite exists to catch errors in.
    """
    body = sql
    # Longest first, so `qw` cannot be eaten by a shorter name that prefixes it.
    for name in sorted(binds, key=len, reverse=True):
        # `::DOUBLE` because DuckDB reads a bare decimal literal as DECIMAL, and an
        # expression over decimals returns one — which then will not subtract from a
        # float. The engine's answer is a double, so the reference must be too.
        body = body.replace(name, f"({binds[name]!r}::DOUBLE)")
    return duck.execute(body).fetchone()[0]


def _batcher(expr, columns: dict[str, list[float]]) -> list:
    """Evaluate one Batcher expression over a column mapping."""
    return bt.from_pydict(columns).select(v=expr).to_pydict()["v"]


def _columns(cases: list[tuple], names: list[str]) -> dict[str, list[float]]:
    """Transpose a list of flat case tuples into a column mapping."""
    return {name: [float(case[i]) for case in cases] for i, name in enumerate(names)}


_Q = ["qx", "qy", "qz", "qw"]
_P = ["px", "py", "pz"]
_T = ["tx", "ty", "tz"]


@pytest.mark.parametrize("component", ["x", "y", "z"])
def test_quat_rotate_matches_the_rotation_matrix(duck, component):
    """Rotating a vector agrees with the matrix form, for every rotation and point."""
    flat = [(*q, *p) for q, p in _ROT_CASES]
    cols = _columns(flat, _Q + _P)
    got = _batcher(getattr(bt, f"quat_rotate_{component}")(*_Q, *_P), cols)
    sql = _rotate_sql(component)
    want = [_scalar(duck, sql, dict(zip(_Q + _P, row, strict=True))) for row in flat]
    for row, g, w in zip(flat, got, want, strict=True):
        assert g == pytest.approx(w, rel=1e-12, abs=1e-12), row


@pytest.mark.parametrize("component", ["x", "y", "z"])
def test_se3_transform_matches_rotate_then_translate(duck, component):
    """A pose applied to a point agrees with the matrix rotation plus the translation."""
    flat = [(*t, *q, *p) for t, q, p in _POSE_CASES]
    cols = _columns(flat, _T + _Q + _P)
    got = _batcher(getattr(bt, f"se3_transform_{component}")(*_T, *_Q, *_P), cols)
    sql = _rotate_sql(component, offset=f" + t{component}")
    want = [_scalar(duck, sql, dict(zip(_T + _Q + _P, row, strict=True))) for row in flat]
    for row, g, w in zip(flat, got, want, strict=True):
        assert g == pytest.approx(w, rel=1e-12, abs=1e-12), row


@pytest.mark.parametrize("component", ["x", "y", "z"])
def test_se3_inverse_transform_matches_translate_then_inverse_rotate(duck, component):
    """The inverse pose agrees with subtracting first and applying the transposed matrix."""
    flat = [(*t, *q, *p) for t, q, p in _POSE_CASES]
    cols = _columns(flat, _T + _Q + _P)
    got = _batcher(getattr(bt, f"se3_inverse_transform_{component}")(*_T, *_Q, *_P), cols)
    # The inverse rotation is the transpose, so a *column* of the matrix is taken
    # rather than a row, over the translated point.
    sql = _inverse_rotate_sql(component)
    want = [_scalar(duck, sql, dict(zip(_T + _Q + _P, row, strict=True))) for row in flat]
    for row, g, w in zip(flat, got, want, strict=True):
        assert g == pytest.approx(w, rel=1e-11, abs=1e-11), row


_A = ["ax", "ay", "az", "aw"]
_B = ["bx", "by", "bz", "bw"]

_HAMILTON = {
    "x": "aw*bx + ax*bw + ay*bz - az*by",
    "y": "aw*by - ax*bz + ay*bw + az*bx",
    "z": "aw*bz + ax*by - ay*bx + az*bw",
    "w": "aw*bw - ax*bx - ay*by - az*bz",
}


@pytest.mark.parametrize("component", ["x", "y", "z", "w"])
def test_quat_multiply_matches_the_hamilton_product(duck, component):
    """Composing two rotations agrees with the Hamilton product written out."""
    pairs = [(a, b) for a in ROTATIONS for b in ROTATIONS]
    flat = [(*a, *b) for a, b in pairs]
    cols = _columns(flat, _A + _B)
    got = _batcher(getattr(bt, f"quat_multiply_{component}")(*_A, *_B), cols)
    sql = f"SELECT {_HAMILTON[component]}"
    want = [_scalar(duck, sql, dict(zip(_A + _B, row, strict=True))) for row in flat]
    for row, g, w in zip(flat, got, want, strict=True):
        assert g == pytest.approx(w, rel=1e-12, abs=1e-12), row


def test_quat_norm_matches_the_four_component_length(duck):
    cols = _columns(ROTATIONS, _Q)
    got = _batcher(bt.quat_norm(*_Q), cols)
    sql = "SELECT sqrt(qx*qx + qy*qy + qz*qz + qw*qw)"
    want = [_scalar(duck, sql, dict(zip(_Q, q, strict=True))) for q in ROTATIONS]
    assert got == pytest.approx(want, rel=1e-14)


@pytest.mark.parametrize("component", ["x", "y", "z", "w"])
def test_quat_normalize_matches_dividing_by_the_length(duck, component):
    cols = _columns(ROTATIONS, _Q)
    got = _batcher(getattr(bt, f"quat_normalize_{component}")(*_Q), cols)
    sql = f"SELECT q{component} / sqrt(qx*qx + qy*qy + qz*qz + qw*qw)"
    want = [_scalar(duck, sql, dict(zip(_Q, q, strict=True))) for q in ROTATIONS]
    assert got == pytest.approx(want, rel=1e-14)


@pytest.mark.parametrize("component", ["x", "y", "z", "w"])
def test_quat_inverse_negates_the_vector_part_of_the_unit_rotation(duck, component):
    cols = _columns(ROTATIONS, _Q)
    got = _batcher(getattr(bt, f"quat_inverse_{component}")(*_Q), cols)
    sign = "" if component == "w" else "-"
    sql = f"SELECT {sign}q{component} / sqrt(qx*qx + qy*qy + qz*qz + qw*qw)"
    want = [_scalar(duck, sql, dict(zip(_Q, q, strict=True))) for q in ROTATIONS]
    assert got == pytest.approx(want, rel=1e-14)


_EULER_SQL = {
    "roll": f"{_UNIT} SELECT atan2(2*(uw*ux + uy*uz), 1 - 2*(ux*ux + uy*uy)) FROM u",
    "pitch": f"{_UNIT} SELECT asin(greatest(-1, least(1, 2*(uw*uy - uz*ux)))) FROM u",
    "yaw": f"{_UNIT} SELECT atan2(2*(uw*uz + ux*uy), 1 - 2*(uy*uy + uz*uz)) FROM u",
}


@pytest.mark.parametrize("angle", ["roll", "pitch", "yaw"])
def test_quat_to_euler_matches_the_textbook_zyx_decomposition(duck, angle):
    """Each Euler angle agrees with the standard formula, away from gimbal lock.

    None of `ROTATIONS` sits at a pole, which is deliberate: there the decomposition is
    not unique and no two implementations need agree on which of the equivalent splits
    they report. That case is covered by an invariant instead — see
    `test_gimbal_lock_still_round_trips_to_the_same_rotation`.
    """
    cols = _columns(ROTATIONS, _Q)
    got = _batcher(getattr(bt, f"quat_to_{angle}")(*_Q), cols)
    want = [_scalar(duck, _EULER_SQL[angle], dict(zip(_Q, q, strict=True))) for q in ROTATIONS]
    for q, g, w in zip(ROTATIONS, got, want, strict=True):
        assert g == pytest.approx(w, rel=1e-11, abs=1e-11), q


_EULERS = [
    (0.0, 0.0, 0.0),
    (0.1, -0.2, 0.3),
    (1.5, 0.9, -2.2),
    (-3.0, -1.4, 3.0),
    (0.7, 0.0, 0.0),
    (0.0, 0.0, math.pi),
]
_E = ["roll", "pitch", "yaw"]

_FROM_EULER_SQL = {
    "x": "sin(roll/2)*cos(pitch/2)*cos(yaw/2) - cos(roll/2)*sin(pitch/2)*sin(yaw/2)",
    "y": "cos(roll/2)*sin(pitch/2)*cos(yaw/2) + sin(roll/2)*cos(pitch/2)*sin(yaw/2)",
    "z": "cos(roll/2)*cos(pitch/2)*sin(yaw/2) - sin(roll/2)*sin(pitch/2)*cos(yaw/2)",
    "w": "cos(roll/2)*cos(pitch/2)*cos(yaw/2) + sin(roll/2)*sin(pitch/2)*sin(yaw/2)",
}


@pytest.mark.parametrize("component", ["x", "y", "z", "w"])
def test_quat_from_euler_matches_the_half_angle_formula(duck, component):
    cols = _columns(_EULERS, _E)
    got = _batcher(getattr(bt, f"quat_from_euler_{component}")(*_E), cols)
    sql = f"SELECT {_FROM_EULER_SQL[component]}"
    want = [_scalar(duck, sql, dict(zip(_E, e, strict=True))) for e in _EULERS]
    assert got == pytest.approx(want, rel=1e-13, abs=1e-14)


#: Rotations far enough from the identity that `2 * acos(|w|)` keeps its precision. The
#: near-identity ones are excluded here on purpose and covered by the Rust unit tests,
#: which check `angle` against the axis-angle it was built from at every scale.
_WELL_SEPARATED = [q for q in ROTATIONS if abs(q[3]) < 0.999]


def test_quat_angle_matches_twice_the_arccosine_of_the_scalar_part(duck):
    cols = _columns(_WELL_SEPARATED, _Q)
    got = _batcher(bt.quat_angle(*_Q), cols)
    sql = f"{_UNIT} SELECT 2*acos(least(1, abs(uw))) FROM u"
    want = [_scalar(duck, sql, dict(zip(_Q, q, strict=True))) for q in _WELL_SEPARATED]
    for q, g, w in zip(_WELL_SEPARATED, got, want, strict=True):
        assert g == pytest.approx(w, rel=1e-9, abs=1e-9), q


def test_quat_angular_distance_matches_twice_the_arccosine_of_the_dot(duck):
    """The geodesic angle agrees with the `acos` form for well-separated rotations."""
    pairs = [(a, b) for a in _WELL_SEPARATED for b in _WELL_SEPARATED if a != b]
    flat = [(*a, *b) for a, b in pairs]
    cols = _columns(flat, _A + _B)
    got = _batcher(bt.quat_angular_distance(*_A, *_B), cols)
    sql = """
        WITH na AS (SELECT sqrt(ax*ax+ay*ay+az*az+aw*aw) AS la),
             nb AS (SELECT sqrt(bx*bx+by*by+bz*bz+bw*bw) AS lb)
        SELECT 2*acos(least(1, abs(
            (ax/la)*(bx/lb) + (ay/la)*(by/lb) + (az/la)*(bz/lb) + (aw/la)*(bw/lb)
        ))) FROM na, nb
    """
    want = [_scalar(duck, sql, dict(zip(_A + _B, row, strict=True))) for row in flat]
    close = 0
    for row, g, w in zip(flat, got, want, strict=True):
        # The `acos` reference degrades as the two rotations approach each other; skip
        # the handful of pairs where it has lost too many digits to be an oracle, and
        # assert there are few enough of them that the test still means something.
        if w < 1e-4 or abs(w - math.pi) < 1e-4:
            close += 1
            continue
        assert g == pytest.approx(w, rel=1e-8, abs=1e-8), row
    assert close < len(flat) // 4, "too many pairs skipped for the oracle to be meaningful"


def test_norm_3d_matches_the_square_root_of_the_sum_of_squares(duck):
    cols = _columns(POINTS, _P)
    got = _batcher(bt.norm_3d(tuple(_P)), cols)
    sql = "SELECT sqrt(px*px + py*py + pz*pz)"
    want = [_scalar(duck, sql, dict(zip(_P, p, strict=True))) for p in POINTS]
    assert got == pytest.approx(want, rel=1e-14)


def test_distance_3d_matches_the_pairwise_euclidean_distance(duck):
    pairs = [(a, b) for a in POINTS for b in POINTS]
    flat = [(*a, *b) for a, b in pairs]
    names = ["ax", "ay", "az", "bx", "by", "bz"]
    cols = _columns(flat, names)
    got = _batcher(bt.distance_3d(("ax", "ay", "az"), ("bx", "by", "bz")), cols)
    sql = "SELECT sqrt((bx-ax)*(bx-ax) + (by-ay)*(by-ay) + (bz-az)*(bz-az))"
    want = [_scalar(duck, sql, dict(zip(names, row, strict=True))) for row in flat]
    assert got == pytest.approx(want, rel=1e-13, abs=1e-13)


def test_azimuth_and_elevation_match_the_spherical_conversion(duck):
    cols = _columns(POINTS, _P)
    az = _batcher(bt.azimuth_3d(tuple(_P)), cols)
    el = _batcher(bt.elevation_3d(tuple(_P)), cols)
    want_az = [_scalar(duck, "SELECT atan2(py, px)", dict(zip(_P, p, strict=True))) for p in POINTS]
    want_el = [
        _scalar(duck, "SELECT atan2(pz, sqrt(px*px + py*py))", dict(zip(_P, p, strict=True)))
        for p in POINTS
    ]
    assert az == pytest.approx(want_az, rel=1e-14, abs=1e-14)
    assert el == pytest.approx(want_el, rel=1e-14, abs=1e-14)


def test_voxel_index_matches_the_floor_of_the_scaled_coordinate(duck):
    """Binning agrees with `floor(c / size)` cast to an integer, on both sides of zero.

    The sign is the whole point. Truncation toward zero — which is what an integer cast
    without the floor does, and what a hand-written version usually does — makes the cell
    straddling the origin twice as wide as every other cell.
    """
    coords = [
        (0.0, 0.0, 0.0),
        (0.05, 0.15, 0.25),
        (-0.05, -0.15, -0.25),
        (-0.0, 0.199_999, -0.200_001),
        (1234.5, -9876.5, 0.0),
    ]
    cols = _columns(coords, _P)
    size = 0.2
    got = bt.from_pydict(cols).select(**bt.voxel_index(tuple(_P), size)).to_pydict()
    for axis in "xyz":
        want = [
            _scalar(
                duck,
                f"SELECT CAST(floor(p{axis} / {size}) AS BIGINT)",
                dict(zip(_P, row, strict=True)),
            )
            for row in coords
        ]
        assert got[f"i{axis}"] == want, axis


def test_voxel_index_makes_every_cell_the_same_width():
    """Two points a cell apart land in adjacent cells, wherever the origin is."""
    edge = 0.2
    # Cell *centres*, not boundaries. A point exactly on a boundary can land either side
    # of it once the coordinate has been through binary floating point, so a test built
    # on boundaries asserts something neither this nor any other implementation promises.
    xs = [(i + 0.5) * edge for i in range(-3, 3)]
    ds = bt.from_pydict({"px": xs, "py": [0.0] * 6, "pz": [0.0] * 6})
    idx = ds.select(**bt.voxel_index(tuple(_P), edge)).to_pydict()["ix"]
    assert idx == [-3, -2, -1, 0, 1, 2], idx
    steps = {b - a for a, b in itertools.pairwise(idx)}
    assert steps == {1}, idx


def test_voxel_downsampling_is_an_ordinary_grouped_aggregation():
    """The point of the function: one row per occupied cube, through `group_by`."""
    cloud = bt.from_pydict(
        {
            "px": [0.01, 0.02, 0.03, 5.00, 5.01],
            "py": [0.0] * 5,
            "pz": [0.0] * 5,
        }
    )
    thinned = (
        cloud.group_by(**bt.voxel_index(tuple(_P), 0.2))
        .agg(n=bt.count(), px=bt.col("px").mean())
        .to_pydict()
    )
    assert sorted(thinned["n"]) == [2, 3]
    assert len(thinned["px"]) == 2


# --- Invariants the oracle cannot express -------------------------------------------


def test_slerp_sweeps_the_angle_at_a_constant_rate():
    """Interpolation is spherical, not component-wise.

    The property that separates the two, and the reason a naively interpolated pose
    makes a lidar sweep bend. Stated as an invariant rather than against DuckDB because
    the SQL for the component-wise version would agree with the wrong answer.
    """
    end = _axis_angle(0.0, 0.0, 1.0, 1.2)
    fracs = [i / 10 for i in range(11)]
    cols = {
        "ax": [0.0] * 11, "ay": [0.0] * 11, "az": [0.0] * 11, "aw": [1.0] * 11,
        "bx": [end[0]] * 11, "by": [end[1]] * 11, "bz": [end[2]] * 11, "bw": [end[3]] * 11,
        "t": fracs,
    }  # fmt: skip
    mid = bt.quat_slerp(tuple(_A), tuple(_B), "t")
    ds = bt.from_pydict(cols).select(**mid)
    angle = ds.select(a=bt.quat_angle("qx", "qy", "qz", "qw")).to_pydict()["a"]
    for f, got in zip(fracs, angle, strict=True):
        assert got == pytest.approx(1.2 * f, abs=1e-9)


def test_gimbal_lock_still_round_trips_to_the_same_rotation():
    """At a pole the reported angles differ from the input, but the rotation does not.

    The only claim that survives gimbal lock, and the one worth pinning: the
    decomposition is not unique, so no implementation can return the angles you started
    with, but every implementation must return angles naming the same rotation.
    """
    ds = bt.from_pydict({"roll": [0.4], "pitch": [math.pi / 2], "yaw": [0.9]}).select(
        **bt.quat_from_euler("roll", "pitch", "yaw")
    )
    back = ds.select(**bt.quat_to_euler(("qx", "qy", "qz", "qw")))
    again = back.select(**bt.quat_from_euler("roll", "pitch", "yaw"))
    together = bt.from_pydict(
        {
            **{f"a{k}": v for k, v in ds.to_pydict().items()},
            **{f"b{k}": v for k, v in again.to_pydict().items()},
        }
    )
    dist = together.select(
        d=bt.quat_angular_distance("aqx", "aqy", "aqz", "aqw", "bqx", "bqy", "bqz", "bqw")
    ).to_pydict()["d"]
    assert dist[0] == pytest.approx(0.0, abs=1e-9)


def test_se3_compose_equals_applying_each_pose_in_turn():
    """The composed pose moves a point exactly where the two poses in sequence do."""
    outer = (2.0, -1.0, 0.5, *_axis_angle(0.0, 0.0, 1.0, 0.9))
    inner = (0.3, 4.0, -2.0, *_axis_angle(1.0, 0.0, 0.0, -1.4))
    o = ("ot0", "ot1", "ot2", "oq0", "oq1", "oq2", "oq3")
    i = ("it0", "it1", "it2", "iq0", "iq1", "iq2", "iq3")
    cols = {
        **{n: [v] for n, v in zip(o, outer, strict=True)},
        **{n: [v] for n, v in zip(i, inner, strict=True)},
        "px": [7.0],
        "py": [-3.0],
        "pz": [1.25],
    }
    ds = bt.from_pydict(cols)
    stepwise = ds.with_columns(**bt.se3_transform(i, ("px", "py", "pz"), prefix="m"))
    stepwise = stepwise.select(**bt.se3_transform(o, ("mx", "my", "mz")))
    composed = ds.with_columns(**bt.se3_compose(o, i, prefix="c"))
    composed = composed.select(
        **bt.se3_transform(("ctx", "cty", "ctz", "cqx", "cqy", "cqz", "cqw"), ("px", "py", "pz"))
    )
    a, b = stepwise.to_pydict(), composed.to_pydict()
    for axis in "xyz":
        assert a[axis][0] == pytest.approx(b[axis][0], abs=1e-12)


def test_se3_inverse_undoes_the_pose_it_inverts():
    pose = (5.0, -2.0, 1.0, *_axis_angle(0.0, 1.0, 0.0, 1.1))
    names = ("tx", "ty", "tz", "qx", "qy", "qz", "qw")
    cols = {n: [v] for n, v in zip(names, pose, strict=True)}
    cols |= {"px": [3.0], "py": [4.0], "pz": [-5.0]}
    ds = bt.from_pydict(cols)
    moved = ds.with_columns(**bt.se3_transform(names, ("px", "py", "pz"), prefix="w"))
    inv = moved.with_columns(**bt.se3_inverse(names, prefix="i"))
    back = inv.select(
        **bt.se3_transform(("itx", "ity", "itz", "iqx", "iqy", "iqz", "iqw"), ("wx", "wy", "wz"))
    ).to_pydict()
    for axis, want in zip("xyz", (3.0, 4.0, -5.0), strict=True):
        assert back[axis][0] == pytest.approx(want, abs=1e-12)


# --- Nulls, emptiness and the degenerate rotation ------------------------------------


def test_a_null_argument_nulls_the_result():
    ds = bt.from_pydict(
        {"qx": [0.0, None], "qy": [0.0, 0.0], "qz": [0.0, 0.0], "qw": [1.0, 1.0],
         "px": [1.0, 1.0], "py": [0.0, 0.0], "pz": [0.0, 0.0]}
    )  # fmt: skip
    got = ds.select(v=bt.quat_rotate_x(*_Q, *_P)).to_pydict()["v"]
    assert got == [1.0, None]


def test_a_zero_quaternion_nulls_rather_than_failing_the_query():
    """One unrecoverable pose in a log must not end a scan over the other billion rows."""
    ds = bt.from_pydict(
        {"qx": [0.0, 0.0], "qy": [0.0, 0.0], "qz": [0.0, 0.0], "qw": [1.0, 0.0],
         "px": [2.0, 2.0], "py": [0.0, 0.0], "pz": [0.0, 0.0]}
    )  # fmt: skip
    got = ds.select(v=bt.quat_rotate_x(*_Q, *_P)).to_pydict()["v"]
    assert got == [2.0, None]
    # And `quat_norm` is how the user finds that row.
    bad = ds.filter(bt.quat_norm(*_Q) == 0.0).select(n=bt.quat_norm(*_Q)).to_pydict()
    assert bad["n"] == [0.0]


def test_an_empty_input_yields_an_empty_column():
    ds = bt.from_pydict({n: [] for n in _Q + _P})
    assert ds.select(v=bt.quat_rotate_x(*_Q, *_P)).to_pydict()["v"] == []


def test_a_float32_point_cloud_transforms_without_an_explicit_cast():
    """A lidar sweep is `float32` on disk; the family must take it as it comes."""
    import pyarrow as pa

    schema = pa.schema([(n, pa.float32()) for n in _T + _Q + _P])
    values = {
        "tx": [1.0], "ty": [2.0], "tz": [3.0],
        "qx": [0.0], "qy": [0.0], "qz": [0.0], "qw": [1.0],
        "px": [10.0], "py": [20.0], "pz": [30.0],
    }  # fmt: skip
    ds = bt.from_pydict(values, schema=schema)
    got = ds.select(v=bt.se3_transform_x(*_T, *_Q, *_P)).to_pydict()["v"]
    assert got[0] == pytest.approx(11.0)


def test_streaming_agrees_with_collect():
    """The same transform over `iter_batches`, which takes a different execution path."""
    rows = 5000
    cols = {
        "tx": [1.0] * rows, "ty": [0.0] * rows, "tz": [0.0] * rows,
        "qx": [0.0] * rows, "qy": [0.0] * rows, "qz": [0.0] * rows, "qw": [1.0] * rows,
        "px": [float(i) for i in range(rows)],
        "py": [0.0] * rows, "pz": [0.0] * rows,
    }  # fmt: skip
    ds = bt.from_pydict(cols).select(v=bt.se3_transform_x(*_T, *_Q, *_P))
    collected = ds.to_pydict()["v"]
    streamed = [v for batch in ds.iter_batches() for v in batch.column("v").to_pylist()]
    assert collected == streamed
    assert streamed[:3] == [1.0, 2.0, 3.0]
