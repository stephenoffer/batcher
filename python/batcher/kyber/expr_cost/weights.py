"""Per-node evaluation costs, and the traversal that reaches every sub-expression.

The numbers are the data plane's per-row work for one `Expr` node, in abstract
work-units normalized so that **one interpreted numeric comparison over one row = 1.0**.
They are grouped by family (binary operator, string function, math function, date part,
and a by-class-name table for everything else) so a new node family is priced by adding
an entry, never by editing the traversal.

Costs describe the Tier-0 Arrow-kernel path. `jit` decides whether the Cranelift tier
runs the expression instead, and `model` applies the resulting speedup.

Where the numbers come from. The families below were **measured**, not guessed: each
function was run as the sole expression of a projection over a million rows in a fresh
process, and a bare column projection was subtracted so only the function's own work is
counted. Some results are unintuitive and are the reason a hand-guessed table is not good
enough — `regexp_matches` is only ~3x `contains` (Arrow's RE2 prefilters on literals),
while `sha256` is ~7x a regex and `levenshtein` ~5x, because their per-row work is a real
O(n) or O(n^2) loop. Even `length` costs ~40x a comparison, since decoding string offsets
dominates the operation itself.

Two caveats. String costs scale with string length (measured on 16-char values), and a
function's cost varies with its arguments (an anchored regex is far cheaper than a
backtracking one). A single scalar per function cannot capture that, so these are
*rankings* good to a small factor, not predictions. The systematic component of the
residual error is absorbed by `jit_speedup`, which `kyber.calibration` fits from the tier
tag the engine reports on every operator.
"""

from __future__ import annotations

import dataclasses

from batcher.plan.expr_ir import (
    Aliased,
    Binary,
    Case,
    Col,
    Expr,
    InList,
    Lit,
    Math2Expr,
    MathExpr,
)
from batcher.plan.expr_ir.node_base import IRNode

__all__ = ["own_cost", "sub_exprs"]

_LEAF_COST = 0.2  # a column buffer read; a literal broadcasts for free
_DEFAULT_COST = 5.0  # an unrecognized node: assume moderately expensive (safe direction)

# Binary operators. Comparisons/arithmetic are single instructions; `concat`
# allocates a string per row, `add_months` walks a calendar.
BINARY_COST: dict[str, float] = {
    "eq": 1.0,
    "ne": 1.0,
    "lt": 1.0,
    "le": 1.0,
    "gt": 1.0,
    "ge": 1.0,
    "add": 1.0,
    "sub": 1.0,
    "mul": 1.0,
    "div": 3.0,  # hardware divide is multi-cycle
    "mod": 3.0,
    "floor_div": 4.0,  # a divide plus the remainder-sign floor correction
    "and": 0.5,
    "or": 0.5,
    "bit_and": 1.0,
    "bit_or": 1.0,
    "bit_xor": 1.0,
    "shift_left": 1.0,
    "shift_right": 1.0,
    "concat": 12.0,
    "add_months": 8.0,
}

# String functions. Measured (see the module docstring); the floor for *any* of them is
# the ~15 units it costs merely to decode a row's string offsets and bytes.
_STR_DEFAULT = 20.0
_STR_COST: dict[str, float] = {
    # Buffer/offset reads — dominated by string decoding, not by the operation.
    "len": 14.5,
    "bit_length": 14.5,
    "octet_length": 14.5,
    "ascii": 14.5,
    # Prefix/suffix tests short-circuit on the first bytes.
    "starts_with": 8.0,
    "ends_with": 8.0,
    # Substring search over the whole value.
    "contains": 20.0,
    "position": 20.0,
    "hash64": 12.0,
    "xxhash64": 12.0,
    # Allocating transforms: a new string per row.
    "upper": 14.5,
    "lower": 14.5,
    "initcap": 18.0,
    "trim": 12.0,
    "l_trim": 12.0,
    "r_trim": 12.0,
    "reverse": 14.5,
    "right": 12.0,
    "substr": 12.0,
    "repeat": 25.0,
    "lpad": 22.0,
    "rpad": 22.0,
    "replace": 22.0,
    "translate": 22.0,
    "overlay": 22.0,
    "split": 35.0,
    "split_part": 25.0,
    "substring_index": 25.0,
    # Glob matching walks the value; `ilike` also case-folds.
    "like": 28.0,
    "ilike": 34.0,
    # Encodings.
    "hex": 25.0,
    "unhex": 25.0,
    "base64": 25.0,
    "from_base64": 25.0,
    # Regex compiles once; the match is a per-row automaton walk. Cheaper than intuition
    # suggests because RE2 prefilters on required literals.
    "regexp_matches": 48.0,
    "regexp_count": 48.0,
    "regexp_extract": 55.0,
    "regexp_extract_all": 70.0,
    "regexp_replace": 65.0,
    "regexp_replace_all": 75.0,
    # Edit distance is a real O(len^2) inner loop — far pricier than a regex.
    "levenshtein": 230.0,
    "soundex": 40.0,
    # Cryptographic digests: a full compression function per row, the priciest string ops.
    "crc32": 60.0,
    "md5": 155.0,
    "sha1": 210.0,
    "sha256": 325.0,
    # JSON extraction parses a document per row.
    "json_extract_bool": 90.0,
    "json_extract_int": 90.0,
    "json_extract_float": 90.0,
    "json_extract_string": 100.0,
}

# Unary math. Measured: a hardware `sqrt` is barely more than an add, and even a libm
# `log` is only ~3x a comparison — nothing like the string family. The transcendentals
# that lower to a libm libcall are the priciest, and the ones the JIT cannot lower (see
# `jit`) pay the interpreter's dispatch on top.
_MATH_DEFAULT = 5.0
_MATH_COST: dict[str, float] = {
    "abs": 1.0,
    "sign": 1.0,
    "ceil": 1.2,
    "floor": 1.2,
    "trunc": 1.2,
    "round": 1.5,
    "degrees": 1.2,
    "radians": 1.2,
    "bit_count": 1.5,
    "sqrt": 1.5,
    "cbrt": 6.0,
    "exp": 4.0,
    "ln": 4.0,
    "log2": 4.0,
    "log10": 4.0,
    "sin": 6.0,
    "cos": 6.0,
    "tan": 6.0,
    "cot": 7.0,
    "asin": 6.0,
    "acos": 6.0,
    "atan": 6.0,
    "sinh": 6.0,
    "cosh": 6.0,
    "tanh": 6.0,
    "asinh": 7.0,
    "acosh": 7.0,
    "atanh": 7.0,
    "factorial": 15.0,
}

# Date-part extraction is a divmod on the epoch value, except where it consults a
# calendar table or formats a name.
_DATE_DEFAULT = 3.0
_DATE_COST: dict[str, float] = {
    "dayname": 8.0,
    "monthname": 8.0,
    "last_day": 5.0,
    "days_in_month": 5.0,
    "is_leap_year": 5.0,
}

# Flat per-node costs keyed by class name, so node families added later (list, struct,
# map, media) are priced without importing every class into this module.
_BY_CLASS_NAME: dict[str, float] = {
    "IsNull": 0.5,
    "IsNotNull": 0.5,
    "IsNan": 0.5,
    "IsInf": 0.5,
    "Not": 0.5,
    "Cast": 2.0,
    "Coalesce": 1.0,
    "NullIf": 1.5,
    "Greatest": 1.0,
    "Least": 1.0,
    "Array": 2.0,
    "MakeStruct": 2.0,
    "Sequence": 5.0,
    "StructField": 1.0,
    "ListJoin": 20.0,
    "MapFunc": 10.0,
    # Nested/list kernels walk a child buffer per row.
    "ListFunc": 15.0,
    "ListGet": 5.0,
    "ListContains": 15.0,
    "ListPosition": 15.0,
    "ListSlice": 10.0,
    "ListBinary": 20.0,
    "ListSet": 25.0,
    # These evaluate a sub-expression per *element*, not per row.
    "ListTransform": 40.0,
    "ListFilter": 40.0,
    # Temporal formatting/parsing walks a format string per row.
    "Strftime": 40.0,
    "Strptime": 45.0,
    "ConvertTimezone": 15.0,
    "DateTrunc": 5.0,
    "DateOffset": 8.0,
    "WindowStart": 5.0,
    "WindowBuckets": 5.0,
    # Media decode dwarfs every scalar op; costing it high is what makes Kyber push
    # filters below an image/audio/video expression. `ImageFunc`/`AudioFunc` are priced
    # per function below -- the spread *within* each family is two orders of magnitude,
    # which a single class-level number cannot express. `VideoFunc` keeps a flat estimate
    # because CI builds the engine without the `video` cargo feature, so its kernels
    # cannot be run and therefore cannot be measured the way everything else here was.
    # It sits at the image decode class, which is a floor rather than a measurement: a
    # sampled clip decodes several frames, so the true cost is higher, and the number has
    # to be at least this or the optimizer would schedule a video decode ahead of an image
    # one it can actually price.
    "VideoFunc": 16_000_000.0,
}

# --- Media, measured ------------------------------------------------------------------
#
# On the same normalization as the rest of the file, and by the same method: each op run
# as the sole expression of a projection, with a bare column projection subtracted. The
# reference inputs are a **512x512 JPEG** and a **3-second 16 kHz mono WAV**; media cost
# scales with resolution and duration far more strongly than a string function's does with
# string length, so these are rankings within a family, not absolute predictions.
#
# The numbers are large because the unit is small, and getting that scale right is the
# whole point rather than a presentational choice. One unit is ~0.2 ns/row, fixed by an
# entry already in this file: `regexp_matches` is 48.0 units and measures 9.5 ns/row net
# of a bare projection. A 512x512 JPEG decode really is tens of millions of times a
# vectorized numeric comparison, and a table that rounds that to a friendlier number is
# not being conservative, it is being wrong in a direction that matters -- an earlier
# revision of this block anchored the family at 500.0 and thereby priced an image header
# read (~15 us/row) *below* a regex (~10 ns/row), so `filter_split` would have run the
# header probe first and paid it on every row.
#
# `benchmarks/scenarios/media_op_cost.py` re-measures all of them, so the table has a
# reproducer rather than only a provenance claim. Expect its absolute numbers to differ:
# it synthesizes a structured, JPEG-compressible frame (what a camera produces) where
# these were taken on incompressible noise (a decoder's worst case), which moves every
# image figure by roughly a fifth. What carries over, and what the optimizer reads, is the
# ordering and the size of the gaps between the bands below. Entries inside a band -- the
# four header reads, or the three hashes -- are within measurement noise of each other and
# are not meant to be distinguishable.
#
# The family used to carry one flat 500.0 for every op, and the measurements say that was
# wrong by more than two orders of magnitude *inside* the family:
#
#     image.format / has_alpha / aspect_ratio / decode      10-17 us/row   (header only)
#     image.dhash / phash / ahash                          800-860 us/row
#     image.brightness / sharpness / entropy              3150-3330 us/row
#     image.blur / sharpen                                    ~13,600 us/row
#
# `probe.rs` reads the container header and never decodes a pixel, so `has_alpha` is ~200x
# cheaper than `sharpness` -- and `filter_split` orders conjuncts by exactly this number
# (Krishnamurthy-Boral-Zaniolo rank), so with both at 500.0 it could not tell the header
# probe from the full decode and had no reason to run the cheap one first.
#
# Two results are worth keeping because a guessed table gets them backwards:
# `to_tensor_f32` is ~3.2x `to_tensor` (the float conversion and normalization cost more
# than the decode-and-resize they follow), and `is_grayscale` is **not** a header fact
# despite living beside the ones that are -- proving an image is grayscale means looking
# at its pixels, so it costs a full decode while `has_alpha` next to it costs nothing.

_IMAGE_DEFAULT = 16_000_000.0  # an untabulated image op: assume it decodes (the safe direction)
_IMAGE_COST: dict[str, float] = {
    # Header only -- no pixel is decoded (`bc-expr::eval::media::image::probe`).
    "format": 51_000.0,
    "aspect_ratio": 62_000.0,
    "has_alpha": 78_000.0,
    "exif_orientation": 80_000.0,
    "decode": 87_000.0,
    # Perceptual hashes: decode, downsample hard, compare cells.
    "ahash": 4_000_000.0,
    "dhash": 4_100_000.0,
    "phash": 4_300_000.0,
    # Decode and resize to a tensor.
    "to_tensor": 8_500_000.0,
    "to_grayscale": 8_800_000.0,
    "center_crop": 9_900_000.0,
    "resize": 12_000_000.0,
    "thumbnail": 13_000_000.0,
    "letterbox": 14_000_000.0,
    # Decode and walk every pixel.
    "is_grayscale": 15_000_000.0,
    "brightness": 16_000_000.0,
    "colorfulness": 16_000_000.0,
    "entropy": 16_000_000.0,
    "mean_color": 17_000_000.0,
    "sharpness": 17_000_000.0,
    # Decode and re-encode. `encode` varies with the target container (JPEG ~4,150 us,
    # PNG ~6,520); the entry is the midpoint, since the format is a plan-time constant
    # this table is not indexed by.
    "encode": 27_000_000.0,
    "to_tensor_f32": 27_000_000.0,
    "convert": 32_000_000.0,
    "auto_orient": 32_000_000.0,
    "flip_vertical": 33_000_000.0,
    "adjust_brightness": 34_000_000.0,
    "rotate": 35_000_000.0,
    "posterize": 36_000_000.0,
    "solarize": 36_000_000.0,
    "adjust_saturation": 36_000_000.0,
    "flip_horizontal": 37_000_000.0,
    "pad": 37_000_000.0,
    "invert": 37_000_000.0,
    "autocontrast": 38_000_000.0,
    "equalize": 38_000_000.0,
    "adjust_contrast": 40_000_000.0,
    "adjust_hue": 40_000_000.0,
    # Convolutions over the full plane -- the priciest ops in the family by a wide margin.
    "blur": 67_000_000.0,
    "sharpen": 69_000_000.0,
}

_AUDIO_DEFAULT = 5_300_000.0  # an untabulated audio op: assume it materializes a waveform
_AUDIO_COST: dict[str, float] = {
    # Decode PCM and reduce to one scalar -- no waveform is materialized, which is the
    # ~800 us/row difference between these and `to_waveform`.
    "decode": 1_100_000.0,
    "dbfs": 1_200_000.0,
    "rms": 1_200_000.0,
    "peak_dbfs": 1_200_000.0,
    "clipping_ratio": 1_300_000.0,
    "silence_ratio": 1_300_000.0,
    "zero_crossing_rate": 1_300_000.0,
    "encode_wav": 1_800_000.0,
    "slice": 2_700_000.0,
    # One STFT, reduced to a scalar per frame.
    "spectral_rolloff": 3_600_000.0,
    "spectral_centroid": 3_700_000.0,
    "spectral_flatness": 3_700_000.0,
    "spectral_bandwidth": 3_900_000.0,
    "pad_or_trim": 4_000_000.0,
    # Decode and hand back the whole signal.
    "to_waveform": 5_300_000.0,
    "trim_silence": 5_400_000.0,
    "peak_normalize": 5_400_000.0,
    "pre_emphasis": 5_500_000.0,
    "rms_normalize": 5_600_000.0,
    # STFT and filterbank front ends.
    "spectrogram": 8_800_000.0,
    "mel_spectrogram": 14_000_000.0,
    "mfcc": 21_000_000.0,
    # Band-limited sinc resampling is the most expensive audio op measured.
    "resample": 25_000_000.0,
}


def own_cost(expr: Expr) -> float:
    """The cost of evaluating `expr`'s own node, excluding its sub-expressions.

    Args:
        expr: The scalar expression node to price.

    Returns:
        Cost in work-units, where one numeric comparison over one row is 1.0.
    """
    if isinstance(expr, (Col, Lit)):
        return _LEAF_COST
    if isinstance(expr, Binary):
        return BINARY_COST.get(expr.op, _DEFAULT_COST)
    if isinstance(expr, Aliased):
        return 0.0  # transparent in the IR
    if isinstance(expr, InList):
        # Lowered to a hash-set probe; a handful of values stays a compare chain.
        return min(6.0, 1.0 + 0.3 * len(expr.values))
    if isinstance(expr, Case):
        # Every branch condition and its result are evaluated (no short-circuit); the
        # per-branch selection itself is what is counted here.
        return 0.5 * (len(expr.branches) + 1)
    if isinstance(expr, MathExpr):
        return _MATH_COST.get(expr.fn, _MATH_DEFAULT)
    if isinstance(expr, Math2Expr):
        return 10.0
    cls = type(expr).__name__
    if cls == "StrFunc":
        return _STR_COST.get(expr.fn, _STR_DEFAULT)
    if cls == "DateFunc":
        return _DATE_COST.get(expr.fn, _DATE_DEFAULT)
    if cls == "ImageFunc":
        return _IMAGE_COST.get(expr.fn, _IMAGE_DEFAULT)
    if cls == "AudioFunc":
        return _AUDIO_COST.get(expr.fn, _AUDIO_DEFAULT)
    return _BY_CLASS_NAME.get(cls, _DEFAULT_COST)


def sub_exprs(expr: Expr) -> tuple[Expr, ...]:
    """The immediate sub-expressions of `expr`.

    `IRNode`s are dataclasses, so their sub-expressions are found by walking the
    declared fields (recursing into lists/tuples, which is how `Case.branches` and
    `MakeStruct.fields` carry their pairs). The three hand-written `Expr` classes use
    `__slots__` and are matched explicitly — a generic `vars()` walk would silently
    miss `InList.input` and report the node as a leaf.

    Args:
        expr: The scalar expression node to descend into.

    Returns:
        Its immediate sub-expressions, in declaration order.
    """
    if isinstance(expr, InList):
        return (expr.input,)
    if isinstance(expr, Aliased):
        return (expr.inner,)
    if isinstance(expr, Lit):
        return ()
    if isinstance(expr, IRNode):
        out: list[Expr] = []
        for name in _field_names(type(expr)):
            _collect_exprs(getattr(expr, name, None), out)
        return tuple(out)
    return ()


# A node class's field names, resolved once. `dataclasses.fields` rebuilds its tuple on
# every call and this runs per node of every expression the cost model prices, which is
# every expression of every plan.
_FIELD_NAMES: dict[type, tuple[str, ...]] = {}


def _field_names(cls: type) -> tuple[str, ...]:
    """`cls`'s dataclass field names, cached per class."""
    names = _FIELD_NAMES.get(cls)
    if names is None:
        names = tuple(f.name for f in dataclasses.fields(cls))
        _FIELD_NAMES[cls] = names
    return names


def _collect_exprs(value: object, out: list[Expr]) -> None:
    """Append every `Expr` reachable from `value` (a field, or a list/tuple of them)."""
    if isinstance(value, Expr):
        out.append(value)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _collect_exprs(item, out)
