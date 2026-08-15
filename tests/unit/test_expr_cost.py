"""Unit tests for the expression cost model and its JIT-subset mirror.

Two contracts are pinned here:

* `jit_compilable` must agree with `crates/bc-codegen/src/analyze.rs` on which
  expressions the Cranelift tier accepts, and must **never** claim an expression is
  compilable when `analyze` rejects it on structure alone (a false positive under-prices
  it; a false negative only over-prices it).
* `expr_cost_factor` must be exactly 1.0 for the archetypal `col OP literal` predicate,
  because every calibrated `CostCoefficients` value was fitted against that baseline.
"""

from __future__ import annotations

import datetime as dt

import pytest

import batcher as bt
from batcher.kyber.expr_cost import (
    JIT_SPEEDUP,
    expr_cost,
    expr_cost_factor,
    jit_compilable,
    raw_expr_cost,
)
from batcher.plan.expr_ir import Binary, Case, Cast, Col, InList, Lit

# --- jit_compilable: the supported subset -----------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "expr",
    [
        bt.col("x") > 5,
        bt.col("x") + 1,
        (bt.col("x") > 5) & (bt.col("y") < 2.0),
        (bt.col("x") > 5) | (bt.col("y") < 2.0),
        ~(bt.col("x") > 5),
        bt.col("x"),  # a bare numeric column is a compilable column clone
        bt.col("x") / 3,  # constant divisor cannot trap
        bt.col("x") / 2.5,  # float division is IEEE, never traps
        Cast(bt.col("x"), "float64", False),
        Cast(bt.col("x"), "double", False),  # dtype alias
        # `analyze.rs` lowers sqrt directly and the transcendentals to a libm call.
        bt.col("x").sqrt(),
        bt.col("x").ln(),
        bt.col("x").abs(),
        # `Expr.__truediv__` casts to float64, and IEEE division never traps.
        bt.col("x") / bt.col("y"),
        bt.col("x") / 0,
        # Temporal comparison against the same temporal type — the ubiquitous TPC-H
        # date filter, which `analyze` compiles.
        bt.col("d") < dt.date(1998, 1, 1),
        bt.col("t") >= dt.datetime(2020, 1, 1),
    ],
)
def test_jit_compilable_accepts_supported_subset(expr):
    assert jit_compilable(expr)


@pytest.mark.unit
@pytest.mark.parametrize(
    "expr",
    [
        bt.col("s") == "abc",  # string literal
        bt.col("b") == True,  # noqa: E712 - bool literal is explicitly unsupported
        bt.col("s").str.contains("a"),  # string function
        bt.col("s").str.regexp_matches("^a"),
        bt.col("x").is_null(),
        bt.col("x").is_not_null(),
        bt.col("x") % bt.col("y"),  # integer modulo by a non-constant divisor traps
        Binary("div", bt.col("x"), bt.col("y")),  # raw integer division
        bt.col("y").round(),  # rounding mode differs from the interpreter
        bt.col("y").sign(),
        bt.col("y").cbrt(),  # Rust's software cbrt differs from libm by 1 ULP
        Cast(bt.col("x"), "int64", True),  # try_cast
        Cast(bt.col("x"), "string", False),  # unsupported target dtype
        InList(bt.col("x"), (1, 2, 3)),  # hash-set probe, interpreter only
        Lit(dt.date(2020, 1, 1)),  # a bare temporal is not a storable JIT output
        # `case` whose result is a string, not a numeric
        Case([(bt.col("x") > 1, Lit("a"))], Lit("b")),
    ],
)
def test_jit_compilable_rejects_unsupported(expr):
    assert not jit_compilable(expr)


@pytest.mark.unit
def test_known_over_claims_from_missing_dtypes():
    """A bare `Col` carries no dtype here, so two cases are reported compilable that
    `analyze` would reject once it sees the batch. Both are harmless — the JIT itself
    makes the real call and falls back — and the cost error is bounded by `JIT_SPEEDUP`.
    Pinned so the limitation stays visible rather than being rediscovered as a bug.
    """
    assert jit_compilable(bt.col("d"))  # a bare Date32 column: `analyze` rejects it
    assert jit_compilable(bt.col("s") == bt.col("t"))  # two string columns compared


@pytest.mark.unit
def test_jit_compilable_numeric_case():
    expr = Case([(bt.col("x") > 1, Lit(1))], Lit(2))
    assert jit_compilable(expr)


@pytest.mark.unit
def test_int_division_by_trapping_constant_is_rejected():
    # cranelift's integer `sdiv` traps on 0 and on i64::MIN / -1; `analyze` refuses both.
    # Built from the raw IR node: `Expr.__truediv__` casts to float64 first, and IEEE
    # float division never traps (see `test_float_division_always_compiles`).
    assert not jit_compilable(Binary("div", bt.col("x"), Lit(0)))
    assert not jit_compilable(Binary("div", bt.col("x"), Lit(-1)))
    assert not jit_compilable(Binary("div", bt.col("x"), bt.col("y")))
    assert jit_compilable(Binary("div", bt.col("x"), Lit(7)))


@pytest.mark.unit
def test_float_division_always_compiles():
    # IEEE division yields inf/nan rather than trapping, so even a zero or non-constant
    # divisor compiles once either operand is a float.
    assert jit_compilable(bt.col("x") / 0)
    assert jit_compilable(bt.col("x") / bt.col("y"))
    assert jit_compilable(bt.col("x") / 2.5)


# --- expr_cost: ranking and the JIT speedup ---------------------------------------


@pytest.mark.unit
def test_compiled_expression_is_cheaper_than_its_raw_cost():
    expr = bt.col("x") > 5
    assert expr_cost(expr) == pytest.approx(raw_expr_cost(expr) / JIT_SPEEDUP)


@pytest.mark.unit
def test_interpreted_expression_pays_no_speedup():
    expr = bt.col("s").str.contains("a")
    assert expr_cost(expr) == pytest.approx(raw_expr_cost(expr))


@pytest.mark.unit
def test_cost_ranks_regex_far_above_a_comparison():
    cheap = expr_cost(bt.col("x") > 5)
    regex = expr_cost(bt.col("s").str.regexp_matches("^a.*z$"))
    assert regex > 100 * cheap


@pytest.mark.unit
def test_cost_ranks_string_ops_between_comparison_and_regex():
    cheap = expr_cost(bt.col("x") > 5)
    contains = expr_cost(bt.col("s").str.contains("a"))
    regex = expr_cost(bt.col("s").str.regexp_matches("^a"))
    assert cheap < contains < regex


@pytest.mark.unit
def test_cost_is_additive_over_the_tree():
    # A conjunction costs at least as much as either of its conjuncts.
    a, b = bt.col("x") > 5, bt.col("s").str.contains("a")
    assert expr_cost(a & b) > expr_cost(b) > expr_cost(a)


@pytest.mark.unit
def test_slotted_nodes_expose_their_children():
    # `InList` and `Aliased` use __slots__; a naive vars() walk would treat them as
    # leaves and under-price them. The child column read must be counted.
    assert raw_expr_cost(InList(bt.col("x"), (1, 2))) > raw_expr_cost(Lit(1))
    aliased = (bt.col("s").str.contains("a")).alias("hit")
    assert raw_expr_cost(aliased) == pytest.approx(raw_expr_cost(bt.col("s").str.contains("a")))


@pytest.mark.unit
def test_raw_cost_is_memoized_per_node():
    # `raw_expr_cost` depends only on structure, so it is cached on the node. `expr_cost`
    # is not, because it also depends on the JIT speedup calibration learns.
    expr = bt.col("s").str.regexp_matches("^a")
    assert raw_expr_cost(expr) == raw_expr_cost(expr)
    assert expr.__dict__["_c_rawcost"] == pytest.approx(raw_expr_cost(expr))


@pytest.mark.unit
def test_speedup_scales_interpreted_expressions_against_compiled_ones():
    # The archetypal compiled predicate is 1.0 at any speedup (its baseline moves with
    # it); a higher measured speedup makes an interpreted regex relatively pricier.
    regex = bt.col("s").str.regexp_matches("^a")
    assert expr_cost_factor(bt.col("x") > 5, 1.0) == pytest.approx(1.0)
    assert expr_cost_factor(bt.col("x") > 5, 8.0) == pytest.approx(1.0)
    assert expr_cost_factor(regex, 8.0) > expr_cost_factor(regex, 1.0)


# --- expr_cost_factor: the multiplier the cost model applies ----------------------


@pytest.mark.unit
def test_baseline_predicate_has_factor_one():
    # The whole point: a plain compiled comparison must not perturb the calibrated
    # per-row coefficients it multiplies.
    assert expr_cost_factor(bt.col("x") > 5) == pytest.approx(1.0)
    assert expr_cost_factor(bt.col("x") < 5) == pytest.approx(1.0)
    assert expr_cost_factor(Col("x") == Lit(5)) == pytest.approx(1.0)


@pytest.mark.unit
def test_factor_is_clamped_at_both_ends():
    # A bare column floors at 0.2. The ceiling sits above the priciest *measured* scalar
    # function, so real costs pass through untruncated; a full media decode is what
    # reaches it.
    from batcher.plan.expr_ir.image import ImageFunc

    assert expr_cost_factor(bt.col("x")) == pytest.approx(0.2)
    assert expr_cost_factor(bt.col("s").str.sha256()) > 200.0  # not truncated
    assert expr_cost_factor(ImageFunc("sharpness", bt.col("s"))) == pytest.approx(1000.0)


@pytest.mark.unit
def test_a_header_probe_is_not_priced_as_a_decode():
    """`.image` used to carry one flat cost for every op, so the ceiling caught them all.

    It is wrong by more than two orders of magnitude *inside* the family: `decode` and the
    `probe` ops read the container header and never touch a pixel (~15 us/row), while
    `sharpness` decodes and walks the whole luma plane (~3,300 us/row). Pricing them alike
    meant the optimizer had no reason to run the cheap one first -- see
    `test_a_cheap_media_predicate_runs_before_an_expensive_one`.
    """
    from batcher.plan.expr_ir.image import ImageFunc

    # Asserted on the *raw* cost, not the factor. `expr_cost_factor` clamps at 1000 so no
    # one expression can dominate the row-count term in join ordering, and every media op
    # is far above that ceiling -- so the factor cannot tell these apart and is not meant
    # to. The rules that order media work (`filter_split`, `cse`) read the raw cost, which
    # is where the resolution has to be.
    header = expr_cost(ImageFunc("decode", bt.col("s")))
    decode = expr_cost(ImageFunc("sharpness", bt.col("s")))
    assert header < decode / 50, "a header read must not be priced as a full decode"
    # ...and both must still outrank the priciest non-media scalar, since even reading a
    # container header costs far more per row than a regex over a short string.
    assert header > expr_cost(bt.col("s").str.regexp_matches("^a"))


@pytest.mark.unit
def test_media_costs_are_ranked_by_what_the_kernel_actually_does():
    """The orderings a guessed table gets wrong, pinned. See `weights` for the numbers.

    `to_tensor_f32` costing more than `to_tensor` is the counter-intuitive one: the float
    conversion and per-channel normalization cost more than the decode-and-resize they
    follow. `is_grayscale` is the other -- it sits beside the header-only probes but
    proving an image is grayscale means looking at its pixels.
    """
    from batcher.plan.expr_ir.audio import AudioFunc
    from batcher.plan.expr_ir.image import ImageFunc

    img = lambda fn, **kw: expr_cost(ImageFunc(fn, bt.col("s"), **kw))  # noqa: E731
    aud = lambda fn: expr_cost(AudioFunc(fn, bt.col("s")))  # noqa: E731

    # Header reads are trivial next to anything that decodes.
    assert img("has_alpha") < img("dhash") < img("brightness") < img("blur")
    # A perceptual hash downsamples hard, so it is far cheaper than a full-plane measure.
    assert img("dhash") < 0.5 * img("sharpness")
    # Normalizing to float costs more than the decode-and-resize before it.
    assert img("to_tensor_f32") > 2 * img("to_tensor")
    # Grayscale-in-fact is a pixel question, not a header one.
    assert img("is_grayscale") > 50 * img("has_alpha")
    # Audio: a scalar reduction never materializes the signal, so it is cheaper than one
    # that does; the STFT front ends are dearer again, and resampling dearest.
    assert aud("rms") < aud("to_waveform") < aud("mel_spectrogram") < aud("resample")


@pytest.mark.unit
def test_a_per_row_crop_is_priced_as_a_decode():
    """`ImageCrop` is a separate node, and that is how it escaped the family table.

    Its window is four sub-expressions rather than four scalars -- the whole point being
    that a detector's predicted box varies per row -- so it is not an `ImageFunc` and the
    per-function table above never saw it. It fell through to `_DEFAULT_COST` and was
    priced at 5.0: cheaper than a regex, for an op that decodes a JPEG, crops it and
    re-encodes the result (measured at 2,957 us/row on this file's reference frame).
    """
    from batcher.plan.expr_ir.image import ImageCrop

    crop = ImageCrop(bt.col("img"), bt.col("x"), bt.col("y"), Lit(64), Lit(64))
    assert expr_cost(crop) > expr_cost(bt.col("s").str.sha256())
    # It belongs in the decode-and-re-encode band, beside the ops it actually resembles.
    assert expr_cost(crop) > 0.5 * expr_cost(bt.col("img").image.thumbnail(64))


@pytest.mark.unit
def test_measured_function_costs_are_ranked_correctly():
    """The table is calibrated against measurement (see `weights`); these orderings are
    the ones a guessed table gets wrong, so they are pinned.
    """
    f = expr_cost_factor
    s = bt.col("s")
    # A regex is only a few times a substring search — RE2 prefilters on literals.
    assert f(s.str.contains("abc")) < f(s.str.regexp_matches("a.*b")) < 4 * f(s.str.contains("abc"))
    # ...but a digest and an edit distance dwarf a regex.
    assert f(s.str.sha256()) > 4 * f(s.str.regexp_matches("a.*b"))
    assert f(s.str.levenshtein("abc")) > 3 * f(s.str.regexp_matches("a.*b"))
    # Even `length` is expensive: decoding string offsets dominates the operation.
    assert f(s.str.len()) > 20 * f(bt.col("x") > 5)
    # Numeric math stays cheap next to any string work.
    assert f(bt.col("y").sqrt()) < f(s.str.len())


@pytest.mark.unit
def test_factor_scales_with_expense():
    assert expr_cost_factor(bt.col("s").str.regexp_matches("^a")) > 50


@pytest.mark.unit
def test_the_vector_ops_an_embedding_dedup_is_built_from_are_priced():
    """`simhash` is the blocking key a near-duplicate pass over an image corpus uses.

    It had no cost entry at all, so it defaulted to 5.0 -- four orders of magnitude under a
    measured ~45 us/row on a 384-dimension embedding. Anything that reads the raw cost
    (`filter_split`, `cse`) therefore treated building a signature over every embedding in
    a corpus as cheaper than a regex.
    """
    simhash = expr_cost(bt.col("v").list.simhash())
    assert simhash > 1000 * expr_cost(bt.col("s").str.regexp_matches("^a"))
    # Element-wise vector arithmetic is far cheaper than hashing, but still not a scalar.
    assert expr_cost(bt.col("v").list.add(bt.col("v"))) < simhash
    assert expr_cost(bt.col("v").list.add(bt.col("v"))) > expr_cost(bt.col("a") + bt.col("b"))
