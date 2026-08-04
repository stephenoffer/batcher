//! Cheap static analyses over `Expr` trees, consulted *before* execution.
//!
//! None of these touch a value. They read the shape of an expression, and at most the
//! schema it will run against, to answer questions the evaluator needs settled up front:
//! how to schedule it ([`Expr::contains_media_decode`]), what it costs relative to its
//! siblings ([`Expr::eval_cost`]), which columns it reads
//! ([`Expr::collect_columns`]), and whether skipping a row could hide an error it would
//! otherwise raise ([`Expr::is_infallible_predicate`]). The last two exist for
//! `super::select`, which short-circuits a conjunctive filter and needs both answers to
//! do it safely.
//!
//! [`Expr::for_each_child`] is the one place the tree's shape is written down, and it is
//! exhaustive so a new variant cannot be added without declaring its children. Kept out
//! of `lib.rs` so the wire-contract enum file stays within its size budget.

use arrow::datatypes::{DataType, Schema};

use crate::{BinaryOp, Expr, StrFunc};

/// True when `expr` is a bare column reference whose values are already UTF-8, so
/// reading it as a string involves no per-row conversion that could reject a value.
///
/// A dictionary of UTF-8 qualifies: decoding it is a gather over values that are
/// already valid. `Binary` deliberately does not — that is the case this exists to
/// exclude. An unknown column is not UTF-8 as far as this is concerned; the caller
/// falls back and the ordinary path reports the missing column.
fn is_utf8_column(expr: &Expr, schema: &Schema) -> bool {
    let Expr::Col { name } = expr else {
        return false;
    };
    let Ok(index) = schema.index_of(name) else {
        return false;
    };
    let mut dtype = schema.field(index).data_type();
    if let DataType::Dictionary(_, value) = dtype {
        dtype = value.as_ref();
    }
    matches!(dtype, DataType::Utf8 | DataType::LargeUtf8)
}

impl Expr {
    /// Flatten a top-level `AND` chain into its conjuncts, left to right.
    ///
    /// `a AND b AND c` parses as `And(And(a, b), c)`, so a filter predicate's
    /// conjuncts are only visible after this flattening. A non-`AND` expression is
    /// its own single conjunct, which is why the result is never empty. Only
    /// *top-level* `AND`s are split: an `And` nested under a `Not` or inside an
    /// `Or` branch stays whole, because neither is a conjunction of the predicate.
    pub fn and_conjuncts(&self) -> Vec<&Expr> {
        let mut out = Vec::new();
        self.push_conjuncts(&mut out);
        out
    }

    fn push_conjuncts<'a>(&'a self, out: &mut Vec<&'a Expr>) {
        match self {
            Expr::Binary {
                op: BinaryOp::And,
                left,
                right,
            } => {
                left.push_conjuncts(out);
                right.push_conjuncts(out);
            }
            other => out.push(other),
        }
    }

    /// True when evaluating this expression can only fail for reasons visible in the
    /// *schema*, never for reasons visible in a particular *row*.
    ///
    /// This is the licence to skip rows. [`Expr::short_circuit_filter_mask`] evaluates
    /// the later conjuncts of an `AND` chain only over the rows the earlier ones kept,
    /// which is observationally identical to evaluating everything over every row
    /// *only if* no skipped row could have raised. A schema-driven failure — a missing
    /// column, a type mismatch, an `Expr` that names an argument it does not carry —
    /// is safe, because it fires on whatever rows *do* survive; the caller's fallback
    /// then re-raises it verbatim. A value-driven failure is not: checked arithmetic
    /// overflows on one row and not its neighbour, `Div` divides by zero on one row,
    /// a non-`try` cast rejects one string, and a UTF-8 cast rejects one byte
    /// sequence. Skipping the row that would have raised is the one way this
    /// optimization could change an outcome, so those are all excluded.
    ///
    /// `schema` is needed because the answer is not always a property of the
    /// expression alone: a string predicate over a `Utf8` column cannot fail on a
    /// value, and the identical expression over a `Binary` column can, because
    /// evaluation casts it and the cast rejects invalid UTF-8 one row at a time.
    ///
    /// Exhaustive by construction, with no wildcard arm: a new `Expr` variant is a
    /// compile error here until it is classified. That is deliberate and load-bearing
    /// — the failure mode of a variant silently defaulting to "infallible" is a query
    /// that stops raising an error it should raise, which no test would notice.
    pub fn is_infallible_predicate(&self, schema: &Schema) -> bool {
        match self {
            // Leaves. A missing column raises, but that is schema-driven.
            Expr::Col { .. } | Expr::Lit { .. } => true,

            // Boolean shaping and null tests — total functions over any input type.
            Expr::Not { input }
            | Expr::IsNull { input }
            | Expr::IsNotNull { input }
            | Expr::IsNan { input }
            | Expr::IsInf { input } => input.is_infallible_predicate(schema),

            // Comparisons and boolean combinators only. Arithmetic can overflow,
            // `Div`/`Mod` can divide by zero, `Concat`/bitwise/`AddMonths` are not
            // predicates. Listing the safe ops positively means a new operator is
            // fallible until someone argues otherwise.
            Expr::Binary { op, left, right } => {
                matches!(
                    op,
                    BinaryOp::Eq
                        | BinaryOp::Ne
                        | BinaryOp::Lt
                        | BinaryOp::Le
                        | BinaryOp::Gt
                        | BinaryOp::Ge
                        | BinaryOp::And
                        | BinaryOp::Or
                ) && left.is_infallible_predicate(schema)
                    && right.is_infallible_predicate(schema)
            }

            // Set membership against literals: a comparison per distinct value.
            Expr::InList { input, .. } => input.is_infallible_predicate(schema),

            // A `TRY_CAST` yields null where a value will not convert, which is
            // precisely the absence of a value-driven failure. A strict `CAST` raises
            // on that row, so it stays out.
            Expr::Cast {
                input,
                try_cast: true,
                ..
            } => input.is_infallible_predicate(schema),

            // The string *predicates* — the ones that answer a question rather than
            // build a string. They are the conjuncts worth ordering: a `LIKE` costs
            // orders of magnitude more than the integer comparison beside it, so
            // running it over the survivors instead of the batch is the whole point.
            //
            // Two things make them safe, and both are load-bearing. They return a
            // boolean, so they never reach `try_map_str`, whose "result exceeds the
            // maximum string length" is a genuine per-row failure of every
            // string-*producing* function (`Repeat`, `Lpad`, `Overlay`, …). And the
            // input must be a column that is already UTF-8: evaluation casts a
            // `Binary` input to `Utf8`, and *that* rejects an invalid byte sequence
            // one row at a time.
            Expr::Str { func, input, .. } => {
                matches!(
                    func,
                    StrFunc::Contains
                        | StrFunc::StartsWith
                        | StrFunc::EndsWith
                        | StrFunc::Like
                        | StrFunc::Ilike
                        | StrFunc::RegexpMatches
                ) && is_utf8_column(input, schema)
            }

            // Everything else is fallible, or not worth arguing about: media decode,
            // string builders, strict casts, `Strptime`, list/map/struct access,
            // `Sequence`, arithmetic wrappers. A predicate containing any of them
            // evaluates on the whole batch, exactly as it did before short-circuiting
            // existed.
            Expr::Cast { .. }
            | Expr::Case { .. }
            | Expr::Date { .. }
            // A geo function raises on a *caller* error (a negative radius, an
            // unsupported EPSG code) rather than on a row's value, so it would qualify
            // as schema-driven — but it is also the most expensive thing an expression
            // can do per row, and short-circuiting exists to avoid evaluating a
            // predicate on rows a cheaper conjunct already rejected. Classifying it
            // fallible keeps it on the far side of that reordering.
            | Expr::Geo { .. }
            | Expr::Image { .. }
            | Expr::Audio { .. }
            | Expr::Video { .. }
            | Expr::Coalesce { .. }
            | Expr::Array { .. }
            | Expr::Hash { .. }
            | Expr::Sequence { .. }
            | Expr::ListSet { .. }
            | Expr::ListZip { .. }
            | Expr::ListTransform { .. }
            | Expr::ListFilter { .. }
            | Expr::MakeStruct { .. }
            // `MakeTemporal` validates ranges per row and answers null on an impossible
            // date, so it never raises — but it is grouped here rather than with the
            // infallible ops because a predicate built on a constructed date is not a
            // shape short-circuiting was measured on.
            | Expr::MakeTemporal { .. }
            | Expr::Math { .. }
            | Expr::Math2 { .. }
            | Expr::List { .. }
            | Expr::NullIf { .. }
            | Expr::Greatest { .. }
            | Expr::Least { .. }
            | Expr::ListGet { .. }
            | Expr::ListSimhash { .. }
            | Expr::StructField { .. }
            | Expr::ListContains { .. }
            | Expr::ListPosition { .. }
            | Expr::Map { .. }
            | Expr::ListSlice { .. }
            | Expr::DateTrunc { .. }
            | Expr::Strftime { .. }
            | Expr::ConvertTimezone { .. }
            | Expr::Strptime { .. }
            | Expr::DateOffset { .. }
            | Expr::ListJoin { .. }
            | Expr::WindowStart { .. }
            | Expr::WindowBuckets { .. }
            | Expr::ListBinary { .. } => false,
        }
    }

    /// A rough static cost of evaluating this expression once per row, in arbitrary
    /// units, used to order the conjuncts of an `AND` chain cheapest-first.
    ///
    /// Cheapest-first is the standard opening order, and the same one DuckDB's
    /// `ExpressionHeuristics` computes: with no selectivity estimate to hand, running
    /// the cheap test first is what makes the expensive one run over fewer rows.
    ///
    /// Unlike [`Expr::is_infallible_predicate`] this deliberately ends in a wildcard.
    /// A mis-costed expression picks a worse *order*, which is a performance choice;
    /// it can never change a result, so the exhaustiveness that guards the safety
    /// classifier would only cost churn here.
    pub fn eval_cost(&self) -> u32 {
        match self {
            Expr::Lit { .. } => 0,
            // A column is an `Arc` clone, unless it is dictionary-encoded and the
            // leaf has to decode it.
            Expr::Col { .. } => 1,
            Expr::Not { input }
            | Expr::IsNull { input }
            | Expr::IsNotNull { input }
            | Expr::IsNan { input }
            | Expr::IsInf { input } => 1 + input.eval_cost(),
            Expr::Binary { op, left, right } => {
                let base: u32 = match op {
                    BinaryOp::And | BinaryOp::Or => 1,
                    BinaryOp::Eq | BinaryOp::Ne => 2,
                    BinaryOp::Lt | BinaryOp::Le | BinaryOp::Gt | BinaryOp::Ge => 2,
                    // Division and modulo carry a zero check per row on top of an
                    // instruction that is itself an order of magnitude slower.
                    BinaryOp::Div | BinaryOp::Mod | BinaryOp::FloorDiv => 12,
                    BinaryOp::Concat => 30,
                    _ => 4,
                };
                base.saturating_add(left.eval_cost())
                    .saturating_add(right.eval_cost())
            }
            // One comparison per distinct value, so the set size is the cost — but
            // the dictionary-native path pays it per *dictionary entry*, not per row,
            // which is why this is capped rather than linear.
            Expr::InList { input, set } => {
                let width = (set.len() as u32).min(32);
                (4 + width).saturating_add(input.eval_cost())
            }
            Expr::Cast { input, .. } => 8u32.saturating_add(input.eval_cost()),
            // A regex or `LIKE` walks an automaton per row; a media decode is orders
            // of magnitude past that. Both are "run me last" and the exact number
            // only has to preserve that ordering.
            Expr::Str { input, .. } => 40u32.saturating_add(input.eval_cost()),
            Expr::Image { .. } | Expr::Audio { .. } | Expr::Video { .. } => 100_000,
            _ => 50,
        }
    }

    /// Collect the names of every column this expression reads, appending to `out`.
    ///
    /// Used to build the narrow sub-batch a short-circuited conjunct is evaluated
    /// over: gathering the two columns a predicate names is the point, and gathering
    /// the other forty a wide table carries is what would sink it. Names may repeat;
    /// the caller deduplicates.
    pub fn collect_columns<'a>(&'a self, out: &mut Vec<&'a str>) {
        if let Expr::Col { name } = self {
            out.push(name.as_str());
        }
        self.for_each_child(&mut |child| child.collect_columns(out));
    }
}

impl Expr {
    /// Apply `visit` to each *direct* sub-expression of this node, in evaluation order.
    ///
    /// The one place the shape of the `Expr` tree is written down, so every analysis
    /// in this module walks children identically instead of restating a forty-arm
    /// match apiece. Exhaustive by construction, with no wildcard arm: a new `Expr`
    /// variant is a compile error here until its children are declared. That is
    /// load-bearing — a variant whose children went unvisited would make every
    /// analysis below quietly blind to whatever it wraps, and none of them would
    /// fail, they would just answer "no".
    pub fn for_each_child<'a>(&'a self, visit: &mut impl FnMut(&'a Expr)) {
        match self {
            // Leaves.
            Expr::Col { .. } | Expr::Lit { .. } => {}

            // Single-child wrappers.
            Expr::Not { input }
            | Expr::Cast { input, .. }
            | Expr::IsNull { input }
            | Expr::IsNotNull { input }
            | Expr::IsNan { input }
            | Expr::IsInf { input }
            | Expr::Str { input, .. }
            | Expr::Date { input, .. }
            | Expr::Image { input, .. }
            | Expr::Audio { input, .. }
            | Expr::Video { input, .. }
            | Expr::InList { input, .. }
            | Expr::Math { input, .. }
            | Expr::List { input, .. }
            | Expr::ListGet { input, .. }
            | Expr::ListSimhash { input, .. }
            | Expr::StructField { input, .. }
            | Expr::ListContains { input, .. }
            | Expr::ListPosition { input, .. }
            | Expr::Map { input, .. }
            | Expr::ListSlice { input, .. }
            | Expr::DateTrunc { input, .. }
            | Expr::Strftime { input, .. }
            | Expr::ConvertTimezone { input, .. }
            | Expr::Strptime { input, .. }
            | Expr::DateOffset { input, .. }
            | Expr::ListJoin { input, .. }
            | Expr::WindowStart { input, .. }
            | Expr::WindowBuckets { input, .. } => visit(input),

            // Two-child nodes.
            Expr::Binary { left, right, .. }
            | Expr::NullIf { left, right }
            | Expr::Math2 { left, right, .. }
            | Expr::ListSet { left, right, .. }
            | Expr::ListZip { left, right, .. }
            | Expr::ListBinary { left, right, .. } => {
                visit(left);
                visit(right);
            }
            Expr::ListTransform { input, func } => {
                visit(input);
                visit(func);
            }
            Expr::ListFilter { input, pred } => {
                visit(input);
                visit(pred);
            }
            Expr::Sequence { start, stop, step } => {
                visit(start);
                visit(stop);
                visit(step);
            }

            // Variadic nodes.
            Expr::Coalesce { inputs }
            | Expr::Hash { inputs, .. }
            | Expr::Greatest { inputs }
            | Expr::Least { inputs } => inputs.iter().for_each(visit),
            Expr::Array { elements } => elements.iter().for_each(visit),
            Expr::Geo { args, .. } => args.iter().for_each(visit),
            Expr::MakeTemporal { args, .. } => args.iter().for_each(visit),
            Expr::MakeStruct { fields } => fields.iter().for_each(|f| visit(&f.value)),
            Expr::Case {
                branches,
                otherwise,
            } => {
                for b in branches {
                    visit(&b.when);
                    visit(&b.then);
                }
                visit(otherwise);
            }
        }
    }

    /// True if this expression tree contains a library-backed media decode
    /// (`.image`/`.audio`/`.video` — JPEG/PNG, audio, or video decode).
    ///
    /// A *scheduling* signal, not a correctness one: media decode does heavy,
    /// embarrassingly-parallel per-row work *inside* a single morsel (see
    /// `eval::media`), so a plan carrying it can saturate every core even when its
    /// input is a single morsel — the case `bc-interp`'s morsel-count-based pool
    /// sizing would otherwise throttle to one thread. It cannot silently miss a
    /// future decode kernel: [`Expr::for_each_child`] is exhaustive, so a new variant
    /// does not compile until its children are declared.
    pub fn contains_media_decode(&self) -> bool {
        if matches!(
            self,
            Expr::Image { .. } | Expr::Audio { .. } | Expr::Video { .. }
        ) {
            return true;
        }
        let mut found = false;
        self.for_each_child(&mut |child| found |= child.contains_media_decode());
        found
    }
}

#[cfg(test)]
mod tests {
    use crate::{Expr, ImageFunc, Literal};

    fn col(name: &str) -> Box<Expr> {
        Box::new(Expr::Col { name: name.into() })
    }

    #[test]
    fn detects_bare_and_nested_image_decode() {
        let bare = Expr::Image {
            func: ImageFunc::ToTensor,
            input: col("bytes"),
            width: Some(224),
            height: Some(224),
            mean: None,
            std: None,
            channels_first: false,
            x: None,
            y: None,
            format: None,
            fill: None,
        };
        assert!(bare.contains_media_decode());

        // Wrapped a few levels deep (cast over a coalesce over the decode).
        let nested = Expr::Cast {
            input: Box::new(Expr::Coalesce {
                inputs: vec![Expr::Col { name: "x".into() }, bare],
            }),
            dtype: "int64".into(),
            try_cast: false,
        };
        assert!(nested.contains_media_decode());
    }

    #[test]
    fn plain_expr_has_no_media_decode() {
        let e = Expr::Binary {
            op: crate::BinaryOp::Add,
            left: col("a"),
            right: Box::new(Expr::Lit {
                value: Literal::Int(1),
            }),
        };
        assert!(!e.contains_media_decode());
    }
}
