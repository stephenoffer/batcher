//! Cheap static predicates over `Expr` trees, consulted *before* execution.
//!
//! These never touch data — they inspect plan structure to make scheduling
//! decisions. Kept out of `lib.rs` so the wire-contract enum file stays within its
//! size budget.

use crate::Expr;

impl Expr {
    /// True if this expression tree contains a library-backed media decode
    /// (`.image`/`.audio`/`.video` — JPEG/PNG, audio, or video decode).
    ///
    /// A *scheduling* signal, not a correctness one: media decode does heavy,
    /// embarrassingly-parallel per-row work *inside* a single morsel (see
    /// `eval::media`), so a plan carrying it can saturate every core even when its
    /// input is a single morsel — the case `bc-interp`'s morsel-count-based pool
    /// sizing would otherwise throttle to one thread. Exhaustive by construction: a
    /// new `Expr` variant is a compile error here until it is classified, so the
    /// signal can never silently miss a future decode kernel.
    pub fn contains_media_decode(&self) -> bool {
        match self {
            Expr::Image { .. } | Expr::Audio { .. } | Expr::Video { .. } => true,

            // Leaves — no sub-expression that could carry a decode.
            Expr::Col { .. } | Expr::Lit { .. } => false,

            // Single-child wrappers.
            Expr::Not { input }
            | Expr::Cast { input, .. }
            | Expr::IsNull { input }
            | Expr::IsNotNull { input }
            | Expr::IsNan { input }
            | Expr::IsInf { input }
            | Expr::Str { input, .. }
            | Expr::Date { input, .. }
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
            | Expr::WindowBuckets { input, .. } => input.contains_media_decode(),

            // Two-child nodes.
            Expr::Binary { left, right, .. }
            | Expr::NullIf { left, right }
            | Expr::Math2 { left, right, .. }
            | Expr::ListSet { left, right, .. }
            | Expr::ListZip { left, right, .. }
            | Expr::ListBinary { left, right, .. } => {
                left.contains_media_decode() || right.contains_media_decode()
            }
            Expr::ListTransform { input, func } => {
                input.contains_media_decode() || func.contains_media_decode()
            }
            Expr::ListFilter { input, pred } => {
                input.contains_media_decode() || pred.contains_media_decode()
            }
            Expr::Sequence { start, stop, step } => {
                start.contains_media_decode()
                    || stop.contains_media_decode()
                    || step.contains_media_decode()
            }

            // Variadic nodes.
            Expr::Coalesce { inputs }
            | Expr::Hash { inputs, .. }
            | Expr::Greatest { inputs }
            | Expr::Least { inputs } => inputs.iter().any(Expr::contains_media_decode),
            Expr::Array { elements } => elements.iter().any(Expr::contains_media_decode),
            Expr::MakeStruct { fields } => fields.iter().any(|f| f.value.contains_media_decode()),
            Expr::Case {
                branches,
                otherwise,
            } => {
                otherwise.contains_media_decode()
                    || branches
                        .iter()
                        .any(|b| b.when.contains_media_decode() || b.then.contains_media_decode())
            }
        }
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
