//! A process-wide memo for compiled expressions.
//!
//! Cranelift compilation is the JIT's whole fixed cost, and [`compile_expr`] is a *pure
//! function* of `(expr, the types of the columns it references, the SIMD override)` — the
//! sample batch is consulted only for column types, never for values. So the artifact is
//! reusable across every batch, every operator instance, and every `execute_plan` call that
//! shares that triple.
//!
//! Without this memo the engine re-compiled every filter/project expression from scratch on
//! *each* `execute_plan` — a measured **16.6 ms of fixed overhead on a 64-row query** with one
//! filter and two projections, which is pure loss for small queries and catastrophic for the
//! per-batch streaming path and the per-operator UDF path (both of which call `execute_plan` in
//! a loop). The compile is Tier-1's admission price; it must be paid once, not once per call.
//!
//! Keys are the *full* structural rendering of the expression plus the referenced column
//! types — compared for equality, never merely hashed — so a hash collision cannot hand back
//! code compiled for a different expression. The interpreter remains the oracle: a miss, an
//! unsupported expression, or an evicted entry all fall back exactly as before.

use std::collections::HashMap;
use std::sync::{Arc, OnceLock, RwLock};

use arrow::array::RecordBatch;

use crate::CompiledExpr;

/// Cap on retained compiled expressions. Each entry owns a `JITModule` (a page or two of
/// executable memory), so the map is bounded rather than growing with the number of distinct
/// query shapes a long-lived driver sees. Well above the working set of any real query — TPC-H's
/// 22 queries together compile far fewer — so steady-state hit rate is ~100%.
const MAX_ENTRIES: usize = 1024;

/// `None` records a *known-unsupported* expression, so the analysis is not repeated either.
type Entry = Option<Arc<CompiledExpr>>;

fn cache() -> &'static RwLock<HashMap<String, Entry>> {
    static CACHE: OnceLock<RwLock<HashMap<String, Entry>>> = OnceLock::new();
    CACHE.get_or_init(|| RwLock::new(HashMap::new()))
}

/// The identity of a compilation: the expression, the types it will be compiled against, and
/// the SIMD policy. Rendered in full and compared by equality — this is a memo key, not a
/// fingerprint, so it must not admit collisions.
///
/// The whole schema (not just the referenced columns) types the key: determining which columns
/// an expression touches means running `analyze`, and the schema is fixed for the life of an
/// operator, so over-keying costs at most a few redundant entries and never a wrong answer.
fn key(expr: &bc_expr::Expr, batch: &RecordBatch, over: bc_arrow::SimdOverride) -> String {
    use std::fmt::Write as _;
    let schema = batch.schema();
    let mut k = format!("{expr:?}\u{1}{over:?}\u{1}");
    for f in schema.fields() {
        // `name:type` per field — `DataType`'s Display is exact (width, unit, nullability of
        // children), so two schemas that differ in any way a compile depends on differ here.
        let _ = write!(k, "{}:{}\u{2}", f.name(), f.data_type());
    }
    k
}

/// [`compile_expr`](crate::compile_expr), memoized — returns the shared artifact, compiling only
/// on the first sight of an `(expr, schema, simd)` triple.
///
/// `None` means "the interpreter handles this one" (outside the supported subset), exactly as an
/// `Err` from `compile_expr` does; the negative result is remembered too, so an unsupported
/// expression costs one analysis rather than one per call.
pub fn compile_expr_cached(
    expr: &bc_expr::Expr,
    batch: &RecordBatch,
    over: bc_arrow::SimdOverride,
) -> Entry {
    let k = key(expr, batch, over);
    if let Ok(map) = cache().read() {
        if let Some(hit) = map.get(&k) {
            return hit.clone();
        }
    }
    // Compile outside the lock: Cranelift takes milliseconds, and holding the write lock across
    // it would serialize every other operator's first-sight compile behind this one. Two threads
    // racing on the same key both compile; that is wasted work but never wrong, since the
    // artifacts are equivalent.
    let compiled = crate::compile_expr_with(expr, batch, over)
        .ok()
        .map(Arc::new);
    if let Ok(mut map) = cache().write() {
        // Bounded: drop the memo wholesale rather than evicting one entry. Compiles are rare
        // (once per operator per shape) and refilling is cheap relative to tracking LRU order.
        // In-flight users hold their own `Arc`, so clearing never frees code still being run.
        if map.len() >= MAX_ENTRIES {
            map.clear();
        }
        // Hand back whatever is *in* the map, so a lost race still converges every caller onto a
        // single artifact rather than leaving each with its own copy.
        return map.entry(k).or_insert(compiled).clone();
    }
    compiled
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::{Float64Array, Int64Array, RecordBatch};
    use arrow::datatypes::{DataType, Field, Schema};

    use super::compile_expr_cached;

    fn batch_f64() -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Float64, false)]));
        RecordBatch::try_new(
            schema,
            vec![Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0]))],
        )
        .unwrap()
    }

    fn batch_i64() -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, false)]));
        RecordBatch::try_new(schema, vec![Arc::new(Int64Array::from(vec![1, 2, 3]))]).unwrap()
    }

    /// `a + 1` — inside the JIT's supported subset for both f64 and i64.
    fn add_one() -> bc_expr::Expr {
        bc_expr::Expr::Binary {
            op: bc_expr::BinaryOp::Add,
            left: Box::new(bc_expr::Expr::Col { name: "a".into() }),
            right: Box::new(bc_expr::Expr::Lit {
                value: bc_expr::Literal::Int(1),
            }),
        }
    }

    #[test]
    fn second_compile_returns_the_same_artifact() {
        let b = batch_f64();
        let e = add_one();
        let first = compile_expr_cached(&e, &b, bc_arrow::SimdOverride::default())
            .expect("supported expr should compile");
        let second = compile_expr_cached(&e, &b, bc_arrow::SimdOverride::default())
            .expect("supported expr should compile");
        // A hit hands back the *same* allocation, not a fresh compile.
        assert!(
            Arc::ptr_eq(&first, &second),
            "cache must reuse the artifact"
        );
    }

    /// The schema is part of the key: the same expression over a different column type must not
    /// hand back code compiled for the other type.
    #[test]
    fn schema_is_part_of_the_key() {
        let e = add_one();
        let over = bc_arrow::SimdOverride::default();
        let f = compile_expr_cached(&e, &batch_f64(), over).expect("f64 compiles");
        let i = compile_expr_cached(&e, &batch_i64(), over).expect("i64 compiles");
        assert!(
            !Arc::ptr_eq(&f, &i),
            "different column types must not share a compiled artifact"
        );
        // And each still evaluates correctly for its own type.
        assert_eq!(f.eval(&batch_f64()).unwrap().len(), 3);
        assert_eq!(i.eval(&batch_i64()).unwrap().len(), 3);
    }

    /// A cached artifact must agree with the interpreter oracle — the whole point of Tier-1.
    #[test]
    fn cached_artifact_matches_the_interpreter() {
        use arrow::array::Array;
        let b = batch_f64();
        let e = add_one();
        let over = bc_arrow::SimdOverride::default();
        for _ in 0..3 {
            let jit = compile_expr_cached(&e, &b, over).expect("compiles");
            let got = jit.eval(&b).unwrap();
            let want = e.eval(&b).unwrap();
            assert_eq!(got.len(), want.len());
            let got = got.as_any().downcast_ref::<Float64Array>().unwrap();
            let want = want.as_any().downcast_ref::<Float64Array>().unwrap();
            for i in 0..got.len() {
                assert_eq!(got.value(i), want.value(i), "JIT must match the oracle");
            }
        }
    }
}
