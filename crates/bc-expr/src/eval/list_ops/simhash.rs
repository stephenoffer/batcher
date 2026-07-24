//! `simhash`: a random-hyperplane LSH signature of an embedding → `List<Int64>` of bits.
//!
//! `str.minhash` estimates *Jaccard* similarity over a set of shingles; it says nothing
//! about vectors. The vector-space analogue is Charikar's SimHash: draw `num_bits`
//! random hyperplanes through the origin and record which side of each the vector falls
//! on. Two vectors separated by angle θ agree on a given bit with probability
//! `1 - θ/π`, so the fraction of agreeing bits estimates the angle — and therefore the
//! cosine similarity — of the originals.
//!
//! That estimate is what makes a *blocked* similarity join possible: band the bits, hash
//! each band, and two vectors share a band only if they are close, so the O(n²) pair
//! comparison collapses to comparing candidates. Batcher then scores the survivors with
//! the **exact** `list.cosine_similarity` over the original vectors, so the LSH controls
//! recall, never precision.
//!
//! The hyperplane normals must be spherically symmetric for `1 - θ/π` to hold, so they
//! are standard Gaussians (Box–Muller over a SplitMix64 stream keyed by
//! `(seed, bit, dim)`) — not the cheaper ±1 Rademacher entries, which break the
//! guarantee. Being hash-derived rather than stored, the same hyperplanes are
//! regenerated identically on every partition and every machine, which is what lets a
//! signature computed on one node be compared with one computed on another.
//!
//! One bit per `Int64` element is deliberately fat (a 64-bit signature costs 512 bytes,
//! not 8). It buys the signature the same `List<Int64>` shape as a MinHash signature, so
//! the *same* banding code, `list.get`, and `hash_rows` serve both — and a signature is
//! a transient blocking key, not something a pipeline stores.

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, Int64Builder, ListArray, ListBuilder};
use arrow::datatypes::DataType;

use crate::ExprError;

/// The `2^-64` scale that maps a `u64` to a uniform `(0, 1]`.
const U64_SCALE: f64 = 1.0 / (u64::MAX as f64 + 1.0);

/// Guards against a signature so wide it would blow up memory (and is never useful).
const MAX_BITS: i64 = 4096;

/// SplitMix64 — the finalizer Batcher already uses for `hash_rows`, so the projection is
/// as well-distributed as the row digest and just as reproducible.
fn mix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// One standard-normal entry of the projection matrix, derived from its coordinates.
///
/// Box–Muller needs two independent uniforms; two different mixes of the same
/// coordinates supply them. `u1` is drawn in `(0, 1]` so `ln(u1)` is finite.
fn gaussian(seed: u64, bit: u64, dim: u64) -> f64 {
    let base = mix64(seed ^ mix64(bit.wrapping_mul(0x9E37_79B9).wrapping_add(dim)));
    let u1 = (mix64(base) as f64 + 1.0) * U64_SCALE;
    let u2 = mix64(base ^ 0xA5A5_A5A5_A5A5_A5A5) as f64 * U64_SCALE;
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

/// The `num_bits × dim` projection matrix, row-major. Regenerated only when `dim`
/// changes, which for a real embedding column is once.
fn projection(seed: u64, num_bits: usize, dim: usize) -> Vec<f64> {
    let mut matrix = Vec::with_capacity(num_bits * dim);
    for bit in 0..num_bits {
        for d in 0..dim {
            matrix.push(gaussian(seed, bit as u64, d as u64));
        }
    }
    matrix
}

/// Evaluate `simhash(list, num_bits, seed)` over a `List<numeric>` column.
///
/// A null list yields a null signature; an empty list has no direction and likewise
/// yields null. A null *element* is read as `0.0` — it contributes nothing to any
/// projection, which is the same thing `list.cosine_similarity` does with it.
pub(crate) fn eval_list_simhash(
    arr: &ArrayRef,
    num_bits: i64,
    seed: i64,
) -> Result<ArrayRef, ExprError> {
    if !(1..=MAX_BITS).contains(&num_bits) {
        return Err(ExprError::InvalidArgument {
            func: "simhash".into(),
            reason: format!("num_bits must be in [1, {MAX_BITS}], got {num_bits}"),
        });
    }
    let list = arr
        .as_any()
        .downcast_ref::<ListArray>()
        .ok_or_else(|| ExprError::ExpectedType {
            func: "simhash".into(),
            want: "a List argument",
            got: arr.data_type().to_string(),
        })?;

    // One numeric read path: cast the flat child once rather than per row.
    let values = arrow::compute::cast(list.values(), &DataType::Float64)?;
    let values = values
        .as_any()
        .downcast_ref::<arrow::array::Float64Array>()
        .ok_or_else(|| ExprError::ExpectedType {
            func: "simhash".into(),
            want: "a numeric list element",
            got: list.values().data_type().to_string(),
        })?;

    let bits = num_bits as usize;
    let mut builder = ListBuilder::new(Int64Builder::new());
    let mut cache: Option<(usize, Vec<f64>)> = None;
    let offsets = list.value_offsets();

    for row in 0..list.len() {
        let (start, end) = (offsets[row] as usize, offsets[row + 1] as usize);
        let dim = end - start;
        if list.is_null(row) || dim == 0 {
            builder.append_null();
            continue;
        }
        // The projection depends only on `dim`, so a homogeneous column builds it once.
        if cache.as_ref().is_none_or(|(cached, _)| *cached != dim) {
            cache = Some((dim, projection(seed as u64, bits, dim)));
        }
        let matrix = &cache.as_ref().expect("just populated").1;

        for bit in 0..bits {
            let plane = &matrix[bit * dim..(bit + 1) * dim];
            let mut dot = 0.0f64;
            for (d, &normal) in plane.iter().enumerate() {
                let idx = start + d;
                if values.is_valid(idx) {
                    dot += normal * values.value(idx);
                }
            }
            // `>= 0` puts the degenerate zero vector on one side rather than nowhere.
            builder.values().append_value(i64::from(dot >= 0.0));
        }
        builder.append(true);
    }
    Ok(Arc::new(builder.finish()) as ArrayRef)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{AsArray, Float64Builder, ListBuilder as LB};
    use arrow::datatypes::Int64Type;

    fn list_of(rows: &[Option<Vec<f64>>]) -> ArrayRef {
        let mut b = LB::new(Float64Builder::new());
        for row in rows {
            match row {
                Some(values) => {
                    for v in values {
                        b.values().append_value(*v);
                    }
                    b.append(true);
                }
                None => b.append_null(),
            }
        }
        Arc::new(b.finish()) as ArrayRef
    }

    fn signature(out: &ArrayRef, row: usize) -> Option<Vec<i64>> {
        let list = out.as_list::<i32>();
        if list.is_null(row) {
            return None;
        }
        let v = list.value(row);
        let v = v.as_primitive::<Int64Type>();
        Some((0..v.len()).map(|i| v.value(i)).collect())
    }

    /// Fraction of positions two signatures agree on — the `list.jaccard` estimator.
    fn agreement(a: &[i64], b: &[i64]) -> f64 {
        let same = a.iter().zip(b).filter(|(x, y)| x == y).count();
        same as f64 / a.len() as f64
    }

    #[test]
    fn the_signature_has_num_bits_zero_one_values() {
        let out = eval_list_simhash(&list_of(&[Some(vec![1.0, 2.0, 3.0])]), 16, 0).unwrap();
        let sig = signature(&out, 0).unwrap();
        assert_eq!(sig.len(), 16);
        assert!(sig.iter().all(|b| *b == 0 || *b == 1));
    }

    #[test]
    fn identical_vectors_have_identical_signatures() {
        let out = eval_list_simhash(
            &list_of(&[Some(vec![1.0, 2.0]), Some(vec![1.0, 2.0])]),
            64,
            0,
        )
        .unwrap();
        assert_eq!(signature(&out, 0), signature(&out, 1));
    }

    #[test]
    fn the_signature_is_scale_invariant() {
        // SimHash sees only direction: `v` and `10v` are the same point on the sphere.
        let out = eval_list_simhash(
            &list_of(&[Some(vec![1.0, 2.0]), Some(vec![10.0, 20.0])]),
            64,
            0,
        )
        .unwrap();
        assert_eq!(signature(&out, 0), signature(&out, 1));
    }

    #[test]
    fn an_antipodal_vector_inverts_every_bit() {
        let out = eval_list_simhash(
            &list_of(&[Some(vec![1.0, 2.0]), Some(vec![-1.0, -2.0])]),
            64,
            0,
        )
        .unwrap();
        let (a, b) = (signature(&out, 0).unwrap(), signature(&out, 1).unwrap());
        // dot(-v) = -dot(v); only an exact 0 lands on the same side, and none do here.
        assert_eq!(agreement(&a, &b), 0.0);
    }

    #[test]
    fn agreement_tracks_the_angle_between_the_vectors() {
        // P(bits agree) = 1 - theta/pi. Orthogonal vectors (theta = pi/2) agree ~half
        // the time; a near-parallel pair agrees almost always. 512 bits keeps the
        // sampling error small enough to assert on.
        let out = eval_list_simhash(
            &list_of(&[
                Some(vec![1.0, 0.0]),
                Some(vec![0.0, 1.0]),  // orthogonal: theta = pi/2
                Some(vec![1.0, 0.01]), // ~0.57 degrees off
            ]),
            512,
            7,
        )
        .unwrap();
        let (a, orth, near) = (
            signature(&out, 0).unwrap(),
            signature(&out, 1).unwrap(),
            signature(&out, 2).unwrap(),
        );
        let orth_agreement = agreement(&a, &orth);
        assert!(
            (0.40..=0.60).contains(&orth_agreement),
            "orthogonal pair should agree ~0.5, got {orth_agreement}"
        );
        assert!(
            agreement(&a, &near) > 0.95,
            "near-parallel pair should agree almost always"
        );
        assert!(agreement(&a, &near) > orth_agreement);
    }

    #[test]
    fn a_null_or_empty_list_has_no_direction_and_yields_null() {
        let out = eval_list_simhash(&list_of(&[None, Some(vec![])]), 8, 0).unwrap();
        assert_eq!(signature(&out, 0), None);
        assert_eq!(signature(&out, 1), None);
    }

    #[test]
    fn the_signature_is_deterministic_across_calls() {
        let a = eval_list_simhash(&list_of(&[Some(vec![3.0, -1.0, 0.5])]), 32, 0).unwrap();
        let b = eval_list_simhash(&list_of(&[Some(vec![3.0, -1.0, 0.5])]), 32, 0).unwrap();
        assert_eq!(signature(&a, 0), signature(&b, 0));
    }

    #[test]
    fn the_seed_selects_a_different_set_of_hyperplanes() {
        let a = eval_list_simhash(&list_of(&[Some(vec![3.0, -1.0, 0.5])]), 64, 0).unwrap();
        let b = eval_list_simhash(&list_of(&[Some(vec![3.0, -1.0, 0.5])]), 64, 1).unwrap();
        assert_ne!(signature(&a, 0), signature(&b, 0));
    }

    #[test]
    fn the_signature_does_not_depend_on_batch_boundaries() {
        // Row 1 alone must hash exactly as it does beside row 0.
        let together = eval_list_simhash(
            &list_of(&[Some(vec![1.0, 2.0]), Some(vec![-3.0, 4.0])]),
            32,
            0,
        )
        .unwrap();
        let alone = eval_list_simhash(&list_of(&[Some(vec![-3.0, 4.0])]), 32, 0).unwrap();
        assert_eq!(signature(&together, 1), signature(&alone, 0));
    }

    #[test]
    fn rows_of_differing_dimension_each_get_their_own_projection() {
        let out = eval_list_simhash(
            &list_of(&[Some(vec![1.0, 2.0]), Some(vec![1.0, 2.0, 3.0])]),
            8,
            0,
        )
        .unwrap();
        assert_eq!(signature(&out, 0).unwrap().len(), 8);
        assert_eq!(signature(&out, 1).unwrap().len(), 8);
    }

    #[test]
    fn a_null_element_reads_as_zero_rather_than_nulling_the_row() {
        let mut b = LB::new(Float64Builder::new());
        b.values().append_value(1.0);
        b.values().append_null();
        b.append(true);
        let with_null = Arc::new(b.finish()) as ArrayRef;
        let out = eval_list_simhash(&with_null, 16, 0).unwrap();
        let padded = eval_list_simhash(&list_of(&[Some(vec![1.0, 0.0])]), 16, 0).unwrap();
        assert_eq!(signature(&out, 0), signature(&padded, 0));
    }

    #[test]
    fn an_out_of_range_bit_count_is_rejected() {
        for bits in [0, -1, MAX_BITS + 1] {
            assert!(eval_list_simhash(&list_of(&[Some(vec![1.0])]), bits, 0).is_err());
        }
    }

    #[test]
    fn an_integer_list_column_is_accepted() {
        let mut b = ListBuilder::new(Int64Builder::new());
        b.values().append_value(3);
        b.values().append_value(-1);
        b.append(true);
        let ints = Arc::new(b.finish()) as ArrayRef;
        let out = eval_list_simhash(&ints, 16, 0).unwrap();
        let floats = eval_list_simhash(&list_of(&[Some(vec![3.0, -1.0])]), 16, 0).unwrap();
        assert_eq!(signature(&out, 0), signature(&floats, 0));
    }
}
