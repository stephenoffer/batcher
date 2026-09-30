//! Put the driving scan on the probe spine, and map the metrics of the re-oriented plan back.
//!
//! The chunked and units executors stream their driving relation through a join's probe side,
//! so an inner join that builds on it is swapped first ([`orient`]). The control plane files each
//! operator metric under the pre-order id of the plan *it* built, so a swapped plan's ids are
//! translated back ([`original_ids`]) rather than dropped.

use std::collections::HashMap;

use bc_ir::{JoinType, RelOp};

/// `plan` with every inner join whose *right* (build) input holds the driving scan swapped, so
/// the driving source ends up on the probe spine.
///
/// Kyber builds the smaller side, and a filtered fact table is sometimes the smaller one — TPC-H
/// q3 at sf100 hashes the date-filtered `lineitem` and probes it with the customer-orders join.
/// Streaming needs the fact table on the probe side, and an inner join is commutative: the swap
/// re-labels which input each output column is read from and changes no row. Outer, semi and
/// anti joins are not commutative and are left alone (`chunkable` then declines them if the
/// driving scan sits on their build side).
pub(super) fn orient(plan: &RelOp, driving: usize) -> RelOp {
    orient_counted(plan, driving).0
}

/// [`orient`], with how many joins it swapped.
pub(super) fn orient_counted(plan: &RelOp, driving: usize) -> (RelOp, usize) {
    let mut out = plan.clone();
    let swaps = orient_in_place(&mut out, driving);
    (out, swaps)
}

fn orient_in_place(plan: &mut RelOp, driving: usize) -> usize {
    let mut swaps = 0;
    if let RelOp::HashJoin {
        left,
        right,
        left_keys,
        right_keys,
        join_type: JoinType::Inner,
        output,
        ..
    } = plan
    {
        if scans_of(right, driving) > 0 && scans_of(left, driving) == 0 {
            swaps += 1;
            std::mem::swap(left, right);
            std::mem::swap(left_keys, right_keys);
            for col in output.iter_mut() {
                col.side = match col.side {
                    bc_ir::JoinSide::Left => bc_ir::JoinSide::Right,
                    bc_ir::JoinSide::Right => bc_ir::JoinSide::Left,
                };
            }
        }
    }
    swaps
        + match plan {
            RelOp::HashJoin { left, .. } => orient_in_place(left, driving),
            RelOp::Filter { input, .. }
            | RelOp::Project { input, .. }
            | RelOp::Aggregate { input, .. }
            | RelOp::Sort { input, .. }
            | RelOp::Limit { input, .. } => orient_in_place(input, driving),
            _ => 0,
        }
}

/// Whether `spine` reaches `Scan { driving }` through probe-side-streamable nodes only.
pub(super) fn probe_spine_reaches(spine: &RelOp, driving: usize) -> bool {
    let mut node = spine;
    loop {
        match node {
            RelOp::Scan { source_id } => return *source_id == driving,
            RelOp::Filter { input, .. } | RelOp::Project { input, .. } => node = input,
            RelOp::HashJoin {
                left,
                join_type: JoinType::Inner | JoinType::Left | JoinType::Semi | JoinType::Anti,
                ..
            } => node = left,
            _ => return false,
        }
    }
}

pub(super) fn scans_of(plan: &RelOp, source: usize) -> usize {
    let own = usize::from(matches!(plan, RelOp::Scan { source_id } if *source_id == source));
    own + plan
        .children()
        .into_iter()
        .map(|c| scans_of(c, source))
        .sum::<usize>()
}

/// For each node of `oriented`, in its pre-order, the pre-order id of the same node in
/// `original` — the plan `orient` swapped the driving spine's joins of.
///
/// Mirrors `orient_in_place` exactly: only joins on the driving *spine* can have been swapped
/// (it recurses into a join's probe side and a unary node's input, nowhere else), and a spine
/// join was swapped iff the driving source is scanned under its right side and not its left.
pub(super) fn original_ids(original: &RelOp, oriented: &RelOp, driving: usize) -> Vec<u32> {
    fn number(node: &RelOp, next: &mut u32, ids: &mut HashMap<usize, u32>) {
        ids.insert(node as *const RelOp as usize, *next);
        *next += 1;
        for c in node.children() {
            number(c, next, ids);
        }
    }
    fn pair(
        orig: &RelOp,
        ran: &RelOp,
        spine: bool,
        driving: usize,
        ids: &HashMap<usize, u32>,
        out: &mut Vec<u32>,
    ) {
        out.push(ids[&(orig as *const RelOp as usize)]);
        let kids = orig.children();
        let ran_kids = ran.children();
        let (order, spines): (Vec<&RelOp>, Vec<bool>) = match orig {
            RelOp::HashJoin {
                left,
                right,
                join_type,
                ..
            } if spine => {
                let swapped = matches!(join_type, JoinType::Inner)
                    && scans_of(right, driving) > 0
                    && scans_of(left, driving) == 0;
                if swapped {
                    (vec![right.as_ref(), left.as_ref()], vec![true, false])
                } else {
                    (vec![left.as_ref(), right.as_ref()], vec![true, false])
                }
            }
            RelOp::Filter { .. }
            | RelOp::Project { .. }
            | RelOp::Aggregate { .. }
            | RelOp::Sort { .. }
            | RelOp::Limit { .. } => (kids.clone(), vec![spine; kids.len()]),
            _ => (kids.clone(), vec![false; kids.len()]),
        };
        for ((o, r), s) in order.iter().zip(ran_kids.iter()).zip(spines) {
            pair(o, r, s, driving, ids, out);
        }
    }
    let mut ids = HashMap::new();
    number(original, &mut 0, &mut ids);
    let mut out = Vec::new();
    pair(original, oriented, true, driving, &ids, &mut out);
    out
}
