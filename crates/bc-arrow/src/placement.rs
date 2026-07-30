//! Which CPU a worker thread should run on.
//!
//! Pinning a pool of `n` workers to CPUs is usually written as `cpu = worker_index % n_cpus`.
//! That is wrong twice over on the machines this engine targets.
//!
//! It is wrong about **which ids exist**. A cgroup or a `taskset` narrows the process to an
//! arbitrary CPU set — `48-95` on the second socket of a two-socket box is a completely
//! ordinary shape under a Ray placement group. Indexing `0..n` then names CPUs the process
//! may not use, `sched_setaffinity` refuses each one, and every worker silently ends up
//! unpinned: the feature reports success and does nothing.
//!
//! It is wrong about **which ids are good**. Linux enumerates SMT siblings adjacently on some
//! parts (`cpu0`/`cpu1` one core) and by halves on others (`cpu0`/`cpu48`). With eight workers
//! on the first layout, `0..8` fills four physical cores twice over and leaves the rest of the
//! machine idle — the pool runs at half throughput with every pair fighting over one L1. And
//! on a two-node host, filling node 0 before touching node 1 puts every worker on one memory
//! controller while the other sits unused.
//!
//! [`pinning_order`] answers both by building the assignment order from the real topology:
//! stride across NUMA nodes, take one CPU per physical core before any second sibling, and
//! only ever name CPUs in the affinity mask. Worker `i` takes `order[i % order.len()]`.

use crate::topology::{affinity_cpus, numa_node_cpus};

/// The CPU ids to pin worker threads to, in assignment order.
///
/// Position `i` is the CPU for worker `i` (callers wrap with `%` past the end). The order is:
///
/// 1. **One CPU per physical core before any SMT sibling.** The second thread of a core shares
///    its L1, L2 and execution units, so it is the last resort, not the second choice.
/// 2. **Round-robin across NUMA nodes.** Consecutive workers land on different memory
///    controllers, so a pool narrower than the machine still spreads its bandwidth demand
///    instead of saturating one node.
/// 3. **Ascending CPU id within a node**, so the order is deterministic and two processes
///    reading it agree.
///
/// Only CPUs in this process's affinity mask appear. Returns an empty vector when the topology
/// cannot be read, which callers must treat as "do not pin" rather than falling back to a
/// modulo over the core count — an unpinned thread is strictly better than one pinned to a
/// CPU chosen by guesswork.
pub fn pinning_order() -> Vec<usize> {
    let allowed = affinity_cpus();
    let siblings = core_groups(&allowed);
    if siblings.is_empty() {
        return Vec::new();
    }

    // Group physical cores by the NUMA node their first CPU belongs to, preserving each
    // node's ascending CPU order.
    let node_of = node_lookup();
    let mut by_node: Vec<(usize, Vec<Vec<usize>>)> = Vec::new();
    for group in siblings {
        let node = node_of(group[0]);
        match by_node.iter_mut().find(|(n, _)| *n == node) {
            Some((_, cores)) => cores.push(group),
            None => by_node.push((node, vec![group])),
        }
    }
    by_node.sort_by_key(|(node, _)| *node);

    // Emit sibling rank 0 of every core (striding nodes), then rank 1, and so on. The widest
    // core decides how many ranks there are; a core with fewer siblings simply stops
    // contributing, which is what makes this correct on a machine with mixed SMT (an
    // Alder Lake P/E split, where the E-cores have no sibling).
    let max_rank = by_node
        .iter()
        .flat_map(|(_, cores)| cores.iter().map(|c| c.len()))
        .max()
        .unwrap_or(1);
    let mut order = Vec::new();
    for rank in 0..max_rank {
        // `core_idx` strides across nodes: node0's core0, node1's core0, node0's core1, ...
        let widest = by_node
            .iter()
            .map(|(_, cores)| cores.len())
            .max()
            .unwrap_or(0);
        for core_idx in 0..widest {
            for (_, cores) in &by_node {
                if let Some(cpu) = cores.get(core_idx).and_then(|c| c.get(rank)) {
                    order.push(*cpu);
                }
            }
        }
    }
    order
}

/// The CPU id for worker `idx`, or `None` when the topology is unreadable.
///
/// The one-call form of [`pinning_order`] for a caller that pins a single thread. Prefer
/// hoisting the order out of a loop when pinning a whole pool — this recomputes it.
pub fn cpu_for_worker(idx: usize) -> Option<usize> {
    let order = pinning_order();
    (!order.is_empty()).then(|| order[idx % order.len()])
}

/// Physical cores as groups of usable sibling CPU ids, each group ascending, groups ordered by
/// their lowest CPU id.
///
/// Falls back to one group per allowed CPU when `thread_siblings_list` is unreadable, which
/// degrades this to "spread over every CPU" — the pre-SMT-aware behavior, and still correct.
#[cfg(target_os = "linux")]
fn core_groups(allowed: &Option<Vec<usize>>) -> Vec<Vec<usize>> {
    use std::collections::BTreeSet;
    let allow: Option<BTreeSet<usize>> = allowed.as_ref().map(|v| v.iter().copied().collect());
    let cpus: Vec<usize> = match allowed {
        Some(v) => v.clone(),
        None => (0..crate::usable_cores()).collect(),
    };
    let mut groups: BTreeSet<Vec<usize>> = BTreeSet::new();
    for cpu in &cpus {
        let path = format!("/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list");
        let mut sibs: Vec<usize> = std::fs::read_to_string(&path)
            .ok()
            .map(|s| crate::topology::parse_cpu_list_public(&s))
            .unwrap_or_default();
        if let Some(allow) = &allow {
            sibs.retain(|c| allow.contains(c));
        }
        if sibs.is_empty() {
            sibs = vec![*cpu];
        }
        sibs.sort_unstable();
        groups.insert(sibs);
    }
    groups.into_iter().collect()
}

#[cfg(not(target_os = "linux"))]
fn core_groups(allowed: &Option<Vec<usize>>) -> Vec<Vec<usize>> {
    match allowed {
        Some(v) => v.iter().map(|c| vec![*c]).collect(),
        None => (0..crate::usable_cores()).map(|c| vec![c]).collect(),
    }
}

/// A closure mapping a CPU id to its NUMA node, defaulting to node 0.
///
/// Built once from the node map rather than read per CPU, so building a pinning order on a
/// 192-CPU host is one directory walk instead of 192 file reads.
fn node_lookup() -> impl Fn(usize) -> usize {
    let nodes = numa_node_cpus();
    move |cpu: usize| {
        nodes
            .iter()
            .find(|(_, cpus)| cpus.binary_search(&cpu).is_ok())
            .map(|(id, _)| *id)
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn the_order_only_names_cpus_this_process_may_use() {
        let order = pinning_order();
        if order.is_empty() {
            return; // topology unreadable: "do not pin" is the contract.
        }
        if let Some(allowed) = affinity_cpus() {
            let allowed: HashSet<usize> = allowed.into_iter().collect();
            for cpu in &order {
                assert!(
                    allowed.contains(cpu),
                    "cpu {cpu} is not in the affinity mask; pinning to it is a silent no-op"
                );
            }
        }
    }

    #[test]
    fn every_usable_cpu_appears_exactly_once() {
        let order = pinning_order();
        if order.is_empty() {
            return;
        }
        let unique: HashSet<usize> = order.iter().copied().collect();
        assert_eq!(
            unique.len(),
            order.len(),
            "a repeated cpu would oversubscribe one core while leaving another idle"
        );
        if let Some(allowed) = affinity_cpus() {
            assert_eq!(
                unique.len(),
                allowed.len(),
                "the order must cover the whole mask, or the tail of the machine never runs"
            );
        }
    }

    #[test]
    fn physical_cores_are_filled_before_smt_siblings() {
        // The property that matters: taking the first `physical_cores` entries must land on
        // that many *distinct* physical cores, never twice on one. This is the failure the
        // naive `idx % n_cpus` has on a host that enumerates siblings adjacently.
        let order = pinning_order();
        let topo = crate::CpuTopology::detect();
        if order.is_empty() || topo.smt_width() < 2 {
            return;
        }
        let allowed = affinity_cpus();
        let groups = core_groups(&allowed);
        let core_of = |cpu: usize| groups.iter().position(|g| g.contains(&cpu));
        let prefix = &order[..topo.physical_cores.min(order.len())];
        let cores: HashSet<Option<usize>> = prefix.iter().map(|c| core_of(*c)).collect();
        assert_eq!(
            cores.len(),
            prefix.len(),
            "the first {} workers must occupy {} distinct physical cores",
            prefix.len(),
            prefix.len()
        );
    }

    #[test]
    fn consecutive_workers_stride_across_numa_nodes() {
        let order = pinning_order();
        let topo = crate::CpuTopology::detect();
        if order.len() < 2 || !topo.is_numa() {
            return;
        }
        let node_of = node_lookup();
        assert_ne!(
            node_of(order[0]),
            node_of(order[1]),
            "on a multi-node host, two workers must not both land on node {}",
            node_of(order[0])
        );
    }

    #[test]
    fn worker_lookup_wraps_and_matches_the_order() {
        let order = pinning_order();
        if order.is_empty() {
            assert_eq!(cpu_for_worker(0), None);
            return;
        }
        assert_eq!(cpu_for_worker(0), Some(order[0]));
        assert_eq!(cpu_for_worker(order.len()), Some(order[0]));
        assert_eq!(
            cpu_for_worker(order.len() * 3 + 1),
            Some(order[1 % order.len()])
        );
    }

    #[test]
    fn core_groups_are_sorted_and_disjoint() {
        let groups = core_groups(&affinity_cpus());
        let mut seen: HashSet<usize> = HashSet::new();
        for g in &groups {
            assert!(!g.is_empty());
            let mut sorted = g.clone();
            sorted.sort_unstable();
            assert_eq!(*g, sorted, "each sibling group must be ascending");
            for cpu in g {
                assert!(seen.insert(*cpu), "cpu {cpu} appears in two physical cores");
            }
        }
    }

    #[test]
    fn the_order_covers_at_least_the_physical_core_count() {
        let order = pinning_order();
        if order.is_empty() {
            return;
        }
        let topo = crate::CpuTopology::detect();
        assert!(
            order.len() >= topo.physical_cores,
            "an order shorter than the physical core count cannot fill the machine"
        );
    }
}
