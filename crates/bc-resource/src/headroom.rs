//! How much memory the machine has left, read rather than estimated.
//!
//! # Why the estimates are not enough
//!
//! Every memory guard in the executors compares an *estimate* of what an operator holds with
//! a budget: a breaker counts the bytes of the batches it collected, the build cache counts
//! its sides and hash tables. That is the right first line -- it decides before allocating --
//! but it cannot see what it does not count: decode buffers in flight, a prefetched unit, the
//! allocator's retained pages, the result Python is still holding, another process on the
//! node. TPC-H q9 at sf1000 on four 64 GB workers ran two unit tasks per node, each handed a
//! budget of about 22 GB, and each grew to 30-35 GB of resident memory before the kernel
//! killed it; every node lost its tasks in a loop and the query failed with a transport error
//! from the shuffle the dead tasks were feeding.
//!
//! So this module reads the one number the kernel acts on -- how much memory is still
//! available, to the machine and to this process's container -- and reports when it falls
//! below a floor. The executors treat that exactly like an estimate over budget: the
//! streaming executor hands the plan to the materializing one, the memory pool refuses a
//! reservation so the operator asking spills, and the materializing executor gives way to the
//! control plane's out-of-core route. A query slows down instead of dying.
//!
//! # What "available" means
//!
//! The smallest of:
//!
//! * the kernel's `MemAvailable` (free memory plus the page cache it can reclaim), and
//! * for each cgroup v2 level that limits this process (the mount root and the process's own
//!   leaf, which differ for a Ray worker under a systemd slice): `memory.max` less what it
//!   holds that cannot be reclaimed, `memory.current - (file - shmem)`. Page cache, active or
//!   not, is reclaimed before the cgroup's OOM killer runs, so a scan that re-reads a few
//!   hundred GB of Parquet must not read as pressure; shared memory is not, and on a Ray node
//!   it is the object store.
//!
//! The floor is a fraction of the tightest total ([`FLOOR_FRACTION`], at least
//! [`FLOOR_MIN_BYTES`]). It sits above Ray's own memory monitor, which kills workers at 95% of
//! the node, so the engine yields first.
//!
//! # Cost
//!
//! The guard is armed by the first query that runs with a memory budget. Arming starts one
//! sampler thread that re-reads the files every [`SAMPLE_INTERVAL`]; the executors' check is
//! a relaxed load of the last sample, so a morsel loop pays one atomic read. An unarmed
//! process (no budget configured, a `cargo test`) never starts the thread and never trips.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Once;
use std::time::Duration;

/// Fraction of the tightest memory total below which the guard trips.
pub const FLOOR_FRACTION: f64 = 0.08;

/// The floor's lower bound in bytes, so a small container still keeps a working margin.
pub const FLOOR_MIN_BYTES: u64 = 1 << 30;

/// How often the sampler re-reads the kernel's figures. Allocation outruns a slower sampler:
/// a decode loop can take a gigabyte in well under a hundred milliseconds.
pub const SAMPLE_INTERVAL: Duration = Duration::from_millis(5);

/// How close to the floor the sampler starts asking the allocator for its retained pages: at
/// `RECLAIM_MARGIN` times the floor. Giving retained pages back is far cheaper than spilling,
/// so it is tried well before the executors would give way.
pub const RECLAIM_MARGIN: u64 = 2;

/// The least time between two reclaims, so a node that stays near its floor is not walked on
/// every sample: a forced collect visits every thread's heap.
pub const RECLAIM_INTERVAL: Duration = Duration::from_millis(250);

/// Hands the process allocator's retained free pages back to the OS (`set_reclaimer`).
static RECLAIMER: std::sync::OnceLock<fn()> = std::sync::OnceLock::new();

/// `u64::MAX` until the first sample: "unknown" must never read as "low".
static AVAILABLE: AtomicU64 = AtomicU64::new(u64::MAX);
static FLOOR: AtomicU64 = AtomicU64::new(0);
static ARMED: AtomicBool = AtomicBool::new(false);
/// A test-only stand-in for the sampled figure; `u64::MAX` means "use the sample".
static FORCED: AtomicU64 = AtomicU64::new(u64::MAX);
static START: Once = Once::new();

/// The machine's memory position when it is low: what is available, and the floor it fell
/// below.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Headroom {
    /// Bytes still available to this process (see the module docs).
    pub available: u64,
    /// The floor the guard trips under.
    pub floor: u64,
}

/// Start sampling, once per process. Idempotent and cheap after the first call.
///
/// Does nothing where the kernel's figures cannot be read (no `/proc/meminfo`), so the guard
/// stays unarmed rather than tripping on a missing file.
pub fn arm() {
    START.call_once(|| {
        let Some(first) = sample() else {
            return;
        };
        FLOOR.store(first.floor, Ordering::Relaxed);
        AVAILABLE.store(first.available, Ordering::Relaxed);
        ARMED.store(true, Ordering::Release);
        let spawned = std::thread::Builder::new()
            .name("bc-headroom".into())
            .spawn(|| {
                let mut last_reclaim: Option<std::time::Instant> = None;
                loop {
                    std::thread::sleep(SAMPLE_INTERVAL);
                    let Some(mut s) = sample() else { continue };
                    let near = s.available < s.floor.saturating_mul(RECLAIM_MARGIN);
                    let due = last_reclaim.is_none_or(|t| t.elapsed() >= RECLAIM_INTERVAL);
                    if near && due {
                        if let Some(reclaim) = RECLAIMER.get() {
                            reclaim();
                            last_reclaim = Some(std::time::Instant::now());
                            s = sample().unwrap_or(s);
                        }
                    }
                    AVAILABLE.store(s.available, Ordering::Relaxed);
                    FLOOR.store(s.floor, Ordering::Relaxed);
                }
            });
        if spawned.is_err() {
            // No thread, no fresh samples: a stale reading could only ever trip wrongly.
            ARMED.store(false, Ordering::Release);
        }
    });
}

/// Register how this process hands its allocator's retained free pages back to the OS.
///
/// The sampler calls it when available memory comes within [`RECLAIM_MARGIN`] times the floor,
/// at most every [`RECLAIM_INTERVAL`]. An allocator that keeps freed pages for reuse -- the
/// engine's mimalloc holds them ten seconds, which is what lets consecutive queries reuse their
/// buffers -- otherwise keeps counting memory nothing is using: a distributed TPC-H q9 unit task
/// at SF1000 read a 7 GB unit at 34 GB resident, and Ray's memory monitor killed it at 95% of
/// the node. This crate has no allocator of its own, so the caller that owns one supplies it.
/// The first registration wins.
pub fn set_reclaimer(reclaim: fn()) {
    let _ = RECLAIMER.set(reclaim);
}

/// `Some` when the guard is armed and available memory is under the floor.
///
/// The hot-path read: two relaxed loads when memory is fine.
#[must_use]
#[inline]
pub fn low() -> Option<Headroom> {
    let forced = FORCED.load(Ordering::Relaxed);
    let available = if forced != u64::MAX {
        forced
    } else if ARMED.load(Ordering::Relaxed) {
        AVAILABLE.load(Ordering::Relaxed)
    } else {
        return None;
    };
    verdict(available, FLOOR.load(Ordering::Relaxed))
}

/// The current reading, low or not: arms the guard if it is not armed yet, and `None` only
/// where the kernel's figures cannot be read.
///
/// For a caller deciding whether it can afford an allocation it controls -- a unit task
/// choosing to prefetch its next unit -- rather than for the executors' trip check.
#[must_use]
pub fn reading() -> Option<Headroom> {
    arm();
    let forced = FORCED.load(Ordering::Relaxed);
    if forced == u64::MAX && !ARMED.load(Ordering::Relaxed) {
        return None;
    }
    let available = if forced != u64::MAX {
        forced
    } else {
        AVAILABLE.load(Ordering::Relaxed)
    };
    let floor = FLOOR.load(Ordering::Relaxed).max(FLOOR_MIN_BYTES);
    Some(Headroom { available, floor })
}

/// Low when `available` is under `floor` (never under [`FLOOR_MIN_BYTES`]).
fn verdict(available: u64, floor: u64) -> Option<Headroom> {
    let floor = floor.max(FLOOR_MIN_BYTES);
    (available < floor).then_some(Headroom { available, floor })
}

/// Pretend the machine has `available` bytes left (`None` restores the real reading).
///
/// For tests in this crate and the executors': the only way to drive the low-memory paths
/// deterministically is to say what the sampler would have read. Process-global, like the
/// guard itself, so a test that sets it must clear it.
#[doc(hidden)]
pub fn force_available(available: Option<u64>) {
    FORCED.store(available.unwrap_or(u64::MAX), Ordering::Relaxed);
}

/// One reading: available bytes and the floor, or `None` when `/proc/meminfo` is unreadable.
fn sample() -> Option<Headroom> {
    let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
    let mut available = meminfo_kib(&meminfo, "MemAvailable:")?.saturating_mul(1024);
    let mut total = meminfo_kib(&meminfo, "MemTotal:")?.saturating_mul(1024);
    for dir in cgroup_dirs() {
        if let Some((limit, unreclaimable)) = cgroup_usage(dir) {
            available = available.min(limit.saturating_sub(unreclaimable));
            total = total.min(limit);
        }
    }
    let floor = ((total as f64 * FLOOR_FRACTION) as u64).max(FLOOR_MIN_BYTES);
    Some(Headroom { available, floor })
}

/// The value in KiB of a `/proc/meminfo` line starting with `key`.
fn meminfo_kib(text: &str, key: &str) -> Option<u64> {
    text.lines()
        .find(|l| l.starts_with(key))?
        .split_whitespace()
        .nth(1)?
        .parse()
        .ok()
}

/// The cgroup v2 directories whose memory limit binds this process: the mount root, the
/// process's own cgroup, and every level between them. Resolved once: a process does not
/// change cgroup while it runs.
fn cgroup_dirs() -> &'static [String] {
    static DIRS: std::sync::OnceLock<Vec<String>> = std::sync::OnceLock::new();
    DIRS.get_or_init(|| {
        let own = std::fs::read_to_string("/proc/self/cgroup").unwrap_or_default();
        resolve_cgroup_dirs(std::path::Path::new("/sys/fs/cgroup"), &own)
    })
}

/// [`cgroup_dirs`] for a mount at `root` and a `/proc/self/cgroup` text.
///
/// The `0::` path is relative to the cgroup root the *kernel* sees, which inside a container is
/// not the mount: an Anyscale node reports `/anyscale/ctr_<id>/workers` while the container's
/// own cgroup is mounted at `/sys/fs/cgroup`, so the leaf is `/sys/fs/cgroup/workers`. Joining
/// the path onto the mount named a directory that does not exist, the leaf was skipped, and the
/// guard never saw the `workers` cgroup's 75% limit -- the one that killed a TPC-H q9 unit task
/// at SF1000 while the node still had 7 GB available. So leading components are dropped until
/// the path exists under the mount, and every level from there up to the mount is read, since a
/// limit set on any of them binds the process.
fn resolve_cgroup_dirs(root: &std::path::Path, proc_cgroup: &str) -> Vec<String> {
    let mut dirs = vec![root.to_string_lossy().into_owned()];
    let Some(path) = proc_cgroup.lines().find_map(|l| l.strip_prefix("0::")) else {
        return dirs;
    };
    let parts: Vec<&str> = path.trim().split('/').filter(|p| !p.is_empty()).collect();
    let leaf = (0..parts.len())
        .map(|skip| {
            parts[skip..]
                .iter()
                .fold(root.to_path_buf(), |d, p| d.join(p))
        })
        .find(|d| d.is_dir());
    let mut level = leaf;
    while let Some(dir) = level {
        if dir == root {
            break;
        }
        dirs.push(dir.to_string_lossy().into_owned());
        level = dir.parent().map(std::path::Path::to_path_buf);
    }
    dirs
}

/// `(memory.max, memory.current - (file - shmem))` for a cgroup v2 directory with a finite
/// limit; `None` for an unlimited one or one that cannot be read.
fn cgroup_usage(dir: &str) -> Option<(u64, u64)> {
    let max = std::fs::read_to_string(format!("{dir}/memory.max")).ok()?;
    let limit: u64 = max.trim().parse().ok()?; // "max" (unlimited) fails to parse: skipped
    let current: u64 = std::fs::read_to_string(format!("{dir}/memory.current"))
        .ok()?
        .trim()
        .parse()
        .ok()?;
    let stat = std::fs::read_to_string(format!("{dir}/memory.stat")).unwrap_or_default();
    let reclaimable = stat_field(&stat, "file ").saturating_sub(stat_field(&stat, "shmem "));
    Some((limit, current.saturating_sub(reclaimable)))
}

/// A `memory.stat` field's value in bytes, `0` when absent.
fn stat_field(stat: &str, key: &str) -> u64 {
    stat.lines()
        .find_map(|l| l.strip_prefix(key))
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn meminfo_lines_parse_to_kib() {
        let text = "MemTotal:       32139936 kB\nMemFree: 1 kB\nMemAvailable:   25233032 kB\n";
        assert_eq!(meminfo_kib(text, "MemTotal:"), Some(32_139_936));
        assert_eq!(meminfo_kib(text, "MemAvailable:"), Some(25_233_032));
        assert_eq!(meminfo_kib(text, "Missing:"), None);
    }

    #[test]
    fn low_only_below_the_floor_and_never_below_the_minimum() {
        // Not through `force_available`: it is process-global, and the pool tests in this
        // crate run concurrently (the executors' binary `headroom_guard` covers forcing).
        assert_eq!(verdict(10 << 30, 4 << 30), None);
        assert_eq!(
            verdict(3 << 30, 4 << 30),
            Some(Headroom {
                available: 3 << 30,
                floor: 4 << 30
            })
        );
        let tiny_floor = verdict(FLOOR_MIN_BYTES / 2, 1).expect("the minimum floor applies");
        assert_eq!(tiny_floor.floor, FLOOR_MIN_BYTES);
    }

    #[test]
    fn page_cache_is_reclaimable_but_shared_memory_is_not() {
        let stat = "anon 100\nfile 900\nfile_mapped 5\nshmem 300\n";
        assert_eq!(stat_field(stat, "file "), 900, "not `file_mapped`");
        assert_eq!(stat_field(stat, "shmem "), 300);
        assert_eq!(stat_field(stat, "absent "), 0);
    }

    #[test]
    fn a_container_cgroup_path_resolves_under_its_mount_with_every_level() {
        let root = std::env::temp_dir().join(format!("bc-headroom-{}", std::process::id()));
        let workers = root.join("workers").join("w1");
        std::fs::create_dir_all(&workers).unwrap();
        let s = |p: &std::path::Path| p.to_string_lossy().into_owned();
        // The kernel's path carries the host's prefix; the mount does not.
        let dirs = resolve_cgroup_dirs(&root, "0::/anyscale/ctr_abc/workers/w1\n");
        assert_eq!(dirs, vec![s(&root), s(&workers), s(&root.join("workers"))]);
        // A path the mount does not hold at all falls back to the mount alone.
        assert_eq!(
            resolve_cgroup_dirs(&root, "0::/nowhere/else\n"),
            vec![s(&root)]
        );
        // The root itself (a cgroup namespace) adds nothing past the mount.
        assert_eq!(resolve_cgroup_dirs(&root, "0::/\n"), vec![s(&root)]);
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn a_real_sample_is_finite_on_linux() {
        if let Some(s) = sample() {
            assert!(s.available > 0 && s.available < u64::MAX);
            assert!(s.floor >= FLOOR_MIN_BYTES);
        }
    }
}
