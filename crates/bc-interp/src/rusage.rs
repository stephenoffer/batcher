//! Reading the operating system's own account of what this process consumed.
//!
//! Wall time and row counts describe what an operator *did*. They say nothing about what it
//! cost the machine, and the difference is where most unexplained slowness lives. Three
//! questions in particular cannot be answered from timing alone, and each has a counter here
//! that answers it directly:
//!
//! * **Was this operator's memory real?** A resident-set figure counts pages the process
//!   holds; it cannot say how many of them had to be fetched from disk to get there.
//!   `major_faults` can. An operator that faults in a hundred thousand pages is being paged
//!   against, which looks exactly like slow compute from the inside and is fixed by an
//!   entirely different action.
//! * **Were the cores this operator asked for actually its own?** Utilization cannot tell an
//!   under-parallelized operator from one that got preempted off the CPU every timeslice.
//!   `invol_ctx_switches` distinguishes them: it counts the times the scheduler took the core
//!   away, which happens when something else wanted it.
//! * **Did a scan read the disk or the page cache?** The same operator over the same file
//!   costs two orders of magnitude apart depending on the answer, and a cost model calibrated
//!   across both learns a coefficient true of neither. `io_read_bytes` counts only what
//!   actually reached the block device.
//!
//! All of it comes from two cheap, portable-enough sources — `getrusage(RUSAGE_SELF)` and
//! Linux's `/proc/self/io` — sampled at operator boundaries, never per row. Every field is
//! process-wide, which is sound here for the same reason the existing CPU-time field is: the
//! interpreter runs operators one at a time and fully joins each before recording it, so the
//! delta across an operator's window is that operator's own consumption across every worker
//! thread. Every field degrades to `0` where the platform cannot report it, and the control
//! plane reads `0` as "unmeasured" rather than "none".

/// One coherent snapshot of the process's resource consumption.
///
/// Read from a single `getrusage` call plus at most one `/proc/self/io` read, so the fields
/// are internally consistent: separate calls would straddle whatever work happened between
/// them and attribute it to the wrong operator.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct ResourceSample {
    /// Process CPU time (user + system, all threads) in nanoseconds.
    pub(crate) cpu_ns: u64,
    /// `ru_maxrss` normalized to bytes — a monotonic high-water mark, not a live reading.
    pub(crate) peak_rss_bytes: u64,
    /// Page faults served without disk I/O (`ru_minflt`): first touch of freshly allocated
    /// memory. The count scales with how much memory the operator newly *committed*.
    pub(crate) minor_faults: u64,
    /// Page faults that required disk I/O (`ru_majflt`): a page read back from swap or from
    /// a memory-mapped file. Any non-trivial count means the operator was waiting on storage
    /// for memory it believed it already had.
    pub(crate) major_faults: u64,
    /// Voluntary context switches (`ru_nvcsw`): the process gave up the CPU because it had
    /// to wait — for I/O, for a lock, for a queue. High against low CPU time means blocked,
    /// not idle.
    pub(crate) vol_ctx_switches: u64,
    /// Involuntary context switches (`ru_nivcsw`): the scheduler took the CPU away. The
    /// direct, per-operator measurement of contention for cores this process nominally owns.
    pub(crate) invol_ctx_switches: u64,
    /// Bytes actually fetched from a block device (`/proc/self/io` `read_bytes`). Excludes
    /// page-cache hits, which is exactly what makes it the honest input to an I/O cost model.
    pub(crate) io_read_bytes: u64,
    /// Bytes sent to a block device (`/proc/self/io` `write_bytes`), including spill writes.
    pub(crate) io_write_bytes: u64,
}

impl ResourceSample {
    /// This sample minus an earlier one, saturating so a counter that did not advance — or a
    /// high-water mark that was already set — reports `0` rather than wrapping.
    pub(crate) fn since(&self, start: &ResourceSample) -> ResourceSample {
        ResourceSample {
            cpu_ns: self.cpu_ns.saturating_sub(start.cpu_ns),
            peak_rss_bytes: self.peak_rss_bytes.saturating_sub(start.peak_rss_bytes),
            minor_faults: self.minor_faults.saturating_sub(start.minor_faults),
            major_faults: self.major_faults.saturating_sub(start.major_faults),
            vol_ctx_switches: self.vol_ctx_switches.saturating_sub(start.vol_ctx_switches),
            invol_ctx_switches: self
                .invol_ctx_switches
                .saturating_sub(start.invol_ctx_switches),
            io_read_bytes: self.io_read_bytes.saturating_sub(start.io_read_bytes),
            io_write_bytes: self.io_write_bytes.saturating_sub(start.io_write_bytes),
        }
    }
}

/// Sample every counter this platform can report, right now.
///
/// One `getrusage` syscall, plus one `/proc/self/io` read on Linux. Called twice per
/// operator, never per morsel or per row, so the cost is amortized over an operator's whole
/// working set. Unreadable counters stay `0`.
pub(crate) fn sample() -> ResourceSample {
    let mut out = rusage_sample();
    let (read_bytes, write_bytes) = proc_io_bytes();
    out.io_read_bytes = read_bytes;
    out.io_write_bytes = write_bytes;
    out
}

#[cfg(unix)]
fn rusage_sample() -> ResourceSample {
    use std::mem::MaybeUninit;

    let mut usage = MaybeUninit::<libc::rusage>::uninit();
    // SAFETY: `getrusage` fully initializes the `rusage` out-param and returns 0 on
    // success; the initialized value is read only on that success path.
    let rc = unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) };
    if rc != 0 {
        return ResourceSample::default();
    }
    let usage = unsafe { usage.assume_init() };
    let tv_ns = |t: &libc::timeval| (t.tv_sec as u64) * 1_000_000_000 + (t.tv_usec as u64) * 1_000;
    let max_rss = usage.ru_maxrss as u64;
    #[cfg(target_os = "linux")]
    let peak_rss_bytes = max_rss.saturating_mul(1024); // Linux reports KiB
    #[cfg(not(target_os = "linux"))]
    let peak_rss_bytes = max_rss; // the BSDs / macOS already report bytes
    ResourceSample {
        cpu_ns: tv_ns(&usage.ru_utime) + tv_ns(&usage.ru_stime),
        peak_rss_bytes,
        minor_faults: usage.ru_minflt as u64,
        major_faults: usage.ru_majflt as u64,
        vol_ctx_switches: usage.ru_nvcsw as u64,
        invol_ctx_switches: usage.ru_nivcsw as u64,
        io_read_bytes: 0,
        io_write_bytes: 0,
    }
}

#[cfg(not(unix))]
fn rusage_sample() -> ResourceSample {
    ResourceSample::default()
}

/// `(read_bytes, write_bytes)` from `/proc/self/io`, or `(0, 0)` off Linux.
///
/// These are the fields that count traffic that *reached the storage layer*, as opposed to
/// the sibling `rchar`/`wchar` fields, which count bytes the process passed to a syscall
/// whether or not they came from the page cache. The distinction is the whole point: a warm
/// scan and a cold scan issue identical syscalls and cost wildly different amounts, and only
/// these two fields tell them apart.
#[cfg(target_os = "linux")]
fn proc_io_bytes() -> (u64, u64) {
    let Ok(text) = std::fs::read_to_string("/proc/self/io") else {
        // Unreadable under a hardened kernel or a restricted container. `0` is the honest
        // answer, and the control plane keeps its prior rather than acting on a fabricated one.
        return (0, 0);
    };
    let mut read_bytes = 0;
    let mut write_bytes = 0;
    for line in text.lines() {
        let Some((key, value)) = line.split_once(':') else {
            continue;
        };
        let parsed = value.trim().parse::<u64>().unwrap_or(0);
        match key {
            "read_bytes" => read_bytes = parsed,
            "write_bytes" => write_bytes = parsed,
            _ => {}
        }
    }
    (read_bytes, write_bytes)
}

#[cfg(not(target_os = "linux"))]
fn proc_io_bytes() -> (u64, u64) {
    (0, 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A sample reports advancing CPU time on a busy span, and the fault counters are
    /// populated on unix (they are lifetime totals, so any real process has some).
    #[test]
    fn sample_reads_the_live_counters() {
        let first = sample();
        let mut acc: u64 = 0;
        for i in 0..20_000_000u64 {
            acc = acc.wrapping_add(i).wrapping_mul(2_654_435_761);
        }
        std::hint::black_box(acc);
        let second = sample();
        if cfg!(unix) {
            assert!(
                second.cpu_ns > first.cpu_ns,
                "a busy span must burn CPU time"
            );
            assert!(
                second.minor_faults > 0,
                "any live process has faulted some pages in"
            );
        }
    }

    /// Differencing saturates rather than wrapping. A counter that did not advance — the
    /// common case for major faults on a healthy box — must read `0`, and a high-water mark
    /// that was already set must not underflow into a nonsense figure.
    #[test]
    fn since_saturates_instead_of_wrapping() {
        let start = ResourceSample {
            cpu_ns: 100,
            peak_rss_bytes: 4096,
            minor_faults: 10,
            major_faults: 3,
            vol_ctx_switches: 7,
            invol_ctx_switches: 2,
            io_read_bytes: 8192,
            io_write_bytes: 512,
        };
        // An "end" sample strictly below the start (impossible in practice, but the arithmetic
        // must not wrap into 2^64 if a platform ever reports counters non-monotonically).
        let delta = ResourceSample::default().since(&start);
        assert_eq!(delta.cpu_ns, 0);
        assert_eq!(delta.peak_rss_bytes, 0);
        assert_eq!(delta.major_faults, 0);
        assert_eq!(delta.io_read_bytes, 0);

        let end = ResourceSample {
            cpu_ns: 350,
            peak_rss_bytes: 8192,
            minor_faults: 40,
            major_faults: 3,
            vol_ctx_switches: 19,
            invol_ctx_switches: 5,
            io_read_bytes: 20480,
            io_write_bytes: 512,
        };
        let delta = end.since(&start);
        assert_eq!(delta.cpu_ns, 250);
        assert_eq!(delta.minor_faults, 30);
        assert_eq!(
            delta.major_faults, 0,
            "a counter that stood still reads zero"
        );
        assert_eq!(delta.invol_ctx_switches, 3);
        assert_eq!(delta.io_read_bytes, 12288);
        assert_eq!(delta.io_write_bytes, 0);
    }
}
