//! Copying a sequence of source byte spans into one output buffer, as few copies as possible.
//!
//! Every variable-width gather in this module ends in the same loop: for each output row, copy
//! that row's bytes out of a source value buffer and append them. Written as one
//! `copy_from_slice` per row, a column of short strings is a libc `memmove` *call* per value —
//! 17% of H2O `join` q2 on its own, for values five to twelve bytes long — and the call costs
//! more than the bytes it moves.
//!
//! Two properties of the input remove most of those calls without changing a byte of output:
//!
//! - **Adjacent spans merge.** A hash join's probe-side index is ascending, and wherever
//!   consecutive probe rows both match, their source bytes are adjacent too. Such a run is one
//!   copy however many rows it covers. The check is one comparison per row, so a scattered
//!   gather pays nothing measurable for it.
//! - **A short run is a fixed-width copy.** A run of at most [`SHORT`] bytes is copied as a
//!   whole [`SHORT`]-byte block when both buffers have that many bytes left, which compiles to
//!   two vector moves instead of a call. The bytes past the run's end land in the output
//!   *ahead* of the write position, where the next copy overwrites them, and the block is
//!   refused within [`SHORT`] bytes of the destination's end, so it can never write past the
//!   slice it was handed. That matters in the parallel paths, where the destination is one
//!   chunk's carve of a shared buffer and the bytes beyond it belong to the next chunk.
//!
//! The output is identical to the per-row loop: the same bytes, in the same order, at the same
//! offsets. Only the number of copies changes.

/// The fixed block a short run is copied as. Sixteen bytes is one SSE move, and it covers the
/// id-, code- and flag-width strings that make up most group keys and dimension attributes.
const SHORT: usize = 16;

/// Accumulates spans of one source buffer and copies each maximal adjacent run at once.
///
/// Drive it with [`SpanCopier::push`] per row in output order, then [`SpanCopier::finish`];
/// the destination is written from index 0.
pub(super) struct SpanCopier<'a> {
    src: &'a [u8],
    run_start: usize,
    run_len: usize,
    at: usize,
}

impl<'a> SpanCopier<'a> {
    pub(super) fn new(src: &'a [u8]) -> Self {
        Self {
            src,
            run_start: 0,
            run_len: 0,
            at: 0,
        }
    }

    /// Append `src[start..start + len]` to the output.
    #[inline]
    pub(super) fn push(&mut self, dst: &mut [u8], start: usize, len: usize) {
        if start == self.run_start + self.run_len {
            self.run_len += len;
            return;
        }
        self.flush(dst);
        self.run_start = start;
        self.run_len = len;
    }

    /// Switch to another source buffer. The pending run belongs to the old one, so it is
    /// copied first — two sources' spans can be numerically adjacent without being adjacent
    /// in memory.
    #[inline]
    pub(super) fn set_source(&mut self, dst: &mut [u8], src: &'a [u8]) {
        if !std::ptr::eq(self.src, src) {
            self.flush(dst);
            self.src = src;
        }
    }

    /// Copy the pending run and return the number of bytes written in total.
    pub(super) fn finish(mut self, dst: &mut [u8]) -> usize {
        self.flush(dst);
        self.at
    }

    #[inline]
    fn flush(&mut self, dst: &mut [u8]) {
        let (s, n, at) = (self.run_start, self.run_len, self.at);
        if n == 0 {
            return;
        }
        if n <= SHORT && s + SHORT <= self.src.len() && at + SHORT <= dst.len() {
            dst[at..at + SHORT].copy_from_slice(&self.src[s..s + SHORT]);
        } else {
            dst[at..at + n].copy_from_slice(&self.src[s..s + n]);
        }
        self.at = at + n;
        self.run_len = 0;
    }
}

/// [`SpanCopier`] for a destination that grows: the serial gather, which does not know its
/// total byte count until it has read every span. Runs merge the same way; a flush appends.
pub(super) struct SpanAppender<'a> {
    src: &'a [u8],
    run_start: usize,
    run_len: usize,
}

impl<'a> SpanAppender<'a> {
    pub(super) fn new(src: &'a [u8]) -> Self {
        Self {
            src,
            run_start: 0,
            run_len: 0,
        }
    }

    #[inline]
    pub(super) fn push(&mut self, out: &mut Vec<u8>, start: usize, len: usize) {
        if start == self.run_start + self.run_len {
            self.run_len += len;
            return;
        }
        self.flush(out);
        self.run_start = start;
        self.run_len = len;
    }

    pub(super) fn finish(mut self, out: &mut Vec<u8>) {
        self.flush(out);
    }

    #[inline]
    fn flush(&mut self, out: &mut Vec<u8>) {
        if self.run_len > 0 {
            out.extend_from_slice(&self.src[self.run_start..self.run_start + self.run_len]);
            self.run_len = 0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The per-row loop the copiers replace — the oracle.
    fn per_row(src: &[u8], spans: &[(usize, usize)]) -> Vec<u8> {
        let mut out = Vec::new();
        for &(s, n) in spans {
            out.extend_from_slice(&src[s..s + n]);
        }
        out
    }

    fn spans_cases() -> Vec<Vec<(usize, usize)>> {
        vec![
            vec![],
            vec![(0, 0)],
            vec![(0, 5), (5, 3), (8, 0), (8, 7)], // one adjacent run
            vec![(10, 4), (0, 3), (30, 20), (3, 2)], // scattered, long and short
            vec![(5, 0), (5, 0), (0, 16), (16, 16)], // empties, then exactly-SHORT runs
            vec![(40, 9), (49, 1), (2, 1), (60, 4)], // a run ending near the source's end
        ]
    }

    #[test]
    fn the_copier_writes_what_the_per_row_loop_writes() {
        let src: Vec<u8> = (0..64u8).collect();
        for spans in spans_cases() {
            let want = per_row(&src, &spans);
            // The destination is exactly sized, as a parallel chunk's carve is, so a block copy
            // that ran past its end would panic here rather than pass.
            let mut dst = vec![0xAA; want.len()];
            let mut c = SpanCopier::new(&src);
            for &(s, n) in &spans {
                c.push(&mut dst, s, n);
            }
            assert_eq!(c.finish(&mut dst), want.len());
            assert_eq!(dst, want, "spans {spans:?}");
        }
    }

    #[test]
    fn the_appender_writes_what_the_per_row_loop_writes() {
        let src: Vec<u8> = (0..64u8).collect();
        for spans in spans_cases() {
            let mut out = Vec::new();
            let mut a = SpanAppender::new(&src);
            for &(s, n) in &spans {
                a.push(&mut out, s, n);
            }
            a.finish(&mut out);
            assert_eq!(out, per_row(&src, &spans), "spans {spans:?}");
        }
    }

    #[test]
    fn spans_of_different_sources_never_merge() {
        // Row 1's span starts where row 0's ends numerically, but in another buffer.
        let (a, b): (Vec<u8>, Vec<u8>) = ((0..8).collect(), (100..108).collect());
        let mut dst = vec![0; 8];
        let mut c = SpanCopier::new(&a);
        c.push(&mut dst, 0, 4);
        c.set_source(&mut dst, &b);
        c.push(&mut dst, 4, 4);
        assert_eq!(c.finish(&mut dst), 8);
        assert_eq!(dst, vec![0, 1, 2, 3, 104, 105, 106, 107]);
    }
}
