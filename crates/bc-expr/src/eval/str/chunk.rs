//! `StrFunc::Chunk` — overlapping text windows (the RAG document splitter).
//!
//! Split out of `str.rs` for file size. Chunking is the one string function whose body
//! is more than a kernel call: it slices on **character** boundaries (a byte-wise slice
//! would cut a UTF-8 codepoint in half and panic) while avoiding the obvious
//! `Vec<char>`, which would cost 4 bytes per character of input.
//!
//! A chunk may also be required to end on a *semantic* boundary — a word, sentence or
//! line — rather than at an arbitrary character. That is not cosmetic for retrieval: a
//! chunk ending `…diagnosed with hyperten` embeds as something the query `hypertension
//! treatment` will not match, so a mid-word cut silently costs recall on exactly the
//! chunk that should have been the answer. The boundary modes back the cut off to the
//! last allowed separator inside the window, and fall back to a hard cut when there is
//! none — a single token longer than the whole chunk must still be emitted.

use std::sync::Arc;

use arrow::array::{ArrayRef, ListBuilder, StringArray, StringBuilder};

use crate::{ExprError, StrFunc};

/// Where a chunk is allowed to end.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Boundary {
    /// Anywhere — fixed-size windows, the historical behaviour.
    Char,
    /// After whitespace, so a word is never split.
    Word,
    /// After `.`, `!` or `?`.
    Sentence,
    /// After a newline.
    Line,
}

impl Boundary {
    fn parse(name: Option<&str>) -> Result<Self, ExprError> {
        match name.unwrap_or("char") {
            "char" => Ok(Self::Char),
            "word" => Ok(Self::Word),
            "sentence" => Ok(Self::Sentence),
            "line" => Ok(Self::Line),
            other => Err(ExprError::InvalidArgument {
                func: format!("{:?}", StrFunc::Chunk),
                reason: format!(
                    "unknown chunk boundary {other:?}; expected one of \
                     \"char\", \"word\", \"sentence\", \"line\""
                ),
            }),
        }
    }

    /// Whether a chunk may end immediately *after* `c`.
    fn ends_after(self, c: char) -> bool {
        match self {
            Self::Char => true,
            Self::Word => c.is_whitespace(),
            Self::Sentence => matches!(c, '.' | '!' | '?'),
            Self::Line => c == '\n',
        }
    }

    /// The separators to try, coarsest first, before giving up and cutting anywhere.
    ///
    /// Asking for sentence boundaries and getting a *mid-word* cut whenever a window
    /// holds no full stop would be worse than not asking at all — the caller wanted
    /// readable chunks, and one bad cut is exactly the chunk that then fails to match.
    /// So each mode degrades to the next-finer one (the recursive-splitter behaviour)
    /// rather than straight to an arbitrary character.
    fn cascade(self) -> &'static [Boundary] {
        match self {
            Self::Char => &[],
            Self::Word => &[Self::Word],
            Self::Sentence => &[Self::Sentence, Self::Word],
            Self::Line => &[Self::Line, Self::Word],
        }
    }
}

/// Evaluate `chunk`: `length` is the chunk size and `start` the overlap, both in
/// characters; `pattern` names the boundary mode. Null input → null list; empty string →
/// empty list.
pub(crate) fn eval_chunk(
    s: &StringArray,
    start: Option<i64>,
    length: Option<i64>,
    pattern: Option<&str>,
) -> Result<ArrayRef, ExprError> {
    let size = length.ok_or_else(|| ExprError::MissingArgument {
        func: format!("{:?}", StrFunc::Chunk),
        arg: "length",
    })?;
    let overlap = start.unwrap_or(0);
    if size < 1 || overlap < 0 || overlap >= size {
        return Err(ExprError::InvalidArgument {
            func: format!("{:?}", StrFunc::Chunk),
            reason: format!(
                "chunk size must be >= 1 and overlap in [0, size), got size={size} \
                 overlap={overlap}"
            ),
        });
    }
    let boundary = Boundary::parse(pattern)?;
    let (size, overlap) = (size as usize, overlap as usize);
    let mut builder = ListBuilder::new(StringBuilder::new());
    // Reused across rows: the byte offset of every character boundary.
    let mut offsets: Vec<usize> = Vec::new();
    for o in s.iter() {
        match o {
            Some(v) => {
                chunk_row(v, size, overlap, boundary, &mut offsets, builder.values());
                builder.append(true);
            }
            None => builder.append(false),
        }
    }
    Ok(Arc::new(builder.finish()))
}

/// Append one row's chunks to `out`, each a borrowed `&str` slice of `text` (so no
/// per-chunk `String` is built).
///
/// ASCII — the bulk of the text a RAG pipeline ingests — needs no boundary table at
/// all, since a character is a byte; otherwise `offsets` is reused across rows to hold
/// the byte offset of every character boundary.
///
/// Chunks stop as soon as one reaches the end, so a trailing chunk wholly contained in
/// its predecessor is never emitted. With `overlap == 0`, any boundary mode reproduces
/// the input exactly when the chunks are concatenated: a separator *ends* the chunk it
/// belongs to rather than being skipped, so no character is dropped.
fn chunk_row(
    text: &str,
    size: usize,
    overlap: usize,
    boundary: Boundary,
    offsets: &mut Vec<usize>,
    out: &mut StringBuilder,
) {
    let mut emit = |chars: usize, byte_at: &dyn Fn(usize) -> usize| {
        // The character at char-index `k`, read through the same offset mapping.
        let char_at = |k: usize| text[byte_at(k)..byte_at(k + 1)].chars().next();
        let mut i = 0;
        while i < chars {
            let hard_end = (i + size).min(chars);
            // Only back off when the window does not already reach the end of the text —
            // the final chunk needs no boundary.
            let end = if hard_end == chars {
                hard_end
            } else {
                // Try each separator in turn, coarsest first; `hard_end` if none of them
                // appears in the window — so an oversized token still emits rather than
                // the loop stalling.
                boundary
                    .cascade()
                    .iter()
                    .find_map(|level| {
                        (i + 1..hard_end)
                            .rev()
                            .find(|&k| char_at(k).is_some_and(|c| level.ends_after(c)))
                            .map(|k| k + 1)
                    })
                    .unwrap_or(hard_end)
            };
            out.append_value(&text[byte_at(i)..byte_at(end)]);
            if end == chars {
                break;
            }
            // Advance by this chunk's own length less the overlap, never by zero — a
            // boundary-shortened chunk must still make progress or this loops forever.
            i += (end - i).saturating_sub(overlap).max(1);
        }
    };

    if text.is_ascii() {
        emit(text.len(), &|i| i);
        return;
    }
    offsets.clear();
    offsets.extend(text.char_indices().map(|(i, _)| i));
    offsets.push(text.len());
    let table = &offsets[..];
    emit(table.len() - 1, &|i| table[i]);
}
