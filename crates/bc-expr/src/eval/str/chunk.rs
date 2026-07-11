//! `StrFunc::Chunk` — fixed-size overlapping text windows (the RAG document splitter).
//!
//! Split out of `str.rs` for file size. Chunking is the one string function whose body
//! is more than a kernel call: it slices on **character** boundaries (a byte-wise slice
//! would cut a UTF-8 codepoint in half and panic) while avoiding the obvious
//! `Vec<char>`, which would cost 4 bytes per character of input.

use std::sync::Arc;

use arrow::array::{ArrayRef, ListBuilder, StringArray, StringBuilder};

use crate::{ExprError, StrFunc};

/// Evaluate `chunk`: `length` is the chunk size and `start` the overlap, both in
/// characters. Null input → null list; empty string → empty list.
pub(crate) fn eval_chunk(
    s: &StringArray,
    start: Option<i64>,
    length: Option<i64>,
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
    // `stride >= 1` follows from `overlap < size`, so the loop always advances.
    let (size, stride) = (size as usize, (size - overlap) as usize);
    let mut builder = ListBuilder::new(StringBuilder::new());
    // Reused across rows: the byte offset of every character boundary.
    let mut offsets: Vec<usize> = Vec::new();
    for o in s.iter() {
        match o {
            Some(v) => {
                chunk_row(v, size, stride, &mut offsets, builder.values());
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
/// Chunks start every `stride` characters and stop as soon as one reaches the end, so a
/// trailing chunk wholly contained in its predecessor is never emitted.
fn chunk_row(
    text: &str,
    size: usize,
    stride: usize,
    offsets: &mut Vec<usize>,
    out: &mut StringBuilder,
) {
    let mut emit = |chars: usize, byte_at: &dyn Fn(usize) -> usize| {
        let mut i = 0;
        while i < chars {
            let end = (i + size).min(chars);
            out.append_value(&text[byte_at(i)..byte_at(end)]);
            if end == chars {
                break;
            }
            i += stride;
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
