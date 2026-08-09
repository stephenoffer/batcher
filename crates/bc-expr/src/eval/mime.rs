//! What a payload is, decided from its bytes — the one magic-number table.
//!
//! It sits beside the function families rather than inside `str/` because it is not a
//! string function: it reads raw bytes, and the `.str` namespace merely exposes it —
//! that namespace is where every byte-oriented op already lives (`octet_length`, `hex`,
//! the digests, `compress`).
//!
//! Two callers share this: the `.str.mime_type()` expression, and the IO layer's
//! `sniff_mime` (through a `bc-py` helper), which adds a filename-extension fallback the
//! expression has no path to attempt. Keeping the table here rather than in both is the
//! point — a format added to one copy and not the other is a divergence nothing would
//! report, on a column whose whole job is to route rows to different branches.
//!
//! Deliberately not "every format": these are the ones an unstructured pipeline routes on.
//! Three families need more than a prefix match and get their own readers, because the
//! signature is not at offset zero or is not unique:
//!
//! * **ISO base media** (MP4, MOV, HEIC, AVIF, 3GP) puts a four-byte box length *before*
//!   the `ftyp` marker, so its signature starts at offset 4 and the format itself is the
//!   brand at offset 8. A prefix table structurally cannot see it, which is how the most
//!   common video container in the world — and every photo a recent phone takes — read as
//!   `application/octet-stream` whenever the object key carried no extension.
//! * **Zip** is the container for the whole Office and EPUB family, one magic number for
//!   formats that differ only in what is stored inside.
//! * **Matroska and WebM** share their magic and differ only in an EBML DocType string.

/// How many leading bytes are enough to identify a payload.
///
/// Every reader below looks inside a fixed prefix, so a caller that only has a header —
/// the media sources read 64 KiB and never the whole file — gets the same answer as one
/// holding the entire payload.
pub(crate) const MAGIC_PEEK: usize = 4096;

/// Prefix-decided formats. Everything here is settled by the leading bytes alone.
const PREFIXES: &[(&[u8], &str)] = &[
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
    (b"BM", "image/bmp"),
    (b"II*\x00", "image/tiff"),
    (b"MM\x00*", "image/tiff"),
    (b"%PDF-", "application/pdf"),
    (b"OggS", "audio/ogg"),
    (b"fLaC", "audio/flac"),
    (b"ID3", "audio/mpeg"),
    (b"\xff\xfb", "audio/mpeg"),
    (b"\xff\xf3", "audio/mpeg"),
    (b"\xff\xf2", "audio/mpeg"),
    (b"\x00\x00\x01\xba", "video/mpeg"),
    (b"\x1f\x8b", "application/gzip"),
    (b"BZh", "application/x-bzip2"),
    (b"\xfd7zXZ\x00", "application/x-xz"),
    (b"\x28\xb5\x2f\xfd", "application/zstd"),
    (b"Obj\x01", "application/avro"),
    (b"PAR1", "application/vnd.apache.parquet"),
    (b"ARROW1", "application/vnd.apache.arrow.file"),
    (b"SQLite format 3\x00", "application/vnd.sqlite3"),
];

/// ISO base media brands. Matched as a prefix because the field is space-padded to four
/// bytes and versioned (`3gp4`, `3gp5`, …). The brand separates a still from a video, and
/// reading it wrong routes a photograph into a video decoder.
const ISO_BRANDS: &[(&[&str], &str)] = &[
    (
        &[
            "heic", "heix", "hevc", "hevx", "heim", "heis", "hevm", "hevs",
        ],
        "image/heic",
    ),
    (&["mif1", "msf1"], "image/heif"),
    (&["avif", "avis"], "image/avif"),
    (&["qt  "], "video/quicktime"),
    (&["3gp", "3g2"], "video/3gpp"),
    (&["M4A", "M4B"], "audio/mp4"),
    (&["crx"], "image/x-canon-cr3"),
];

/// Zip-container formats, keyed by a marker in the archive's leading bytes. EPUB is exact
/// by construction (the spec requires an uncompressed `mimetype` entry first); the Office
/// formats are matched by their first stored part's name, which every producer writes early
/// enough to fall inside the peek.
const ZIP_MARKERS: &[(&[u8], &str)] = &[
    (b"mimetypeapplication/epub+zip", "application/epub+zip"),
    (
        b"word/",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ),
    (
        b"ppt/",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ),
    (
        b"xl/",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ),
];

/// The MIME type `data`'s leading bytes identify, or `None` when nothing recognizes them.
///
/// `None` rather than `application/octet-stream`: this reader knows only what the bytes
/// say, and "I do not recognize this" is a different claim from "this is opaque binary".
/// The IO layer turns the first into the second only after its extension fallback has also
/// declined; an expression, which has no filename to fall back to, leaves it null.
pub(crate) fn sniff(data: &[u8]) -> Option<&'static str> {
    let head = &data[..data.len().min(MAGIC_PEEK)];
    riff(head)
        .or_else(|| iso_base_media(head))
        .or_else(|| matroska(head))
        .or_else(|| zip(head))
        .or_else(|| prefix(head))
}

fn prefix(head: &[u8]) -> Option<&'static str> {
    PREFIXES
        .iter()
        .find(|(magic, _)| head.starts_with(magic))
        .map(|(_, mime)| *mime)
}

/// RIFF is a container tag, not a format — the format is four bytes at offset 8.
fn riff(head: &[u8]) -> Option<&'static str> {
    if !head.starts_with(b"RIFF") {
        return None;
    }
    match head.get(8..12)? {
        b"WEBP" => Some("image/webp"),
        b"WAVE" => Some("audio/x-wav"),
        b"AVI " => Some("video/x-msvideo"),
        _ => None,
    }
}

/// MP4 and its relatives: `ftyp` at offset 4, the brand at offset 8.
fn iso_base_media(head: &[u8]) -> Option<&'static str> {
    if head.get(4..8)? != b"ftyp" {
        return None;
    }
    let brand = std::str::from_utf8(head.get(8..12)?).ok()?;
    for (brands, mime) in ISO_BRANDS {
        if brands.iter().any(|b| brand.starts_with(b)) {
            return Some(mime);
        }
    }
    // `isom`, `mp41`, `mp42`, `dash`, `avc1` and the rest of the long tail are all MP4.
    Some("video/mp4")
}

/// Matroska and WebM share the EBML magic and differ only in their DocType. Reporting
/// every WebM as `video/x-matroska` is not wrong so much as useless: routing on the mime
/// type is exactly where the distinction is wanted.
fn matroska(head: &[u8]) -> Option<&'static str> {
    if !head.starts_with(b"\x1aE\xdf\xa3") {
        return None;
    }
    let window = &head[..head.len().min(64)];
    Some(if contains(window, b"webm") {
        "video/webm"
    } else {
        "video/x-matroska"
    })
}

/// Zip carries the whole Office and EPUB family under one magic number.
fn zip(head: &[u8]) -> Option<&'static str> {
    if !(head.starts_with(b"PK\x03\x04")
        || head.starts_with(b"PK\x05\x06")
        || head.starts_with(b"PK\x07\x08"))
    {
        return None;
    }
    for (marker, mime) in ZIP_MARKERS {
        if contains(head, marker) {
            return Some(mime);
        }
    }
    Some("application/zip")
}

/// Whether `needle` appears anywhere in `haystack`.
///
/// A hand-rolled scan rather than a dependency: the needles are a handful of bytes and the
/// haystack is at most the peek, so this is not the interesting cost next to the file read
/// that produced the bytes.
fn contains(haystack: &[u8], needle: &[u8]) -> bool {
    haystack
        .windows(needle.len())
        .any(|window| window == needle)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An ISO base media header: a box length, `ftyp`, then the brand.
    fn iso(brand: &[u8]) -> Vec<u8> {
        let mut out = b"\x00\x00\x00\x20ftyp".to_vec();
        out.extend_from_slice(brand);
        out.extend_from_slice(b"\x00\x00\x02\x00");
        out
    }

    /// The families a prefix table cannot reach. Kept as its own test because these are
    /// exactly the ones that were missing, and the reason they were missing is structural
    /// rather than an oversight: their signature does not start at byte zero.
    #[test]
    fn the_signature_is_not_always_at_offset_zero() {
        assert_eq!(sniff(&iso(b"isom")), Some("video/mp4"));
        assert_eq!(sniff(&iso(b"qt  ")), Some("video/quicktime"));
        assert_eq!(sniff(&iso(b"heic")), Some("image/heic"));
        assert_eq!(sniff(&iso(b"avif")), Some("image/avif"));
        assert_eq!(sniff(&iso(b"3gp4")), Some("video/3gpp"));
        assert_eq!(sniff(&iso(b"M4A ")), Some("audio/mp4"));
    }

    #[test]
    fn one_magic_number_can_be_several_formats() {
        let mut epub = b"PK\x03\x04".to_vec();
        epub.extend_from_slice(&[0u8; 26]);
        epub.extend_from_slice(b"mimetypeapplication/epub+zip");
        assert_eq!(sniff(&epub), Some("application/epub+zip"));
        assert_eq!(
            sniff(b"PK\x03\x04\x14\x00\x00\x00word/document.xml"),
            Some("application/vnd.openxmlformats-officedocument.wordprocessingml.document")
        );
        assert_eq!(
            sniff(b"PK\x03\x04\x14\x00\x00\x00stuff/thing.txt"),
            Some("application/zip")
        );
    }

    #[test]
    fn matroska_and_webm_are_told_apart_by_their_doctype() {
        let base = b"\x1aE\xdf\xa3\x01\x00\x00\x00\x00\x00\x00\x23B\x82\x84";
        let mut webm = base.to_vec();
        webm.extend_from_slice(b"webm");
        let mut mkv = base.to_vec();
        mkv.extend_from_slice(b"matroska");
        assert_eq!(sniff(&webm), Some("video/webm"));
        assert_eq!(sniff(&mkv), Some("video/x-matroska"));
    }

    /// RIFF names a container, so an unknown chunk type is unrecognized rather than
    /// reported as whichever RIFF format happens to be listed first.
    #[test]
    fn an_unknown_riff_chunk_is_not_guessed() {
        assert_eq!(sniff(b"RIFF\x00\x00\x00\x00WEBPVP8 "), Some("image/webp"));
        assert_eq!(sniff(b"RIFF\x00\x00\x00\x00WAVEfmt "), Some("audio/x-wav"));
        assert_eq!(sniff(b"RIFF\x00\x00\x00\x00XXXXdata"), None);
    }

    /// A zero-length object is a real thing in a blob store, and every reader here slices
    /// — so the short cases have to be answered rather than panicked on.
    ///
    /// A truncated *container* header is a different matter from an unrecognized payload:
    /// `PK` alone is not yet a zip (the magic is four bytes) and a bare `ftyp` box carries
    /// no brand to read, so both are unrecognized. The EBML magic is complete at four
    /// bytes, so it identifies a Matroska file even with nothing after it — see
    /// `a_truncated_ebml_header_is_still_matroska`.
    #[test]
    fn a_short_or_empty_payload_is_unrecognized_not_a_panic() {
        for head in [
            b"".as_slice(),
            b"\xff",
            b"RIFF",
            b"PK",
            b"\x00\x00\x00\x20ftyp",
        ] {
            assert_eq!(sniff(head), None, "{head:?}");
        }
    }

    /// The DocType distinguishes WebM from Matroska; its absence does not un-identify the
    /// file. Defaulting to Matroska is the honest reading of a complete EBML magic, and it
    /// is what the header-only callers need — a media source peeks 64 KiB, but a truncated
    /// upload may hold less than the DocType.
    #[test]
    fn a_truncated_ebml_header_is_still_matroska() {
        assert_eq!(sniff(b"\x1aE\xdf\xa3"), Some("video/x-matroska"));
    }

    /// Only the peek is read, so a header-only caller and a whole-payload caller agree.
    #[test]
    fn only_the_leading_bytes_decide() {
        let mut png = b"\x89PNG\r\n\x1a\n".to_vec();
        png.extend(std::iter::repeat_n(0u8, MAGIC_PEEK * 4));
        assert_eq!(sniff(&png), Some("image/png"));
        assert_eq!(sniff(&png[..16]), Some("image/png"));
    }

    #[test]
    fn nothing_recognized_is_none_rather_than_octet_stream() {
        assert_eq!(sniff(b"\x00\x01\x02\x03 not a known format"), None);
    }
}
