//! `strip_html`: recover the readable text of an HTML document.
//!
//! The first step of almost every unstructured-text pipeline — scraped pages, product
//! descriptions, email bodies — is throwing away the markup. Doing it with
//! `regexp_replace('<[^>]*>', '')` is the idiom people reach for and it is wrong in
//! ways that quietly poison a training corpus: it leaves the *contents* of `<script>`
//! and `<style>` in the text, it leaves `&amp;` and `&nbsp;` undecoded, and it welds
//! `<p>a</p><p>b</p>` into `ab`.
//!
//! This is a text extractor, not an HTML parser: it does not build a DOM, validate
//! nesting, or resolve `<br>` vs `<br/>`. It makes one pass over the bytes and is
//! deliberately lenient — unclosed tags, stray `<`, and malformed entities pass through
//! rather than erroring, because a scrape of the open web contains all of them and one
//! bad row must not abort a scan.

/// Elements whose *text content* is code, not prose, and is dropped with the tag.
const SKIPPED_ELEMENTS: [&str; 2] = ["script", "style"];

/// The named entities that actually occur in scraped prose. The numeric forms
/// (`&#38;`, `&#x26;`) are decoded generally, so this list only needs the names.
const NAMED_ENTITIES: [(&str, char); 6] = [
    ("amp", '&'),
    ("lt", '<'),
    ("gt", '>'),
    ("quot", '"'),
    ("apos", '\''),
    ("nbsp", ' '),
];

/// The longest entity name we will scan for before giving up and treating the `&` as
/// literal text. `&#x10FFFF;` is the worst case.
const MAX_ENTITY_LEN: usize = 10;

/// Strip tags, drop script/style content, decode entities, and collapse whitespace.
///
/// Element boundaries become a single space, so block structure survives as word
/// separation (`<p>a</p><p>b</p>` → `"a b"`). Leading/trailing whitespace is trimmed.
pub(super) fn strip_html_text(html: &str) -> String {
    // A page is mostly text, so the output is nearly as long as the input.
    let mut out = String::with_capacity(html.len());
    let bytes = html.as_bytes();
    let mut i = 0;

    while i < bytes.len() {
        match bytes[i] {
            b'<' => {
                if let Some(rest) = html[i..].strip_prefix("<!--") {
                    // A comment ends at `-->`; an unterminated one eats the rest.
                    i += match rest.find("-->") {
                        Some(end) => "<!--".len() + end + "-->".len(),
                        None => html.len() - i,
                    };
                    continue;
                }
                let Some(tag_end) = html[i..].find('>') else {
                    // A stray `<` with no `>`: literal text, not a truncated tag.
                    push_text(&mut out, &html[i..]);
                    break;
                };
                let tag = &html[i + 1..i + tag_end];
                i += tag_end + 1;
                push_space(&mut out);
                if let Some(element) = skipped_element(tag) {
                    i += skip_element_content(&html[i..], element);
                }
            }
            b'&' => match decode_entity(&html[i..]) {
                Some((decoded, consumed)) => {
                    push_char(&mut out, decoded);
                    i += consumed;
                }
                None => {
                    push_char(&mut out, '&');
                    i += 1;
                }
            },
            _ => {
                // Copy the run up to the next `<` or `&` in one go, decoding whitespace.
                let run_end = html[i..]
                    .find(['<', '&'])
                    .map_or(html.len(), |offset| i + offset);
                push_text(&mut out, &html[i..run_end]);
                i = run_end;
            }
        }
    }

    while out.ends_with(' ') {
        out.pop();
    }
    out
}

/// The element name if this open tag begins a `<script>`/`<style>` block.
fn skipped_element(tag: &str) -> Option<&'static str> {
    if tag.starts_with('/') {
        return None;
    }
    let name_end = tag.find(|c: char| c.is_whitespace()).unwrap_or(tag.len());
    let name = &tag[..name_end];
    SKIPPED_ELEMENTS
        .iter()
        .find(|element| name.eq_ignore_ascii_case(element))
        .copied()
}

/// Bytes to skip from just after `<script>` through its `</script>`, inclusive.
///
/// An unterminated block consumes the remainder — the same choice a browser makes, and
/// the safe one: the alternative is emitting JavaScript as prose.
fn skip_element_content(rest: &str, element: &str) -> usize {
    let mut search = 0;
    while let Some(offset) = rest[search..].find("</") {
        let close = search + offset + "</".len();
        // Compare as bytes: `close + element.len()` need not be a char boundary.
        let name = rest.as_bytes().get(close..close + element.len());
        if name.is_some_and(|n| n.eq_ignore_ascii_case(element.as_bytes())) {
            return match rest[close..].find('>') {
                Some(end) => close + end + 1,
                None => rest.len(),
            };
        }
        search = close;
    }
    rest.len()
}

/// Decode the entity starting at `&`, returning the character and the bytes consumed.
fn decode_entity(rest: &str) -> Option<(char, usize)> {
    // Scan for `;` over *bytes*: `MAX_ENTITY_LEN` need not fall on a char boundary,
    // and `;` is ASCII, so it can never occur inside a multi-byte character.
    let limit = MAX_ENTITY_LEN + 2;
    let semicolon = rest.bytes().take(limit).position(|b| b == b';')?;
    let body = &rest[1..semicolon];
    let consumed = semicolon + 1;

    if let Some(digits) = body.strip_prefix('#') {
        let code = match digits.strip_prefix(['x', 'X']) {
            Some(hex) => u32::from_str_radix(hex, 16).ok()?,
            None => digits.parse::<u32>().ok()?,
        };
        return char::from_u32(code).map(|c| (c, consumed));
    }
    NAMED_ENTITIES
        .iter()
        .find(|(name, _)| body.eq_ignore_ascii_case(name))
        .map(|(_, c)| (*c, consumed))
}

/// Append text, collapsing every run of whitespace to a single space.
fn push_text(out: &mut String, text: &str) {
    for c in text.chars() {
        push_char(out, c);
    }
}

fn push_char(out: &mut String, c: char) {
    if c.is_whitespace() {
        push_space(out);
    } else {
        out.push(c);
    }
}

/// A space, unless one is already there or the output is still empty — this is what
/// keeps `<p>a</p>\n<p>b</p>` at exactly one space and strips the leading indent.
fn push_space(out: &mut String) {
    if !out.is_empty() && !out.ends_with(' ') {
        out.push(' ');
    }
}

#[cfg(test)]
mod tests {
    use super::strip_html_text;

    #[test]
    fn tags_are_removed_and_text_kept() {
        assert_eq!(strip_html_text("<b>hello</b> world"), "hello world");
    }

    #[test]
    fn element_boundaries_separate_words() {
        assert_eq!(strip_html_text("<p>a</p><p>b</p>"), "a b");
    }

    #[test]
    fn script_and_style_content_is_dropped() {
        assert_eq!(strip_html_text("a<script>var x = 1 < 2;</script>b"), "a b");
        assert_eq!(strip_html_text("a<STYLE>p {color: red}</STYLE>b"), "a b");
    }

    #[test]
    fn a_script_tag_with_attributes_is_still_skipped() {
        assert_eq!(
            strip_html_text("<script type='text/js'>junk()</script>ok"),
            "ok"
        );
    }

    #[test]
    fn an_unterminated_script_eats_the_rest() {
        assert_eq!(strip_html_text("a<script>junk()"), "a");
    }

    #[test]
    fn comments_are_dropped() {
        assert_eq!(strip_html_text("a<!-- note <b>x</b> -->b"), "ab");
        assert_eq!(strip_html_text("a<!-- unterminated"), "a");
    }

    #[test]
    fn named_entities_decode() {
        assert_eq!(
            strip_html_text("a &amp; b &lt;c&gt; &quot;d&quot;"),
            "a & b <c> \"d\""
        );
    }

    #[test]
    fn nbsp_becomes_a_single_space() {
        assert_eq!(strip_html_text("a&nbsp;&nbsp;b"), "a b");
    }

    #[test]
    fn numeric_entities_decode_in_both_bases() {
        assert_eq!(strip_html_text("&#65;&#x42;&#X43;"), "ABC");
    }

    #[test]
    fn a_malformed_entity_is_literal_text() {
        assert_eq!(strip_html_text("a & b"), "a & b");
        assert_eq!(strip_html_text("&notanentity;"), "&notanentity;");
        assert_eq!(strip_html_text("&#zz;"), "&#zz;");
    }

    #[test]
    fn whitespace_runs_collapse_and_edges_trim() {
        assert_eq!(strip_html_text("  a \n\t b  "), "a b");
    }

    #[test]
    fn a_stray_open_bracket_with_no_close_is_literal_text() {
        // Scraped prose contains bare `<`. A `<` that never closes is text, not a
        // truncated tag — dropping the rest of the row would be the worse failure.
        assert_eq!(strip_html_text("2 < 3"), "2 < 3");
        assert_eq!(strip_html_text("a<b"), "a<b");
    }

    #[test]
    fn empty_and_markup_only_inputs_give_the_empty_string() {
        assert_eq!(strip_html_text(""), "");
        assert_eq!(strip_html_text("<div></div>"), "");
    }

    #[test]
    fn unicode_survives_and_is_not_split() {
        assert_eq!(strip_html_text("<p>héllo 🌍</p>"), "héllo 🌍");
    }

    #[test]
    fn the_regex_idiom_this_replaces_would_get_these_wrong() {
        // `regexp_replace('<[^>]*>', '')` leaves script bodies, undecoded entities,
        // and welds block elements together. Pin the contrast.
        let html = "<p>Tom &amp; Jerry</p><p>x</p><script>if (a<b) f();</script>";
        assert_eq!(strip_html_text(html), "Tom & Jerry x");
    }
}
