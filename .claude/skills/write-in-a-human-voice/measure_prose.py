#!/usr/bin/env python3
"""Measure the shape of Markdown prose: the tells a reader catches while skimming.

This is the deterministic floor under the `write-in-a-human-voice` skill, and it is a
floor rather than a verdict. It counts what a regex can see -- heading density, bullet
ratio, sentence-length distribution, em-dashes, a short filler list, and the two
syntactic constructions a current model produces most reliably. It cannot see vacuity,
a claim with no source, a weak recommendation, or a fabricated number, which are the
tells that matter most. A page can score clean here and still read as generated.

Code is excluded before anything is counted: fenced blocks, MyST directive fences, YAML
front matter, table rows, and inline code spans. A docs page is mostly code by volume
and counting it would make every number meaningless.

Usage:
    python3 .claude/skills/write-in-a-human-voice/measure_prose.py docs/user-guide/joins.md
    python3 .claude/skills/write-in-a-human-voice/measure_prose.py docs/user-guide/**/*.md
    python3 .claude/skills/write-in-a-human-voice/measure_prose.py --summary docs
"""

from __future__ import annotations

import pathlib
import re
import statistics
import sys

#: Words that are nearly always filler in this repo's prose. A floor, not a specification:
#: see the substitution table in `.claude/skills/docs-grammar-style/SKILL.md`.
FILLER = (
    "simply",
    "easily",
    "of course",
    "note that",
    "in order to",
    "utilize",
    "leverage",
    "facilitate",
    "robust",
    "powerful",
    "seamless",
    "comprehensive",
    "cutting-edge",
    "best-in-class",
    "delve",
    "a variety of",
    "in the realm of",
    "it is worth noting",
)

#: "What made this hard was ...", "The reason X is that ...", "It is X that ...".
CLEFT = re.compile(
    r"\b(?:what|the reason(?:s)?)\b[^.!?\n]{0,60}?\b(?:is|was|are|were)\b"
    r"|\bit(?:'s| is| was)\b[^.!?\n]{0,40}?\bthat\b",
    re.IGNORECASE,
)

#: ", making it easier to", ", allowing teams to", ", ensuring that".
PARTICIPIAL_TAIL = re.compile(
    r",\s+(?:making|allowing|ensuring|enabling|giving|providing|helping)\b"
)

#: An em-dash that is a table cell's whole content, or sits alone inside a tag, is a
#: *glyph* meaning "not applicable", not prose punctuation. `docs/index.md`'s support
#: matrix uses 19 of them and its legend defines them. Counting those as tells pushes an
#: editor into breaking a table to satisfy a rule about sentences.
_DASH_GLYPH = re.compile(r"(?:<td[^>]*>|\|\s*|>)\s*—\s*(?:\||</td>|<)")

_FENCE = re.compile(r"^\s*(`{3,}|~{3,})")
_HEADING = re.compile(r"^#{1,6}\s")
_BULLET = re.compile(r"^\s*(?:[-*+]\s|\d+\.\s)")
_BOLD = re.compile(r"\*\*[^*\n]+\*\*")
_INLINE_CODE = re.compile(r"`[^`\n]*`")
_SENTENCE_END = re.compile(r"(?<=[.!?])[\s\n]+")


def strip_code(text: str) -> tuple[list[str], list[str]]:
    """Split a Markdown document into prose lines and the lines that are not prose.

    Args:
        text: The raw file contents.

    Returns:
        A pair of (prose lines, all content lines outside front matter). The second
        element is what heading and bullet density are measured against, because a
        table row is part of the page's shape even though it is not a sentence.
    """
    lines = text.splitlines()
    if lines and lines[0].strip() == "---":
        end = next((i for i, ln in enumerate(lines[1:], 1) if ln.strip() == "---"), 0)
        lines = lines[end + 1 :]

    prose: list[str] = []
    content: list[str] = []
    fence: str | None = None
    for line in lines:
        match = _FENCE.match(line)
        if fence is not None:
            if match and line.strip().startswith(fence):
                fence = None
            continue
        if match:
            fence = match.group(1)
            continue
        content.append(line)
        stripped = line.strip()
        if (
            not stripped
            or _HEADING.match(line)
            or stripped.startswith("|")
            or stripped.startswith(":")
        ):
            # A blank line, a heading and a table row all end a paragraph; keeping the
            # break is what makes the paragraph-length spread measurable.
            prose.append("")
            continue
        prose.append(_INLINE_CODE.sub("CODE", line))
    return prose, content


def sentences(prose: list[str]) -> list[int]:
    """Return the word count of every prose sentence, longest-first ordering not applied."""
    blob = " ".join(ln.strip() for ln in prose if ln.strip())
    return [len(s.split()) for s in _SENTENCE_END.split(blob) if len(s.split()) > 1]


def paragraphs(prose: list[str]) -> list[int]:
    """Return the word count of every prose paragraph."""
    out: list[int] = []
    current: list[str] = []
    for line in prose:
        if line.strip():
            current.append(line)
        elif current:
            out.append(len(" ".join(current).split()))
            current = []
    if current:
        out.append(len(" ".join(current).split()))
    return out


def cov(values: list[int]) -> float:
    """Coefficient of variation: the spread of a distribution, scaled by its mean."""
    if len(values) < 2:
        return 0.0
    mean = statistics.fmean(values)
    return statistics.stdev(values) / mean if mean else 0.0


def measure(path: pathlib.Path) -> dict[str, float]:
    """Compute the prose-shape metrics for one Markdown file.

    Args:
        path: The file to measure.

    Returns:
        A mapping of metric name to value. `words` is the prose word count, which is
        what the per-1,000 densities are scaled by.
    """
    text = path.read_text(errors="ignore")
    prose, content = strip_code(text)
    lengths = sentences(prose)
    words = sum(lengths) or 1
    per_k = 1000 / words
    blob = " ".join(prose).lower()
    return {
        "words": float(words),
        "sentences": float(len(lengths)),
        "headings_per_1k": sum(bool(_HEADING.match(ln)) for ln in content) * per_k,
        "bullet_line_ratio": (
            sum(bool(_BULLET.match(ln)) for ln in content)
            / max(1, sum(bool(ln.strip()) for ln in content))
        ),
        "bold_per_1k": len(_BOLD.findall(" ".join(prose))) * per_k,
        "short_ratio": sum(n <= 8 for n in lengths) / max(1, len(lengths)),
        "mid_band_ratio": sum(12 <= n <= 26 for n in lengths) / max(1, len(lengths)),
        "sentence_cov": cov(lengths),
        "paragraph_cov": cov(paragraphs(prose)),
        "em_dashes": float(text.count("—") - len(_DASH_GLYPH.findall(text))),
        "dash_glyphs": float(len(_DASH_GLYPH.findall(text))),
        "filler": float(sum(blob.count(f) for f in FILLER)),
        "clefts": float(len(CLEFT.findall(" ".join(prose)))),
        "participial_tails": float(len(PARTICIPIAL_TAIL.findall(" ".join(prose)))),
    }


def report(path: pathlib.Path, m: dict[str, float]) -> None:
    """Print one file's metrics, with the target beside each one that has a target."""
    print(f"\n{path}  ({int(m['words'])} prose words, {int(m['sentences'])} sentences)")
    print(
        f"  shape   headings/1k {m['headings_per_1k']:5.1f}   bullet-line ratio {m['bullet_line_ratio']:.2f}"
        f"   bold/1k {m['bold_per_1k']:5.1f}"
    )
    print(
        f"  rhythm  short {m['short_ratio']:.2f} (>=0.12)   mid-band {m['mid_band_ratio']:.2f} (<=0.72)"
        f"   sent CoV {m['sentence_cov']:.2f} (>=0.5)   para CoV {m['paragraph_cov']:.2f} (>=0.3)"
    )
    print(f"  syntax  clefts {int(m['clefts'])}   participial tails {int(m['participial_tails'])}")
    glyphs = f"   +{int(m['dash_glyphs'])} table glyphs (leave them)" if m["dash_glyphs"] else ""
    print(
        f"  diction em-dashes {int(m['em_dashes'])} (0 in docs/)   filler {int(m['filler'])}{glyphs}"
    )


def main(argv: list[str]) -> int:
    """Measure each path given, or every Markdown file beneath it."""
    args = [a for a in argv if not a.startswith("--")]
    summary = "--summary" in argv
    if not args:
        print(__doc__)
        return 2
    paths: list[pathlib.Path] = []
    for arg in args:
        p = pathlib.Path(arg)
        paths.extend(
            sorted(q for q in p.rglob("*.md") if "_build" not in str(q)) if p.is_dir() else [p]
        )

    rows = [(p, measure(p)) for p in paths if p.exists()]
    if not rows:
        print("no Markdown files found", file=sys.stderr)
        return 2
    if summary:
        keys = (
            "headings_per_1k",
            "bullet_line_ratio",
            "short_ratio",
            "mid_band_ratio",
            "sentence_cov",
        )
        print(f"{len(rows)} files, {int(sum(m['words'] for _, m in rows))} prose words")
        for key in keys:
            values = sorted(m[key] for _, m in rows)
            print(f"  {key:20} median {statistics.median(values):.2f}   worst {values[-1]:.2f}")
        flagged = [(m["em_dashes"], str(p)) for p, m in rows if m["em_dashes"]]
        print(
            f"  em-dashes            {int(sum(v for v, _ in flagged))} across {len(flagged)} files"
        )
        for value, name in sorted(flagged, reverse=True)[:5]:
            print(f"      {int(value):4}  {name}")
        return 0
    for path, m in rows:
        report(path, m)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
