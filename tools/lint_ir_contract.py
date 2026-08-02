"""Mechanically check the JSON IR wire contract: Python tags == Rust serde tags.

`CLAUDE.md` invariant #8 says the Python `to_ir()` tags and the Rust `serde` tags are
one contract that must change in the same commit. Until this checker existed, nothing
enforced that. The stated reconciliation was "the round-trip / differential tests"
(`plan/ir_tags.py`), and those only cover a tag some test actually *exercises* — a
vocabulary entry no differential test names could drift silently in either direction.
That is not hypothetical: `io/predicate.py` once emitted `{"node": "null"}` against
`bc_io::Pred::IsNull`'s `"is_null"`, which made every filter containing a null test
prune zero row-groups, and it shipped because pruning only ever affects speed.

The check is a *string* comparison against the tags Rust will actually accept, derived
from the enum definitions the same way serde derives them:

* the container's `rename_all` (`snake_case` throughout) applied to each variant name,
* overridden by a per-variant `#[serde(rename = "...")]`,
* skipping `#[serde(skip)]` variants.

`tests/unit/test_ir_contract.py` runs this in CI. Run it directly for the report:

    python tools/lint_ir_contract.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CRATES = REPO / "crates"

# Each entry pairs a Rust enum with the Python object that must hold the same tag set.
# `python` is a dotted path to a class (its public `str` attributes are the vocabulary)
# or to a `frozenset`/`StrEnum` (its members are).
PAIRS: list[tuple[str, str, str]] = [
    ("bc-ir/src/lib.rs", "RelOp", "batcher.plan.ir_tags:Op"),
    ("bc-expr/src/lib.rs", "Expr", "batcher.plan.ir_tags:ExprTag"),
    ("bc-expr/src/lib.rs", "StrFunc", "batcher.plan.expr_ir.fn_names:STR_FNS"),
    ("bc-expr/src/lib.rs", "MathFunc", "batcher.plan.expr_ir.fn_names:MATH_FNS"),
    ("bc-expr/src/lib.rs", "ListFunc", "batcher.plan.expr_ir.fn_names:LIST_FNS"),
    ("bc-expr/src/lib.rs", "DateFunc", "batcher.plan.expr_ir.fn_names:DATE_FNS"),
    ("bc-expr/src/lib.rs", "GeoFunc", "batcher.plan.expr_ir.fn_names:GEO_FNS"),
    ("bc-expr/src/lib.rs", "Math2Func", "batcher.plan.expr_ir.fn_names:Math2Fn"),
    ("bc-expr/src/lib.rs", "MapFunc", "batcher.plan.expr_ir.fn_names:MapFn"),
    ("bc-expr/src/lib.rs", "ListSetOp", "batcher.plan.expr_ir.fn_names:ListSetFn"),
    ("bc-expr/src/lib.rs", "ListZipOp", "batcher.plan.expr_ir.fn_names:ListZipFn"),
    (
        "bc-expr/src/lib.rs",
        "ListBinaryFunc",
        "batcher.plan.expr_ir.fn_names:ListBinaryFn",
    ),
    (
        "bc-expr/src/lib.rs",
        "MakeTemporalFunc",
        "batcher.plan.expr_ir.fn_names:MAKE_TEMPORAL_FNS",
    ),
]


def _snake(name: str) -> str:
    """Render a Rust variant name the way `serde(rename_all = "snake_case")` does."""
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def _enum_body(src: str, enum: str) -> str:
    """Return the brace-matched body of `pub enum <enum>`."""
    marker = f"pub enum {enum} "
    start = src.index(marker)
    open_brace = src.index("{", start)
    depth = 0
    i = open_brace
    while True:
        if src[i] == "{":
            depth += 1
        elif src[i] == "}":
            depth -= 1
            if depth == 0:
                break
        i += 1
    return src[open_brace + 1 : i]


def _top_level_lines(body: str) -> list[str]:
    """Lines of the enum body at brace/paren depth 0, so field lists don't leak in.

    Depth is updated *after* a line is classified, so a variant and its attributes —
    which sit at depth 0 — are kept whole. Stripping parenthesized text instead would
    eat `#[serde(rename = "...")]`, silently turning an honoured rename into a
    spurious drift report.
    """
    kept: list[str] = []
    depth = 0
    for raw in body.splitlines():
        line = raw.strip()
        if depth == 0 and line:
            kept.append(line)
        depth += raw.count("{") + raw.count("(") - raw.count("}") - raw.count(")")
    return kept


def rust_tags(rel_path: str, enum: str) -> set[str]:
    """The exact tag strings serde will accept for `enum`, honouring renames."""
    src = (CRATES / rel_path).read_text()
    container = re.search(r"#\[serde\(([^)]*)\)\]\s*\npub enum " + re.escape(enum) + r"\b", src)
    rename_all = "snake_case" in (container.group(1) if container else "")

    tags: set[str] = set()
    pending_rename: str | None = None
    skip_next = False
    for line in _top_level_lines(_enum_body(src, enum)):
        if line.startswith("//"):
            continue
        if line.startswith("#["):
            rn = re.search(r'serde\(\s*rename\s*=\s*"([^"]+)"', line)
            if rn:
                pending_rename = rn.group(1)
            if re.search(r"serde\(\s*skip\b", line):
                skip_next = True
            continue
        m = re.match(r"^([A-Z][A-Za-z0-9]*)", line)
        if not m:
            continue
        if skip_next:
            skip_next = False
            pending_rename = None
            continue
        tags.add(pending_rename or (_snake(m.group(1)) if rename_all else m.group(1)))
        pending_rename = None
    return tags


def python_tags(dotted: str) -> set[str]:
    """The tag strings the Python control plane can emit for one vocabulary."""
    import importlib

    module_name, _, attr = dotted.partition(":")
    obj = getattr(importlib.import_module(module_name), attr)
    if isinstance(obj, frozenset | set):
        return set(obj)
    if isinstance(obj, type) and issubclass(obj, str):  # StrEnum
        return {member.value for member in obj}
    return {
        value
        for name, value in vars(obj).items()
        if not name.startswith("_") and isinstance(value, str)
    }


def main() -> int:
    sys.path.insert(0, str(REPO / "python"))
    failures = 0
    for rel_path, enum, dotted in PAIRS:
        rust = rust_tags(rel_path, enum)
        py = python_tags(dotted)
        only_rust = sorted(rust - py)
        only_py = sorted(py - rust)
        if only_rust or only_py:
            failures += 1
            print(f"FAIL {enum} ({rel_path}) vs {dotted}")
            if only_py:
                print(f"  Python emits, Rust rejects: {only_py}")
            if only_rust:
                print(f"  Rust accepts, Python never emits: {only_rust}")
        else:
            print(f"ok   {enum:18} {len(rust):3d} tags == {dotted}")

    if failures:
        print(
            f"\nlint-ir-contract: FAIL ({failures} vocabulary/vocabularies drifted).\n"
            "A tag Python emits that Rust rejects is a hard runtime error; a tag Rust\n"
            "accepts that Python never emits is dead wire surface. Change both sides in\n"
            "the same commit (CLAUDE.md invariant #8)."
        )
        return 1
    print(f"\nlint-ir-contract: OK ({len(PAIRS)} vocabularies agree)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
