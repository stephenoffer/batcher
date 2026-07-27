"""Spark SQL census: run Spark's own documented examples through `bt.sql`.

Spark annotates every builtin with `@ExpressionDescription(examples = ...)` holding
`> SELECT _FUNC_(args);` lines followed by the expected output. `FunctionRegistry.scala`
maps each expression class to its SQL name. Together they are an executable oracle that
needs no JVM: substitute `_FUNC_`, run the query in Batcher, compare to Spark's own
documented answer.
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import sys
import warnings

warnings.filterwarnings("ignore")

import batcher as bt  # noqa: E402

DIALECT = "spark"
SPARK = pathlib.Path(os.environ.get("SPARK_SOURCE", "/mnt/shared_storage/ref/spark"))
CATALYST = SPARK / "sql/catalyst/src/main/scala/org/apache/spark/sql/catalyst/expressions"
REGISTRY = (
    SPARK / "sql/catalyst/src/main/scala/org/apache/spark/sql/catalyst"
    "/analysis/FunctionRegistry.scala"
)

# `expression[Sqrt]("sqrt")` / `expressionBuilder("split", …)` etc.
_REG = re.compile(r'expression(?:Builder|GeneratorOuter)?\[?(\w+)?\]?\s*\(\s*"([\w$]+)"')
_DESC = re.compile(r"@ExpressionDescription\((.*?)\n\)", re.S)
_CASE = re.compile(r"\ncase class (\w+)|\ncase object (\w+)|\nobject (\w+)")
_EXAMPLE = re.compile(r"^\s*>\s*(SELECT .*?);\s*$", re.I)


def registered_names() -> dict[str, str]:
    """Expression class name → its SQL function name."""
    text = REGISTRY.read_text()
    out: dict[str, str] = {}
    for klass, name in _REG.findall(text):
        if klass and klass not in out:
            out[klass] = name
    return out


def examples_by_class() -> dict[str, list[tuple[str, str]]]:
    """Expression class name → its documented (query, expected) example pairs."""
    out: dict[str, list[tuple[str, str]]] = {}
    for path in CATALYST.rglob("*.scala"):
        try:
            text = path.read_text()
        except OSError:
            continue
        if "@ExpressionDescription" not in text:
            continue
        for match in re.finditer(r"@ExpressionDescription\(", text):
            tail = text[match.end() : match.end() + 8000]
            klass = _CASE.search(tail)
            if klass is None:
                continue
            name = next(g for g in klass.groups() if g)
            pairs = []
            lines = tail[: klass.start()].splitlines()
            for i, line in enumerate(lines):
                query = _EXAMPLE.match(line)
                if query is None or i + 1 >= len(lines):
                    continue
                expected = lines[i + 1].strip()
                if expected.startswith(">") or not expected:
                    continue
                pairs.append((query.group(1).strip(), expected))
            if pairs:
                out.setdefault(name, []).extend(pairs)
    return out


def normalize(value) -> str:
    """Spark's CLI rendering of a value, so a Batcher result can be compared to it."""
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if value != value:
            return "NaN"
        if value == int(value) and abs(value) < 1e15:
            return f"{value:.1f}"
        return repr(value)
    if isinstance(value, list):
        return "[" + ",".join(normalize(v) for v in value) + "]"
    if isinstance(value, dict):
        return "{" + ",".join(f"{k} -> {normalize(v)}" for k, v in value.items()) + "}"
    return str(value)


def out_of_scope(name: str) -> str | None:
    """Why `name` is not a function Batcher is measured against, or None.

    The same denominator audit the DuckDB census needed. Spark's builtin list carries a
    large tail that says nothing about a query engine's expression surface: the cluster
    and session it is running in, deliberately nondeterministic values, its own bitmap
    and variant encodings, a geospatial extension, and an XPath/XML family. Counting them
    as missing functions makes the figure a statement about Spark's product scope rather
    than about Batcher.
    """
    if name.startswith("st_"):
        return "geospatial extension"
    if name.startswith("xpath") or "xml" in name:
        return "XPath / XML family"
    if name.startswith(("bitmap_", "variant_")) or "variant" in name:
        return "Spark-internal bitmap or variant encoding"
    if name in _SESSION:
        return "session or cluster introspection"
    if name in _NONDETERMINISTIC:
        return "nondeterministic by definition"
    return None


_SESSION = frozenset(
    {
        "current_catalog", "current_database", "current_user", "current_schema",
        "current_path", "version", "spark_partition_id", "monotonically_increasing_id",
        "input_file_block_length", "input_file_block_start", "input_file_name",
        "typeof", "collation", "assert_true", "reflect", "try_reflect", "java_method",
    }
)  # fmt: skip
_NONDETERMINISTIC = frozenset({"random", "randn", "randstr", "uniform", "uuid", "shuffle"})


def main() -> None:
    names = registered_names()
    examples = examples_by_class()
    gap, mismatch, match = [], [], []
    scope: list[tuple[str, str]] = []
    probed = set()
    for klass, pairs in sorted(examples.items()):
        fn = names.get(klass)
        if fn is None:
            continue
        reason = out_of_scope(fn)
        if reason is not None:
            scope.append((fn, reason))
            continue
        for query, expected in pairs:
            sql = query.replace("_FUNC_", fn)
            if " FROM " in sql.upper() or "(" not in sql:
                continue  # needs a table, or is not a call
            probed.add(fn)
            try:
                out = bt.sql(sql, dialect=DIALECT).to_pydict()
            except Exception as exc:
                gap.append((fn, sql[:70], f"{type(exc).__name__}: {exc}".split("\n")[0][:100]))
                break
            got = normalize(next(iter(out.values()))[0])
            if got.lower() == expected.lower():
                match.append(fn)
            else:
                mismatch.append((fn, sql[:70], expected[:40], got[:40]))
            break  # one example per function is enough for a census
    print(
        json.dumps(
            {
                "match": sorted(set(match)),
                "gap": gap,
                "mismatch": mismatch,
                "out_of_scope": sorted(set(scope)),
            },
            indent=1,
            default=str,
        )
    )
    # The denominator is match + gap + mismatch; `out_of_scope` is excluded because it
    # measures Spark's product scope rather than Batcher's expression surface.
    scored = len(set(match)) + len(gap) + len(mismatch)
    pct = 100.0 * len(set(match)) / scored if scored else 0.0
    print(
        f"\n# scored={scored} match={len(set(match))} ({pct:.0f}%) gap={len(gap)} "
        f"mismatch={len(mismatch)} | out_of_scope={len(set(scope))} "
        f"(registered={len(names)} documented={len(examples)} probed={len(probed)})",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
