"""Differential function census: every DuckDB scalar/aggregate function, called with
synthesized arguments, run in DuckDB and then in Batcher's SQL front-end.

Output: three buckets — MATCH (Batcher agrees), MISMATCH (both ran, answers differ),
GAP (DuckDB ran, Batcher raised).
"""

from __future__ import annotations

import json
import sys
import warnings

warnings.filterwarnings("ignore")

import duckdb  # noqa: E402

import batcher as bt  # noqa: E402

con = duckdb.connect()

SAMPLE = {
    "BIGINT": "3",
    "INTEGER": "3",
    "SMALLINT": "3",
    "TINYINT": "3",
    "HUGEINT": "3",
    "UBIGINT": "3",
    "UINTEGER": "3",
    "DOUBLE": "2.5",
    "FLOAT": "2.5",
    "DECIMAL": "2.5",
    "VARCHAR": "'abc'",
    "BLOB": "'abc'::BLOB",
    "BOOLEAN": "true",
    "DATE": "DATE '2024-03-05'",
    "TIMESTAMP": "TIMESTAMP '2024-03-05 06:07:08'",
    "TIMESTAMP WITH TIME ZONE": "TIMESTAMPTZ '2024-03-05 06:07:08'",
    "TIME": "TIME '06:07:08'",
    "INTERVAL": "INTERVAL 1 DAY",
    "BIGINT[]": "[1,2,3]",
    "DOUBLE[]": "[1.0,2.0,3.0]",
    "VARCHAR[]": "['a','b']",
    "ANY[]": "[1,2,3]",
    "ANY": "3",
    "BIT": "'0101'::BIT",
    "UUID": "gen_random_uuid()",
    "JSON": "'{\"a\":1}'",
}

# Functions whose result is nondeterministic or environment-bound: comparing them is
# meaningless, so they are excluded from the census rather than reported as mismatches.
SKIP_NAMES = {
    "random",
    "setseed",
    "uuid",
    "uuidv4",
    "uuidv7",
    "gen_random_uuid",
    "now",
    "get_current_timestamp",
    "transaction_timestamp",
    "current_date",
    "current_localtime",
    "current_localtimestamp",
    "current_timestamp",
    "txid_current",
    "version",
    "current_query",
    "current_schema",
    "current_schemas",
    "current_database",
    "current_setting",
    "in_search_path",
    "sleep_ms",
    "vector_type",
    "index_key",
    "alias",
    "stats",
    "typeof",
    "get_type",
    "make_type",
    "replace_type",
    "cast_to_type",
    "can_cast_implicitly",
    "hash",
    "uuid_extract_timestamp",
    "uuid_extract_version",
}


def arg_for(t: str) -> str | None:
    t = (t or "").upper()
    if t in SAMPLE:
        return SAMPLE[t]
    if t.endswith("[]"):
        return "[1,2,3]"
    if t.startswith("DECIMAL"):
        return "2.5"
    if t.startswith("STRUCT"):
        return None
    if t.startswith("MAP"):
        return None
    return None


def call_sql(name: str, params: list[str]) -> str | None:
    args = []
    for p in params:
        a = arg_for(p)
        if a is None:
            return None
        args.append(a)
    ident = name if name.replace("_", "").isalnum() else None
    if ident is None:
        return None
    return f"SELECT {name}({', '.join(args)}) AS r"


def main() -> None:
    rows = con.execute(
        """
        SELECT DISTINCT function_name, function_type, parameter_types
        FROM duckdb_functions()
        WHERE function_type IN ('scalar', 'aggregate')
        ORDER BY function_name, parameter_types::VARCHAR
        """
    ).fetchall()

    by_name: dict[str, tuple[str, list]] = {}
    for name, ftype, params in rows:
        if name in SKIP_NAMES:
            continue
        by_name.setdefault(name, (ftype, []))[1].append(list(params or []))

    seen: set[str] = set()
    gaps: list[tuple[str, str, str]] = []
    mismatch: list[tuple[str, str, str, str]] = []
    match: list[str] = []

    for name, (ftype, overloads) in sorted(by_name.items()):
        # Try every overload: a function counts as supported if *any* signature agrees
        # with DuckDB, so a single unrepresentable argument type (BLOB, BIT, LIST) does
        # not report the whole function as a gap.
        best: tuple[str, str, str] | None = None
        matched = False
        for params in overloads:
            sql = call_sql(name, params)
            if sql is None:
                continue
            try:
                expected = con.execute(sql).fetchall()[0][0]
            except Exception:
                continue
            seen.add(name)
            try:
                got = bt.sql(sql).to_pydict()["r"][0]
            except Exception as exc:
                if best is None:
                    best = ("gap", f"{type(exc).__name__}: {exc}".split("\n")[0][:130], "")
                continue
            if same(expected, got):
                matched = True
                break
            best = ("mismatch", repr(expected)[:60], repr(got)[:60])
        if name not in seen:
            continue
        if matched:
            match.append(name)
        elif best is None or best[0] == "gap":
            gaps.append((name, ftype, best[1] if best else "no representable overload"))
        else:
            mismatch.append((name, ftype, best[1], best[2]))

    out = {
        "match": sorted(match),
        "gap": sorted(gaps),
        "mismatch": sorted(mismatch),
    }
    print(json.dumps(out, indent=1, default=str))
    print(
        f"\n# probed={len(seen)} match={len(match)} gap={len(gaps)} mismatch={len(mismatch)}",
        file=sys.stderr,
    )


def same(a, b) -> bool:
    if a is None and b is None:
        return True
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        try:
            return abs(float(a) - float(b)) <= 1e-9 * max(1.0, abs(float(a)))
        except (OverflowError, ValueError):
            return str(a) == str(b)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b, strict=False))
    return str(a) == str(b)


if __name__ == "__main__":
    main()
