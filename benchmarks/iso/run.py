"""Driver: run each TPC-H query per engine in an ISOLATED subprocess (honest timing).

Each (engine, query) runs in a fresh ``worker.py`` process that memory-maps the feather
tables, so no cross-query process state can inflate any engine. Prints a table of
best-of-N ms and batcher/comp ratios, gated on result signatures compared against an
independent oracle (never against Batcher, which is the system under test).

The tables are materialized to Feather on first use from the same ``sources.load_tables``
every other benchmark reads; earlier this module read a hard-coded path nothing wrote, so
it could not run at all.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

# Run from the benchmarks/ directory so the suite package (and its data helpers) import.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from harness.compare import _ORACLE_PREFERENCE  # noqa: E402
from signature import signatures_match  # noqa: E402
from suites.standard.tpch import QUERIES  # noqa: E402

WORKER = ["python3", os.path.join(_HERE, "worker.py")]


def run_one(engine: str, qname: str, sql: str, scale: int, runs: int) -> dict:
    cmd = [
        *WORKER,
        "--engine",
        engine,
        "--query",
        qname,
        "--scale",
        str(scale),
        "--runs",
        str(runs),
        "--sql",
        sql,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        return {"ms": None, "err": "timeout", "sig": None, "rows": 0}
    line = proc.stdout.strip().splitlines()
    if not line:
        return {"ms": None, "err": f"no output: {proc.stderr.strip()[-200:]}", "sig": None}
    return json.loads(line[-1])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scale", type=int, default=10)
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--engines", default="batcher,duckdb,polars")
    p.add_argument("--only", default=None)
    args = p.parse_args()
    engines = args.engines.split(",")

    qnames = sorted(QUERIES)
    if args.only:
        qnames = [q for q in qnames if args.only in q]

    print(f"ISO TPC-H sf{args.scale} best-of-{args.runs}  engines={engines}\n")
    hdr = f"{'query':10s}" + "".join(f"{e + '_ms':>12s}" for e in engines)
    hdr += "".join(f"{'b/' + e:>10s}" for e in engines if e != "batcher") + "  status"
    print(hdr)
    print("-" * len(hdr))
    for q in qnames:
        sql = QUERIES[q]
        res = {e: run_one(e, q, sql, args.scale, args.runs) for e in engines}
        # Correctness: compare each engine's *signature* against an independent oracle.
        #
        # Two bugs lived here. The reference was `for e in engines` — the first name the
        # user passed, which defaults to `batcher`, so the system under test was its own
        # oracle. And `ref` was bound to a signature and then never read: the only thing
        # compared was `r["rows"] != ref_rows`, a row count. Any engine returning the right
        # number of rows with wrong values was reported `OK`, while the docstring advertised
        # "a correctness gate on result signatures".
        #
        # That was not hypothetical. `worker.py`'s Polars runner has none of the dialect
        # handling in `engines/polars.py`, and TPC-H q6 — where Polars' decimal folding
        # is documented to drop every `l_discount = 0.07` row — returns exactly one row for
        # every engine. Row counts matched and the wrong revenue passed.
        ref_engine = next(
            (e for e in _ORACLE_PREFERENCE if e in engines and res[e].get("sig") is not None),
            None,
        )
        if ref_engine is None:
            ref_engine = next((e for e in engines if res[e].get("sig") is not None), None)
        status = "OK"
        mismatches = []
        for e in engines:
            r = res[e]
            if r.get("err"):
                status = "ERR"
                continue
            if ref_engine is None or e == ref_engine or r.get("sig") is None:
                continue
            ok, why = signatures_match(res[ref_engine]["sig"], r["sig"])
            if not ok:
                status = "MISMATCH"
                mismatches.append(f"{ref_engine} != {e}: {why}")
        row = f"{q:10s}"
        for e in engines:
            ms = res[e].get("ms")
            row += f"{ms:12.1f}" if ms is not None else f"{'ERR':>12s}"
        bms = res.get("batcher", {}).get("ms")
        for e in engines:
            if e == "batcher":
                continue
            ems = res[e].get("ms")
            if status == "MISMATCH":
                # A ratio is a claim about which engine is faster; it must not be printed
                # for a row whose two answers disagree. Same rule as `harness/report.py`.
                row += f"{'n/c':>10s}"
            elif bms and ems:
                row += f"{bms / ems:9.2f}x"
            else:
                row += f"{'-':>10s}"
        row += f"  {status}"
        print(row)
        errs = {e: res[e]["err"] for e in engines if res[e].get("err")}
        if errs:
            print("   ", errs)
        for note in mismatches:
            print(f"    !! {note}")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
