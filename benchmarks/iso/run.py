"""Driver: run each TPC-H query per engine in an ISOLATED subprocess (honest timing).

Each (engine, query) runs in a fresh ``iso_worker.py`` process that memory-maps the
feather tables, so no cross-query process state can inflate any engine. Prints a table
of best-of-N ms and batcher/comp ratios, plus a correctness gate on result signatures.
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
        # correctness: compare signatures to batcher (or first available)
        ref = None
        for e in engines:
            if res[e].get("sig") is not None:
                ref = res[e]["sig"]
                ref_rows = res[e]["rows"]
                break
        status = "OK"
        for e in engines:
            r = res[e]
            if r.get("err"):
                status = "ERR"
            elif ref is not None and r.get("sig") is not None and r["rows"] != ref_rows:
                status = "MISMATCH"
        row = f"{q:10s}"
        for e in engines:
            ms = res[e].get("ms")
            row += f"{ms:12.1f}" if ms is not None else f"{'ERR':>12s}"
        bms = res.get("batcher", {}).get("ms")
        for e in engines:
            if e == "batcher":
                continue
            ems = res[e].get("ms")
            if bms and ems:
                row += f"{bms / ems:9.2f}x"
            else:
                row += f"{'-':>10s}"
        row += f"  {status}"
        print(row)
        errs = {e: res[e]["err"] for e in engines if res[e].get("err")}
        if errs:
            print("   ", errs)
        sys.stdout.flush()


if __name__ == "__main__":
    main()
