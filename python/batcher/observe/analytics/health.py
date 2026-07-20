"""The engine's current verdict — a small set of checks over recent runs.

Every check that fires carries the runs that caused it, so a verdict is never a dead end:
the UI links straight from the claim to its evidence.
"""

from __future__ import annotations

import time
from typing import Any

__all__ = ["health_report"]


def health_report(
    queries: list[dict[str, Any]], details: list[dict[str, Any]], system: dict[str, Any]
) -> dict[str, Any]:
    """A single verdict for the engine right now, plus the checks behind it.

    Deliberately a small set of binary checks with their evidence rather than a weighted
    score. A score compresses "one query failed" and "memory is at 94%" into one number
    nobody can act on, and invites tuning the number instead of the engine.

    Args:
        queries: Run summaries.
        details: Full run documents for the profiled runs.
        system: The `system_snapshot` document.

    Returns:
        ``{"status": "ok"|"warn"|"critical", "checks": [...], "uptime_s": float}``.
    """
    checks: list[dict[str, Any]] = []
    finished = [q for q in queries if q.get("status") != "running"]
    failed = [q for q in queries if q.get("status") == "error"]

    checks.append(
        _check(
            "Failures",
            "critical" if failed else "ok",
            f"{len(failed)} of {len(finished)} runs failed" if finished else "No runs yet",
            "Open the failed run and read its error." if failed else "",
            runs=[q.get("query_id", "") for q in failed[:8]],
        )
    )

    spilled = [
        d for d in details if any(n.get("spilled") for n in (d.get("dag") or {}).get("nodes", []))
    ]
    checks.append(
        _check(
            "Spilling",
            "warn" if spilled else "ok",
            f"{len(spilled)} run(s) spilled to disk" if spilled else "No run spilled",
            "Raise memory.max_memory_bytes, or filter earlier." if spilled else "",
            runs=[d.get("query_id", "") for d in spilled[:8]],
        )
    )

    host = system.get("host") or {}
    total, free = host.get("memory_total_bytes"), host.get("memory_available_bytes")
    if total and free is not None:
        used = 1 - free / total
        checks.append(
            _check(
                "Host memory",
                "critical" if used > 0.95 else "warn" if used > 0.85 else "ok",
                f"{used * 100:.0f}% of {_gib(total)} in use",
                "Other processes are competing for RAM." if used > 0.85 else "",
            )
        )

    misestimates = [
        d
        for d in details
        for n in (d.get("dag") or {}).get("nodes", [])
        if n.get("est_error") and (n["est_error"] > 10 or n["est_error"] < 0.1)
    ]
    checks.append(
        _check(
            "Plan estimates",
            "warn" if misestimates else "ok",
            f"{len(misestimates)} step(s) missed their row estimate by 10x or more"
            if misestimates
            else "Estimates were within 10x",
            "Re-run: measured cardinalities are learned and reused." if misestimates else "",
            runs=list(dict.fromkeys(d.get("query_id", "") for d in misestimates))[:8],
        )
    )

    if not (system.get("engine") or {}).get("native_loaded", True):
        checks.append(
            _check("Engine", "critical", "The native engine is not loaded", "Run `just build`.")
        )

    rank = {"ok": 0, "warn": 1, "critical": 2}
    status = max((c["status"] for c in checks), key=lambda s: rank[s], default="ok")
    return {"status": status, "checks": checks, "uptime_s": time.time() - _STARTED_AT}


def _check(
    name: str, status: str, detail: str, action: str, runs: list[str] | None = None
) -> dict[str, Any]:
    """One health check. `runs` are the run ids it fired on, so the UI can link to them."""
    return {
        "name": name,
        "status": status,
        "detail": detail,
        "action": action,
        "runs": [r for r in (runs or []) if r],
    }


def _gib(n: int) -> str:
    return f"{n / 2**30:.1f} GiB"


#: Process start, so the dashboard can report how long this engine has been up.
_STARTED_AT = time.time()
