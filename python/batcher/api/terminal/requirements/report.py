"""The worker-requirements preflight behind ``ds.explain(requirements=True)``.

A distributed UDF stage fails in the most expensive place there is when a worker cannot
rebuild its `fn`: after the cluster has scaled up, after a GPU actor has loaded its model,
on the first batch. Every input to that failure is visible on the driver beforehand — the
`fn`, what it serializes to, and which modules it names — so this reads them there, for every
UDF stage in the plan (`map_batches`, `map`, `flat_map`, `filter(fn)`, and the `ds.ml`
model stages that are built on them), and reports:

* **status** — ``"error"`` when a stage cannot be serialized at all; ``"warn"`` when a stage
  needs a *local* module (a file no installed distribution provides, which a worker has only
  if it is shipped), names a module the driver cannot import (often a worker-only package
  such as vLLM, so it is a prompt to check rather than a verdict), or carries a closure past
  the size `core.udf.processes` already warns at; ``"ok"`` otherwise.
* per stage, the installed packages (with versions) to match on the workers, the local
  modules to ship with ``runtime_env={"py_modules": [...]}`` or ``working_dir``, and the
  serialized size.

What it cannot see is the cluster: whether the workers' images carry the same packages. A
local module is reported as *covered* only when Ray is initialized here with a
``runtime_env`` whose ``working_dir`` or ``py_modules`` contains its file.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from batcher.api.terminal.requirements.scan import ModuleNeed, StageNeeds, scan_callable

if TYPE_CHECKING:
    from batcher.plan.logical import LogicalPlan

__all__ = ["annotate_requirements", "requirements_report"]


def _fat_closure_bytes() -> int:
    """The closure size the process-pool path already warns at — one threshold, not two."""
    from batcher.core.udf.processes import _FAT_CLOSURE_BYTES

    return _FAT_CLOSURE_BYTES


def _shipped_paths() -> list[str] | None:
    """Local paths the active Ray job's ``runtime_env`` ships, or `None` without Ray."""
    import sys

    ray = sys.modules.get("ray")
    if ray is None or not ray.is_initialized():
        return None
    try:
        env = dict(ray.get_runtime_context().runtime_env or {})
    except Exception:  # the report decorates; it must never fail over a context read
        return None
    entries = [env.get("working_dir"), *(env.get("py_modules") or ())]
    return [os.path.abspath(e) for e in entries if isinstance(e, str) and os.path.exists(e)]


def _covered(need: ModuleNeed, shipped: list[str] | None) -> bool | None:
    """Whether a local module's file sits under a shipped path; `None` when unknowable."""
    if shipped is None or need.path is None:
        return None
    path = os.path.abspath(need.path)
    return any(path == root or path.startswith(root.rstrip(os.sep) + os.sep) for root in shipped)


def _stage_dict(stage: StageNeeds, shipped: list[str] | None) -> dict[str, Any]:
    problems: list[str] = []
    if not stage.picklable:
        problems.append(stage.error or "cannot be serialized")
    modules = []
    for need in stage.modules:
        entry: dict[str, Any] = {"module": need.module, "kind": need.kind, "how": need.how}
        if need.kind == "package":
            entry.update(distribution=need.distribution, version=need.version)
        elif need.kind == "local":
            entry.update(path=need.path, covered=_covered(need, shipped))
        modules.append(entry)
    return {
        "stage": stage.label,
        "picklable": stage.picklable,
        "closure_bytes": stage.closure_bytes,
        "modules": modules,
        "problems": problems,
    }


def _status(stages: list[dict[str, Any]]) -> str:
    fat = _fat_closure_bytes()
    status = "ok"
    for stage in stages:
        kinds = {m["kind"] for m in stage["modules"]}
        if not stage["picklable"]:
            return "error"
        if "missing" in kinds:
            status = "warn"
        uncovered = any(
            m["kind"] == "local" and m.get("covered") is not True for m in stage["modules"]
        )
        if uncovered or (stage["closure_bytes"] or 0) >= fat:
            status = "warn"
    return status


def requirements_report(plan: LogicalPlan) -> dict[str, Any]:
    """What every UDF stage in `plan` needs on a remote worker, as a JSON-ready dict.

    Args:
        plan: The logical plan as the user built it.

    Returns:
        ``{"status": "ok"|"warn"|"error", "stages": [...]}``, one entry per UDF stage.
    """
    from importlib.metadata import packages_distributions

    from batcher.api.dataset._udf.cluster import _label, _map_stages

    stages = list(_map_stages(plan))
    if not stages:
        return {"status": "ok", "stages": []}
    distributions = packages_distributions()
    shipped = _shipped_paths()
    reports = [
        _stage_dict(scan_callable(stage.fn, _label(stage.fn), distributions), shipped)
        for stage in reversed(stages)  # source-to-sink order reads like the pipeline
    ]
    return {"status": _status(reports), "stages": reports}


def _render_module(module: dict[str, Any]) -> str:
    name = module["module"]
    if module["kind"] == "package":
        dist, version = module["distribution"], module.get("version")
        pinned = f"{dist}=={version}" if version else dist
        return pinned if dist == name else f"{name} (from {pinned})"
    if module["kind"] == "missing":
        return f"{name} [not importable on the driver; the workers must have it]"
    covered = {True: "shipped by runtime_env", False: "NOT shipped", None: "ship it"}
    return f"{name} [local: {module['path']}; {covered[module.get('covered')]}]"


def _render_text(report: dict[str, Any]) -> str:
    lines = [f"worker requirements: {report['status']}"]
    if not report["stages"]:
        lines.append("  no UDF stages; workers need only the engine")
    fat = _fat_closure_bytes()
    for stage in report["stages"]:
        size = stage["closure_bytes"]
        shipped = f"{size:,} bytes serialized" if size is not None else "not serializable"
        lines.append(f"  stage {stage['stage']!r}: {shipped}")
        lines.extend(f"    ERROR: {p}" for p in stage["problems"])
        if size is not None and size >= fat:
            lines.append(
                "    WARN: the closure carries data; load it in a class UDF's __init__ instead"
            )
        if stage["modules"]:
            lines.append("    needs: " + ", ".join(_render_module(m) for m in stage["modules"]))
        local = [
            m for m in stage["modules"] if m["kind"] == "local" and m.get("covered") is not True
        ]
        if local:
            lines.append(
                "    ship local modules with ray.init(runtime_env={'py_modules': [...]}) or "
                "a working_dir, or install them on the workers"
            )
    return "\n".join(lines)


def annotate_requirements(rendered: str, plan: LogicalPlan, fmt: str) -> str:
    """Add the worker-requirements report to an `explain()` rendering.

    Args:
        rendered: The plan as `explain` rendered it, text or a JSON document.
        plan: The logical plan as the user built it.
        fmt: ``"text"`` to append a section, ``"json"`` to add a ``"requirements"`` object.

    Returns:
        The rendering with the report added.
    """
    report = requirements_report(plan)
    if fmt == "json":
        import json

        doc = json.loads(rendered)
        doc["requirements"] = report
        return json.dumps(doc, indent=2, default=str)
    return f"{rendered.rstrip()}\n{_render_text(report)}\n"
