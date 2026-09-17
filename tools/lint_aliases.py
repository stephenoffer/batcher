"""Find second spellings on the public surface: one capability reachable under two names.

Batcher keeps one spelling per capability (`CLAUDE.md`, invariant 11; `.claude/rules/
python-control-plane.md`, "one obvious way"). The rule was prose for a long time, and about
a hundred and fifty second spellings accumulated under it anyway, each one reasonable on its
own (`groupby` for pandas users, `to_dicts` for Polars users, `isna` bound onto `Expr`). A
migrating user is better served by one name and a traceback that names it, plus the
`batcher.migrate` codemod, than by a surface that is the union of four other libraries.

Five shapes are reported, three of them blocking:

``same-object``
    Two public names on one receiver bound to the same function object
    (`n_unique = count_distinct`).
``delegate``
    A public method whose body, after its docstring, is one ``return`` calling another
    public method of the same receiver and forwarding its own parameters, possibly through
    a normalizing helper (`def to_dicts(self): return self.to_pylist()`). A delegate that
    also passes a *constant* (`def month_start(self): return self.truncate("month")`) is
    reported separately as a ``preset``: it may carry a distinct meaning, and whether it
    stays is a naming decision rather than a mechanical one.
``same-ir``
    Two expression methods with independent bodies that build byte-identical IR when called
    with no arguments (`str.lower` and `str.to_lowercase`).
``wrapper``
    A one-line body that computes its arguments from something other than its own
    parameters (`def isna(self): return self.select([...])`). Reported, never blocking.

Run ``python tools/lint_aliases.py`` for the report, or ``--check`` to exit 1 when any
``same-object``, ``delegate`` or ``same-ir`` finding is not in ``ALLOW``. ``ALLOW`` may only shrink.
"""

from __future__ import annotations

import argparse
import ast
import inspect
import json
import sys
import textwrap
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Findings that are not second spellings despite their shape, as `receiver.name`, each with
# a reason. It may only shrink, and a stale entry fails `--check`.
ALLOW: dict[str, str] = {
    "StreamingQuery.process_all_available": (
        "Spark's processAllAvailable is a distinct contract (block until the input available "
        "now is processed, then return while the query keeps running) that today shares "
        "await_termination's implementation; the names mean different things."
    ),
}

# The kinds that are a second spelling by construction. `preset` and `wrapper` carry a
# meaning of their own and are reported for review only.
BLOCKING = frozenset({"same-object", "delegate", "same-ir"})


@dataclass(frozen=True)
class Finding:
    """One second spelling, with the name it duplicates."""

    kind: str
    receiver: str
    name: str
    target: str

    @property
    def key(self) -> str:
        return f"{self.receiver}.{self.name}"


def _receivers() -> dict[str, type]:
    from tools.parity.batcher_targets import receivers

    return {k: v for k, v in receivers().items() if inspect.isclass(v)}


def _function(member: Any) -> Any:
    if isinstance(member, (staticmethod, classmethod)):
        return member.__func__
    if isinstance(member, property):
        return member.fget
    return member if inspect.isfunction(member) else None


def _public_members(cls: type) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, member in vars(cls).items():
        fn = _function(member)
        if not name.startswith("_") and fn is not None:
            out[name] = fn
    return out


def _same_object(receiver: str, members: dict[str, Any]) -> list[Finding]:
    by_id: dict[int, list[str]] = defaultdict(list)
    for name, fn in members.items():
        by_id[id(fn)].append(name)
    out = []
    for names in by_id.values():
        if len(names) < 2:
            continue
        primary = getattr(members[names[0]], "__name__", names[0])
        keep = primary if primary in names else sorted(names)[0]
        out.extend(Finding("same-object", receiver, n, keep) for n in names if n != keep)
    return out


def _delegate_call(fn: Any) -> tuple[str, str] | None:
    """Classify `fn` when it is one `return self.<public>(...)`.

    Returns `(target, kind)`, where kind is ``delegate`` when the call forwards exactly the
    method's own parameters (each at most wrapped in a normalizing call), ``preset`` when it
    forwards them plus literal constants, and ``wrapper`` when it computes arguments from
    anything else (module constants, comprehensions, other attributes). Only ``delegate`` is
    a second spelling by construction; the other two carry meaning of their own.
    """
    try:
        source = textwrap.dedent(inspect.getsource(fn))
    except (OSError, TypeError):
        return None
    tree = ast.parse(source)
    if not tree.body or not isinstance(tree.body[0], ast.FunctionDef):
        return None
    func = tree.body[0]
    body = func.body
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        body = body[1:]
    if len(body) != 1 or not isinstance(body[0], ast.Return):
        return None
    call = body[0].value
    if not isinstance(call, ast.Call):
        return None
    generated = _generated_forward(fn, call)
    if generated is not None:
        return generated, "delegate"
    if isinstance(call.func, ast.Name):
        # A module-level function delegating to another one: `def f(x): return g(x)`.
        target = call.func.id
    elif (
        isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "self"
    ):
        target = call.func.attr
    else:
        return None
    if target.startswith("_") or target == func.name:
        return None
    params = {
        a.arg
        for a in [*func.args.args, *func.args.kwonlyargs, func.args.vararg, func.args.kwarg]
        if a is not None
    } - {"self"}
    args = [*call.args, *(k.value for k in call.keywords)]
    used: set[str] = set()
    kind = "delegate"
    for arg in args:
        arg_kind = _arg_kind(arg, params, used)
        if arg_kind == "wrapper":
            return target, "wrapper"
        if arg_kind == "preset":
            kind = "preset"
    if used != params:
        # A parameter the method accepts but does not forward changes behaviour.
        return target, "wrapper"
    return target, kind


def _generated_forward(fn: Any, call: ast.Call) -> str | None:
    """The target of a forwarder built at import time: `getattr(self, _t)(*args, **kwargs)`.

    A table-driven binder can stamp out one function per row that looks up its target by
    a name held in a keyword default. Its source is the template's, so the plain AST check
    sees no `self.<public>(...)` call; the target is only in `__kwdefaults__`. Such a
    forwarder passes every argument through unchanged, which makes it a second spelling.
    """
    inner = call.func
    if not (
        isinstance(inner, ast.Call)
        and isinstance(inner.func, ast.Name)
        and inner.func.id == "getattr"
        and len(inner.args) == 2
        and isinstance(inner.args[0], ast.Name)
        and inner.args[0].id == "self"
        and isinstance(inner.args[1], ast.Name)
    ):
        return None
    starred = [a for a in call.args if isinstance(a, ast.Starred)]
    if len(call.args) != len(starred) or any(k.arg is not None for k in call.keywords):
        return None
    target = (getattr(fn, "__kwdefaults__", None) or {}).get(inner.args[1].id)
    return target if isinstance(target, str) and not target.startswith("_") else None


def _arg_kind(arg: ast.AST, params: set[str], used: set[str]) -> str:
    """How one call argument relates to the method's own parameters."""
    if isinstance(arg, ast.Starred):
        arg = arg.value
    if isinstance(arg, ast.Name) and arg.id in params:
        used.add(arg.id)
        return "delegate"
    if isinstance(arg, ast.Constant):
        return "preset"
    if isinstance(arg, ast.Call) and isinstance(arg.func, ast.Name):
        # A normalizer such as `column_name(name, arg="name", api="with_row_count")`: it
        # forwards a parameter, and its own keyword constants only label error messages.
        inner = [_arg_kind(a, params, used) for a in arg.args]
        if inner and all(k == "delegate" for k in inner):
            return "delegate"
    return "wrapper"


def _delegates(receiver: str, members: dict[str, Any]) -> list[Finding]:
    out = []
    for name, fn in members.items():
        hit = _delegate_call(fn)
        if hit is None:
            continue
        target, kind = hit
        if target in members:
            out.append(Finding(kind, receiver, name, target))
    return out


def _top_level_members() -> dict[str, Any]:
    import batcher as bt

    return {
        n: obj
        for n in bt.__all__
        if inspect.isfunction(obj := getattr(bt, n, None)) and not n.startswith("_")
    }


def _zero_arg_ir(bound: Any) -> str | None:
    """The IR `bound()` builds, when it can be called with no arguments; else `None`."""
    try:
        sig = inspect.signature(bound)
    except (TypeError, ValueError):
        return None
    required = [
        p
        for p in sig.parameters.values()
        if p.default is p.empty and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
    ]
    if required:
        return None
    try:
        built = bound()
        ir = built.to_ir()
    except Exception:  # a method that cannot be built bare is simply not probed
        return None
    return json.dumps(ir, sort_keys=True, default=str)


def _same_ir(receiver: str, instance: Any) -> list[Finding]:
    """Expression methods with independent bodies that build byte-identical IR.

    A delegate is visible in the source; two methods that each construct the same node are
    not, and they are the larger share of the expression surface's second spellings
    (`str.lower`/`str.to_lowercase`). Probing is limited to methods callable with no
    arguments, so this is a floor on what exists rather than a census of it. Of a group with
    different signatures, the names with fewer parameters are presets of the widest one.
    """
    groups: dict[str, list[str]] = defaultdict(list)
    for name in sorted(_public_members(type(instance))):
        if name == "to_ir":
            continue
        ir = _zero_arg_ir(getattr(instance, name))
        if ir is not None:
            groups[ir].append(name)
    out = []
    for names in groups.values():
        if len(names) < 2:
            continue
        arity = {n: len(inspect.signature(getattr(instance, n)).parameters) for n in names}
        widest = max(arity.values())
        full = sorted(n for n in names if arity[n] == widest)
        keep = full[0]
        for n in names:
            if n != keep:
                kind = "same-ir" if arity[n] == widest else "preset"
                out.append(Finding(kind, receiver, n, keep))
    return out


def _expression_instances() -> dict[str, Any]:
    import batcher as bt

    col = bt.col("x")
    namespaces = ("str", "dt", "list", "struct", "json", "map", "image", "audio", "video", "seq")
    return {"Expr": col, **{f"Expr.{ns}": getattr(col, ns) for ns in namespaces}}


def find_all() -> list[Finding]:
    """Scan every class receiver on the public surface, and `bt` itself.

    Returns:
        Every finding, sorted by receiver and name.
    """
    top = _top_level_members()
    findings: list[Finding] = _same_object("bt", top) + _delegates("bt", top)
    for receiver, cls in _receivers().items():
        members = _public_members(cls)
        findings += _same_object(receiver, members)
        findings += _delegates(receiver, members)
    for receiver, instance in _expression_instances().items():
        findings += _same_ir(receiver, instance)
    by_key: dict[str, Finding] = {}
    for f in findings:
        # A pair found both syntactically and by IR keeps the syntactic finding, which names
        # the method the delegate actually calls.
        if f.key not in by_key or by_key[f.key].kind == "same-ir":
            by_key[f.key] = f
    return sorted(by_key.values(), key=lambda f: (f.receiver, f.name, f.kind))


def main(argv: list[str]) -> int:
    """Print the report; with `--check`, fail on any unallowed alias."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    findings = find_all()
    if args.json:
        print(json.dumps([asdict(f) for f in findings], indent=1))
    else:
        for f in findings:
            print(f"{f.kind:12} {f.key:55} -> {f.target}")
        counts = defaultdict(int)
        for f in findings:
            counts[f.kind] += 1
        print(f"\n{dict(counts)}", file=sys.stderr)
    if not args.check:
        return 0
    blocking = [f for f in findings if f.kind in BLOCKING and f.key not in ALLOW]
    stale = sorted(set(ALLOW) - {f.key for f in findings})
    for key in stale:
        print(f"ALLOW entry {key} no longer matches a finding; remove it", file=sys.stderr)
    return 1 if blocking or stale else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
