"""Every target a `Dataset`/`GroupBy` migration hint names resolves, and every shown call runs.

`test_guidance_is_executable.py` checks the first name after ``ds.``/``bt.`` and a curated
list of idioms. That left the rest of each chain unchecked, and the hints rotted exactly
there: ``ds.write.bigquery(table)`` named a sink that does not exist, five sink entries
said "No NumPy/TFRecord/WebDataset/ClickHouse/XML sink" while ``ds.write`` had every one,
``ds.approx_quantile(column, [0.5])`` passed a list to a method that takes one float, and
the Ray Data ``iterator`` hint said ``ds.iter_rows()`` yields dicts when it yields tuples.

So this test walks *every* ``ds.…``/``bt.…`` chain (and every relative ``.method(...)``
chain) in the two tables, segment by segment, against live objects:

* each attribute must exist on the object the previous segment produced;
* a call whose arguments are all literals is executed on a small fixture whose columns
  match the names the hints use, and a resulting `Dataset` is collected;
* a call with placeholder arguments (``...``, ``<column>``, an unbound name such as
  ``other_ds``) is checked with `inspect.Signature.bind`, so its arity and keyword names
  must still be right, and the walk continues on the declared return type.

Calls on the writer, reader and ``ds.ml`` namespaces, and the few that mutate process
state, are bound rather than executed: a hint must not write files or register views.
"""

from __future__ import annotations

import ast
import datetime
import inspect
import re
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import pytest

import batcher as bt
from batcher.api.dataset.compat.guidance._dataset_table import DATASET_UNSUPPORTED
from batcher.api.dataset.compat.guidance._groupby_table import GROUPBY_UNSUPPORTED

pytestmark = pytest.mark.unit

TABLES = {"Dataset": DATASET_UNSUPPORTED, "GroupBy": GROUPBY_UNSUPPORTED}

#: Methods a hint may show but this test must not call, because calling them changes state
#: outside the query (the default session, process config, the clipboard, a remote store).
_NEVER_CALL = frozenset({"register", "set_config", "config_context", "lookup_join", "to_clipboard"})


@dataclass(frozen=True)
class _Segment:
    name: str
    args: str | None  # the text between the parentheses, or None for a plain attribute


def _fixture() -> bt.Dataset:
    """Four rows carrying every column name the hints use, typed the way the hints use them."""
    t0 = datetime.datetime(2024, 1, 1, 9, 30)
    return bt.from_pydict(
        {
            "a": [3, 1, 2, 5],
            "b": [1, 2, 3, 4],
            "x": [1.5, 2.5, 0.5, 4.0],
            "v": [1.0, 2.0, 3.0, 4.0],
            "t": [t0 + datetime.timedelta(hours=i) for i in range(4)],
            "k": ["p", "q", "p", "q"],
            "g": ["p", "q", "p", "q"],
            "key": [1, 2, 1, 2],
            "col": ["u", "w", "u", "w"],
            "flag": [True, False, True, True],
            "old": [1, 2, 3, 4],
            "value_1": [7, 8, 9, 10],
            "tags": [["m"], ["n", "o"], [], ["m"]],
        }
    )


def _namespace(ds: bt.Dataset) -> dict[str, Any]:
    """The names a literal argument may refer to; anything else marks a placeholder."""
    return {
        "bt": bt,
        "ds": ds,
        "other": _fixture(),
        "fn": lambda batch: batch,
        "n": 2,
        "offset": 1,
        "value": "p",
        "cond": bt.col("flag"),
        "a": bt.col("a"),
        "b": bt.col("b"),
    }


# --- parsing ---------------------------------------------------------------------------


def _close(text: str, start: int) -> int:
    """Index of the bracket closing the one at `start`, skipping quoted strings."""
    depth, quote, i = 0, "", start
    while i < len(text):
        ch = text[i]
        if quote:
            quote = "" if ch == quote else quote
        elif ch in "'\"":
            quote = ch
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def _segments(text: str, j: int) -> list[_Segment]:
    """Parse `.name` / `.name(args)` segments starting at `j`."""
    out: list[_Segment] = []
    while j + 1 < len(text) and text[j] == "." and (text[j + 1].isalpha() or text[j + 1] == "_"):
        m = re.match(r"\w+", text[j + 1 :])
        assert m is not None
        name, j = m.group(), j + 1 + m.end()
        args = None
        if j < len(text) and text[j] == "(":
            end = _close(text, j)
            if end < 0:
                break
            args, j = text[j + 1 : end], end + 1
        out.append(_Segment(name, args))
    return out


def _chains(text: str) -> Iterator[tuple[str, list[_Segment]]]:
    """Every ``ds.…``, ``bt.…`` and relative ``.name(…)`` chain in a hint, with its root."""
    i = 0
    while i < len(text):
        prev = text[i - 1] if i else " "
        if text.startswith(("ds.", "bt."), i) and not (prev.isalnum() or prev in "._"):
            segs = _segments(text, i + 2)
            if segs:
                yield text[i : i + 2], segs
            i += 2  # not past the chain: a chain nested in its arguments is walked too
            continue
        if text[i] == "." and prev in " (/" and i + 1 < len(text) and text[i + 1].isalpha():
            segs = _segments(text, i)
            if segs and (segs[0].args is not None or len(segs) > 1):
                yield ".", segs
            i += 1
            continue
        i += 1


# --- evaluation ------------------------------------------------------------------------


def _is_placeholder(value: Any) -> bool:
    if value is Ellipsis:
        return True
    if isinstance(value, str):
        return "..." in value or bool(re.fullmatch(r"<[^>]+>", value))
    if isinstance(value, dict):
        return any(_is_placeholder(k) or _is_placeholder(v) for k, v in value.items())
    if isinstance(value, (list, tuple, set)):
        return any(_is_placeholder(v) for v in value)
    return False


def _literal_args(args: str, ns: dict[str, Any]) -> tuple[tuple, dict] | None:
    """The evaluated ``(args, kwargs)``, or None when any argument is a placeholder."""
    try:
        pos, kw = eval(f"(lambda *a, **k: (a, k))({args})", dict(ns))
    except (NameError, SyntaxError):  # an unbound name, or prose such as `{'a': 1, ...}`
        return None
    if _is_placeholder(pos) or _is_placeholder(kw):
        return None
    return pos, kw


def _bind(method: Any, args: str) -> str | None:
    """Bind a placeholder call's shape to the signature; return the mismatch, if any."""
    try:
        call = ast.parse(f"_f({args})", mode="eval").body
    except SyntaxError:
        return None  # prose such as `<column>` outside quotes: only existence is checkable
    assert isinstance(call, ast.Call)
    try:
        sig = inspect.signature(method)
    except (TypeError, ValueError):
        return None
    positional = list(call.args)
    # A trailing positional `...` reads "and the rest": `f(uri, ...)` and `f(...)` elide
    # arguments, so what is shown must fit but what is missing is not a defect.
    elided = bool(positional) and _is_ellipsis(positional[-1])
    if elided:
        positional.pop()
    pos = [object()] * len(positional)
    kw = {k.arg: object() for k in call.keywords if k.arg is not None}
    try:
        (sig.bind_partial if elided else sig.bind)(*pos, **kw)
    except TypeError as exc:
        return str(exc)
    return None


def _is_ellipsis(node: ast.expr) -> bool:
    return isinstance(node, ast.Constant) and node.value is Ellipsis


def _stand_in(obj: Any, method: Any) -> Any | None:
    """For an unexecuted call, continue the walk on its declared return type, if known."""
    try:
        ret = inspect.signature(method).return_annotation
    except (TypeError, ValueError):
        return None
    ret = ret if isinstance(ret, str) else getattr(ret, "__name__", "")
    if ret in ("Dataset", "Self") and isinstance(obj, bt.Dataset):
        return obj
    if ret in ("Expr", "Self") and isinstance(obj, bt.Expr):
        return obj
    return None


def _callable_here(obj: Any, name: str, ds: bt.Dataset) -> bool:
    """Whether this test may actually invoke `obj.name(...)`."""
    if name in _NEVER_CALL:
        return False
    if isinstance(obj, (type(ds.write), type(ds.ml), type(bt.read))):
        return False
    module = obj.__name__ if inspect.ismodule(obj) else type(obj).__module__
    return module.split(".")[0] == "batcher"


def _walk(obj: Any, segments: list[_Segment], ds: bt.Dataset) -> str | None:
    """Resolve and, where possible, run one chain; return the first defect found."""
    ns = _namespace(ds)
    executed = True  # False once a stand-in replaces a result: later calls are bound only
    for seg in segments:
        if not hasattr(obj, seg.name):
            return f"{type(obj).__name__} has no attribute {seg.name!r}"
        attr = getattr(obj, seg.name)
        if seg.args is None:
            obj = attr
            continue
        if not callable(attr):
            return f"{seg.name!r} is not callable but is shown as a call"
        literal = _literal_args(seg.args, ns)
        if executed and literal is not None and _callable_here(obj, seg.name, ds):
            try:
                obj = attr(*literal[0], **literal[1])
            except Exception as exc:  # any rejection is the defect being reported
                return f"{seg.name}({seg.args}) raised {type(exc).__name__}: {exc}"
            continue
        executed = False
        mismatch = _bind(attr, seg.args)
        if mismatch:
            return f"{seg.name}({seg.args}) does not fit the signature: {mismatch}"
        obj = _stand_in(obj, attr)
        if obj is None:
            return None  # nothing typed to walk on; everything reached so far resolved
    if executed and isinstance(obj, bt.Dataset):
        try:
            obj.collect()
        except Exception as exc:  # any rejection is the defect being reported
            return f"collecting the result raised {type(exc).__name__}: {exc}"
    return None


def _check(root: str, segments: list[_Segment]) -> str | None:
    ds = _fixture()
    if root == "ds":
        return _walk(ds, segments, ds)
    if root == "bt":
        return _walk(bt, segments, ds)
    # A relative chain continues an object the prose names: a GroupBy, an Expr or a Dataset.
    defects = [_walk(owner, segments, ds) for owner in (ds.group_by("g"), bt.col("x"), ds)]
    return None if None in defects else defects[0]


@pytest.fixture(autouse=True)
def _registered_t() -> Iterator[None]:
    """The view hints register a dataset as ``t`` and then query it; give them that ``t``."""
    session = bt.current_session()
    session.register("t", _fixture())
    yield
    session.drop("t")


def _render(segments: list[_Segment]) -> str:
    return "".join(f".{s.name}" + ("" if s.args is None else f"({s.args})") for s in segments)


def _cases() -> list[tuple[str, str, str, list[_Segment]]]:
    return [
        (label, key, root, segs)
        for label, table in TABLES.items()
        for key, text in table.items()
        for root, segs in _chains(text)
    ]


_CASES = _cases()


def test_the_parser_found_the_chains() -> None:
    """Guard the parametrisation below: an empty walk would pass every case vacuously."""
    assert len(_CASES) > 400
    keys = {key for _, key, _, _ in _CASES}
    assert {"approxQuantile", "write_numpy", "iterator", "createTempView"} <= keys


@pytest.mark.parametrize(
    ("label", "key", "root", "segs"),
    _CASES,
    ids=[f"{lab}[{key}]{root}{_render(segs)[:60]}" for lab, key, root, segs in _CASES],
)
def test_every_named_target_resolves(label: str, key: str, root: str, segs: list[_Segment]) -> None:
    """The chain a migrant would copy out of the traceback names real API and runs."""
    defect = _check(root, segs)
    assert defect is None, f"{label}[{key!r}] -> {root}{_render(segs)}: {defect}"


@pytest.mark.parametrize("label", sorted(TABLES))
def test_no_hint_denies_a_sink_that_exists(label: str) -> None:
    """ "No NumPy sink" while ``ds.write.numpy`` exists sends a migrant to a workaround."""
    sinks = {n.lower() for n in dir(_fixture().write) if not n.startswith("_")}
    wrong = [
        f"{label}[{key!r}] says 'No {name} sink'"
        for key, text in TABLES[label].items()
        for name in re.findall(r"\bNo (\w+) sink", text, flags=re.IGNORECASE)
        if name.lower() in sinks
    ]
    assert not wrong, "\n  ".join(["hint denies an existing sink:", *wrong])


def test_no_hint_denies_the_view_registry() -> None:
    """Sessions register named datasets and `CREATE VIEW` works, so "no view registry" lies."""
    session = bt.Session()
    session.register("t", _fixture())
    session.sql("CREATE VIEW v AS SELECT a FROM t")
    assert {"t", "v"} <= set(session.list())
    wrong = [k for k, text in DATASET_UNSUPPORTED.items() if "no view registry" in text.lower()]
    assert not wrong, f"hints deny the view registry: {wrong}"


def test_iter_rows_claims_match_its_behaviour() -> None:
    """``iter_rows()`` yields tuples; only ``named=True`` yields dicts."""
    ds = bt.from_pydict({"a": [1]})
    assert next(ds.iter_rows()) == (1,)
    assert next(ds.iter_rows(named=True)) == {"a": 1}
    wrong = [
        k
        for k, text in DATASET_UNSUPPORTED.items()
        if "dicts" in text and "iter_rows()" in text and "iter_rows(named=True)" not in text
    ]
    assert not wrong, f"hints claim iter_rows() yields dicts: {wrong}"


@pytest.mark.parametrize(
    "broken",
    [
        "ds.approx_quantile('x', [0.5])",  # a list where one float is taken
        "ds.write.teradata(path)",  # a sink that does not exist
        "ds.sort('x').head(1).no_such_method()",  # a later segment that does not exist
        "ds.union(other, by_name=True)",  # a keyword the method does not take
        "ds.group_by('g').agg(n=bt.col('missing').sum())",  # fails only once collected
    ],
)
def test_the_walker_would_notice_a_broken_hint(broken: str) -> None:
    """The guard on the guard: each historical failure shape must still be caught."""
    root, segs = next(_chains(broken))  # the outermost chain; nested ones come after it
    assert _check(root, segs) is not None
