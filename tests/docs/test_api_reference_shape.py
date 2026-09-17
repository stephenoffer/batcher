"""The generated API reference stays split into pages a reader can hold.

A class documented with ``.. autoclass:: X`` and ``:members:`` renders every member inline,
so a few source lines become one enormous page: ``Dataset``, ``Expr`` and the ``.str``
namespace each rendered past 20,000 words that way, on pages whose Markdown was under 150
lines. The large surfaces are therefore documented as a class overview (``:no-members:``)
plus ``.. autosummary::`` tables grouped by task, and every member gets its own generated
page.

That split has one way to go wrong silently: a member missing from every table has no
page, and the global coverage test cannot see it, because it matches bare names and
``filter`` or ``count`` exists on several classes. So this module holds a split class to
the exact list of its public members.

The rendered size itself is checked after the HTML build, by ``tools/check_page_size.py``
in ``just docs``, because only the built page knows how long it is.
"""

from __future__ import annotations

import importlib
import inspect
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_DOCS = Path(__file__).resolve().parents[2] / "docs"

_EVAL_RST = re.compile(r"```\{eval-rst\}(.*?)```", re.DOTALL)
_CURRENTMODULE = re.compile(r"^\s*\.\. currentmodule:: ([\w.]+)", re.MULTILINE)
_AUTOCLASS = re.compile(
    r"^\s*\.\. autoclass:: ([\w.]+)(.*?)(?=^\s*\.\. |\Z)", re.DOTALL | re.MULTILINE
)
_NO_MEMBERS = re.compile(r"^\s*:no-members:", re.MULTILINE)
_AUTOSUMMARY = re.compile(r"^\s*\.\. autosummary::(.*?)(?=^\s*\.\. |\Z)", re.DOTALL | re.MULTILINE)
_ENTRY = re.compile(r"^~?([\w.]+)$")

# Dunders a split class may expose on purpose. Everything else starting with `_` is private.
_PROTOCOL_DUNDERS = frozenset(
    {"__getitem__", "__len__", "__iter__", "__contains__", "__arrow_c_stream__"}
)


def _resolve(dotted: str) -> object:
    parts = dotted.split(".")
    for i in range(len(parts), 0, -1):
        try:
            obj: object = importlib.import_module(".".join(parts[:i]))
        except ModuleNotFoundError:
            continue
        for attr in parts[i:]:
            obj = getattr(obj, attr)
        return obj
    raise ImportError(dotted)


def _api_pages() -> list[Path]:
    return sorted(p for p in (_DOCS / "api").rglob("*.md") if "generated" not in p.parts)


def _qualify(target: str, prefix: str) -> str:
    """Resolve an autodoc target the way Sphinx does: as written, then under currentmodule."""
    try:
        _resolve(target)
        return target
    except (ImportError, AttributeError):
        return prefix + target


def _scan() -> tuple[dict[str, str], set[str]]:
    """Return the split classes (qualified name -> page) and every autosummary entry."""
    split: dict[str, str] = {}
    entries: set[str] = set()
    for page in _api_pages():
        prefix = ""
        for block in _EVAL_RST.findall(page.read_text(encoding="utf-8")):
            scope = _CURRENTMODULE.search(block)
            if scope:
                prefix = f"{scope.group(1)}."
            for target, body in _AUTOCLASS.findall(block):
                if _NO_MEMBERS.search(body):
                    split[_qualify(target, prefix)] = str(page.relative_to(_DOCS))
            for body in _AUTOSUMMARY.findall(block):
                for line in body.splitlines():
                    match = _ENTRY.match(line.strip())
                    if match:
                        entries.add(prefix + match.group(1))
    return split, entries


def _public_members(cls: type, documented_elsewhere: frozenset[type] = frozenset()) -> set[str]:
    """Public methods and properties a reader can call on `cls`, including inherited ones.

    A member inherited from another split class is documented on that class's page, so it
    is not owed a second entry here.
    """
    names: set[str] = set()
    for name in dir(cls):
        if name.startswith("_") and name not in _PROTOCOL_DUNDERS:
            continue
        for owner in cls.__mro__:
            if name in vars(owner):
                member = vars(owner)[name]
                break
        else:
            continue
        if owner in documented_elsewhere or not owner.__module__.startswith("batcher"):
            continue
        if inspect.isfunction(member) or isinstance(member, (property, staticmethod, classmethod)):
            names.add(name)
    return names


def test_split_classes_exist() -> None:
    """Guard the guard: the reference must actually contain split classes to check."""
    split, _ = _scan()
    assert split, "no `.. autoclass:: ... :no-members:` found under docs/api"


@pytest.mark.parametrize("qualified", sorted(_scan()[0]))
def test_split_class_lists_every_member(qualified: str) -> None:
    split, entries = _scan()
    cls = _resolve(qualified)
    assert inspect.isclass(cls), f"{qualified} is not a class"
    short = qualified.rsplit(".", 1)[-1]
    listed = {
        e.rsplit(".", 1)[1]
        for e in entries
        if "." in e
        and (e.rsplit(".", 1)[0] == qualified or e.rsplit(".", 1)[0].endswith(f".{short}"))
    }
    others = frozenset(_resolve(q) for q in split if q != qualified)
    missing = sorted(_public_members(cls, others) - listed)
    assert not missing, (
        f"{qualified} (documented on {split[qualified]} with :no-members:) has "
        f"{len(missing)} public member(s) in no autosummary table: {missing}\n"
        f"Add each as `{short}.<name>` to the table for the task it belongs to."
    )
