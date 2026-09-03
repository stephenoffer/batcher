"""The lazy re-export façades must serve exactly what the eager ones did.

`batcher`, `batcher.api`, and `batcher.api.session` resolve their public names through
a generated routing table (`batcher._exports`) instead of importing them at package
load. That is what makes `import batcher` cost single-digit milliseconds rather than
545, and it is worth nothing if the table can silently drift from the surface.

These tests are the gate on that drift. They import every leaf module the table names
and assert the object it routes to **is** the object the eager façade bound — identity,
not equality, because a table entry pointing at a same-named function in a neighbouring
module would compare equal in every way a weaker check can see and still be the wrong
function.

The complementary property, that the surface itself did not change, is `just
surface-diff`; this file is about the mechanism that serves it.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[2]

#: The façades that resolve lazily, paired with the table each one is driven by.
_FACADES = [
    ("batcher", "EXPORTS"),
    ("batcher.api", "API_EXPORTS"),
    ("batcher.api.session", "SESSION_EXPORTS"),
]


def _table(name: str) -> dict[str, str]:
    """The generated routing table called `name`."""
    return dict(getattr(importlib.import_module("batcher._exports"), name))


@pytest.mark.parametrize(("facade", "table_name"), _FACADES)
def test_every_routed_name_resolves_to_the_object_it_names(facade: str, table_name: str) -> None:
    """Each entry imports, defines the attribute, and hands back that exact object."""
    module = importlib.import_module(facade)
    for name, target in _table(table_name).items():
        path, _, attr = target.partition(":")
        leaf = importlib.import_module(path)
        assert hasattr(leaf, attr or name), f"{facade}.{name} -> {target} does not define it"
        assert getattr(leaf, attr or name) is getattr(module, name), (
            f"{facade}.{name} routes to a different object than {target} holds"
        )


@pytest.mark.parametrize(("facade", "table_name"), _FACADES)
def test_dunder_all_is_the_routing_table(facade: str, table_name: str) -> None:
    """`__all__` is rebuilt from the table, so the two cannot disagree on what is public."""
    module = importlib.import_module(facade)
    expected = list(_table(table_name))
    if facade == "batcher":
        expected = [*expected, "__version__"]
    assert list(module.__all__) == expected


def test_the_committed_table_is_what_the_generator_produces() -> None:
    """A public name added without `just gen-exports` fails here rather than at runtime.

    Regenerating writes the file, so this compares against a copy taken first and puts
    the original back either way — the test must not leave the tree dirty.
    """
    target = _ROOT / "python" / "batcher" / "_exports.py"
    before = target.read_text()
    try:
        proc = subprocess.run(
            [sys.executable, str(_ROOT / "tools" / "gen_lazy_exports.py")],
            capture_output=True,
            text=True,
            cwd=str(_ROOT),
        )
        assert proc.returncode == 0, proc.stderr
        after = target.read_text()
    finally:
        target.write_text(before)
    assert after == before, "python/batcher/_exports.py is stale — run `just gen-exports`"


def test_import_batcher_does_not_import_the_surface() -> None:
    """The whole point, asserted as a property rather than a timing.

    A fresh interpreter that imports `batcher` and touches nothing must not have loaded
    pyarrow, numpy, or the executor — those are what the 545 ms was. Asserting the module
    set rather than a duration keeps the test meaningful on a slow or loaded machine.
    """
    code = (
        "import sys, batcher\n"
        "loaded = {m for m in sys.modules}\n"
        "heavy = sorted({'pyarrow', 'numpy', 'pandas'} & loaded)\n"
        "batcher_modules = sorted(m for m in loaded if m.startswith('batcher'))\n"
        "print(repr((heavy, batcher_modules)))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=str(_ROOT)
    )
    assert proc.returncode == 0, proc.stderr
    heavy, batcher_modules = eval(proc.stdout)
    assert heavy == [], f"import batcher pulled in {heavy}"
    # The root package, the generated table, and the façade helper. Anything else means a
    # module-scope import crept back into the façade.
    assert batcher_modules == ["batcher", "batcher._exports", "batcher._lazy"], batcher_modules


def test_the_shadow_sets_are_exactly_the_real_collisions() -> None:
    """A new export that collides with a submodule name must land in the shadow set.

    Computed from the filesystem rather than restated, so adding `batcher/api/foo.py`
    beside an exported `foo` fails here instead of resolving to whichever the importing
    script touched first.
    """
    for facade, shadow_name, table_name in [
        ("batcher", "ROOT_SHADOWED", "EXPORTS"),
        ("batcher.api", "API_SHADOWED", "API_EXPORTS"),
        ("batcher.api.session", "SESSION_SHADOWED", "SESSION_EXPORTS"),
    ]:
        directory = Path(importlib.import_module(facade).__file__ or "").parent
        names = {p.stem for p in directory.glob("*.py") if p.name != "__init__.py"}
        names |= {p.name for p in directory.iterdir() if (p / "__init__.py").exists()}
        expected = names & set(_table(table_name))
        actual = set(getattr(importlib.import_module("batcher._exports"), shadow_name))
        assert actual == expected, f"{facade}: shadow set is {actual}, collisions are {expected}"


@pytest.mark.parametrize(
    ("facade", "shadow_name"),
    [
        ("batcher", "ROOT_SHADOWED"),
        ("batcher.api", "API_SHADOWED"),
        ("batcher.api.session", "SESSION_SHADOWED"),
    ],
)
def test_a_name_that_is_also_a_submodule_still_resolves_to_the_export(
    facade: str, shadow_name: str
) -> None:
    """Importing the same-named submodule first must not change what the façade serves.

    `batcher.api.security` is a package and `security` is a function; the eager façade
    won that race by binding the function last, and laziness removes the ordering. This
    imports the submodule *first*, which is the losing order, and then asserts the
    export is still what comes back.
    """
    shadowed = getattr(importlib.import_module("batcher._exports"), shadow_name)
    module = importlib.import_module(facade)
    for name in shadowed:
        importlib.import_module(f"{facade}.{name}")
        served = getattr(module, name)
        assert not isinstance(served, type(sys)), f"{facade}.{name} resolved to the submodule"
        assert served is _resolve_through_table(facade, name)


def _resolve_through_table(facade: str, name: str) -> object:
    """What the routing table says `facade.name` is, resolved directly."""
    table_name = {
        "batcher": "EXPORTS",
        "batcher.api": "API_EXPORTS",
        "batcher.api.session": "SESSION_EXPORTS",
    }[facade]
    path, _, attr = _table(table_name)[name].partition(":")
    return getattr(importlib.import_module(path), attr or name)


def test_read_is_the_reader_namespace_and_not_the_session_function() -> None:
    """`bt.read` must be the accessor namespace, which is the richer of two same-named exports.

    `batcher.api.session` exports a plain `read` function and `batcher.api.io_namespace`
    exports the namespace object that is *also* callable with the same signature. The eager
    façade resolved the collision by assigning the namespace last. A routing table generated
    from imports sorted by isort put the function last instead, and `bt.read` came back as a
    bare function: `bt.read("f.parquet")` still worked, so every smoke test passed, while
    `bt.read.parquet(...)` — the documented spelling — raised `AttributeError`.

    Asserting the *methods* rather than the type is deliberate: that is what a user calls, and
    it is the half a same-signature callable can pass without.
    """
    import batcher as bt

    assert callable(bt.read), "bt.read(path) must still work"
    for fmt in ("parquet", "csv", "json", "delta", "iceberg"):
        assert hasattr(bt.read, fmt), f"bt.read.{fmt} is missing — read resolved to a function"


def test_a_name_bound_twice_resolves_to_the_last_binding() -> None:
    """Where two contributing modules export one name, the later declaration wins.

    The rule the eager façades ran on, and the one the generated tables have to reproduce:
    `session`'s `concat` deliberately shadows `functions`' string form, and the reader
    namespace shadows `session`'s `read`. Both are collisions the surface is *supposed* to
    have, so pinning them is how a regeneration that silently flips one gets caught.
    """
    import batcher as bt
    import batcher.api.functions as functions
    import batcher.api.session as session

    assert "concat" in functions.__all__ and "concat" in session.__all__
    assert bt.concat is session.concat, "concat must be the frame form, not the string form"
    assert bt.concat is not functions.concat


def test_a_missing_top_level_name_still_gives_migration_guidance() -> None:
    """The onboarding hint survives the rewrite — it is why `__getattr__` takes a fallback."""
    import batcher as bt

    with pytest.raises(AttributeError, match="Dataset"):
        bt.DataFrame  # noqa: B018 - the attribute access is the assertion


def test_public_subpackages_are_reachable_as_attributes() -> None:
    """`bt.ml` and friends are not imported at load, so this is the only thing exposing them."""
    import batcher as bt

    for name in ("config", "governance", "io"):
        assert importlib.import_module(f"batcher.{name}") is getattr(bt, name)
