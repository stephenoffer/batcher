"""No connector renders a credential it was given, swept over the registry.

`test_credential_redaction.py` already guards this from two directions and this file exists
for the gap between them. Its static scan reads dataclass *field names* against a list of
secret-shaped hints, which cannot see a credential buried in a dict -- its own comment says
so, and records that three lakehouse leaks survived it. Its behavioural check does look at a
real secret *value*, but over a hand-written list of four sources, and only in `repr()`.

The gap is the one that file's docstring argues against having: "a *new* connector cannot
reintroduce the leak by simply not being in a hand-written list". The value check is such a
list, it holds no sinks at all, and a sink takes credentials exactly as a source does.

So this sweeps both registries, gives every credential-shaped parameter a real secret, and
looks for that secret in every string a connector renders itself into -- `repr`, `str`, and
the naming hooks (`identity`, `table_name`, `governed_name`) that end up in plan caches,
statistics keys and governance decisions rather than only in a traceback.

Why the value and not the field name: a `repr` that prints `_conn_kwargs={'password': ...}`
leaks without any field being called `password`, and that is the shape the static scan is
blind to.
"""

from __future__ import annotations

import inspect

import pytest

from batcher.io.formats.base import SINKS, SOURCES

pytestmark = pytest.mark.unit

#: Distinctive enough that finding it in a rendered string is unambiguous.
SECRET = "hunter2-SUPERSECRET-do-not-render"

#: Parameter-name fragments that mean "this argument is a credential". Deliberately broad:
#: the cost of a false positive is passing a secret to a field that did not want one, which
#: only makes the test stricter, while a false negative is a credential never under test.
_CREDENTIAL_HINTS = (
    "password",
    "token",
    "secret",
    "credential",
    "auth",
    "connstr",
    "conn_uri",
    "connection_string",
    "api_key",
    "access_key",
)

#: The strings a connector renders itself into. `repr` is where a traceback finds it;
#: `identity` and the governance names travel further -- into the statistics cache key and
#: into an audit record -- so a secret in one of those outlives the process.
_RENDERERS = ("repr", "str", "identity", "table_name", "governed_name")


def _poisoned_kwargs(cls) -> dict[str, object] | None:
    """The least `cls` needs to construct, with every credential-shaped argument poisoned."""
    try:
        signature = inspect.signature(cls)
    except (TypeError, ValueError):
        return None
    kwargs: dict[str, object] = {}
    for parameter in signature.parameters.values():
        if parameter.name == "self":
            continue
        if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            continue
        if any(hint in parameter.name for hint in _CREDENTIAL_HINTS):
            kwargs[parameter.name] = SECRET
        elif parameter.default is inspect._empty:
            name = parameter.name
            kwargs[name] = "/data/table" if ("path" in name or "uri" in name) else "x"
    return kwargs


def _rendered(connector) -> dict[str, str]:
    """Every string `connector` will render itself into, by renderer name."""
    out: dict[str, str] = {}
    for renderer in _RENDERERS:
        try:
            if renderer == "repr":
                out[renderer] = repr(connector)
            elif renderer == "str":
                out[renderer] = str(connector)
            else:
                hook = getattr(connector, renderer, None)
                if callable(hook):
                    out[renderer] = str(hook())
        except Exception:
            # A renderer that raises leaks nothing. Constructing these with placeholder
            # arguments makes some of them raise, and that is not what is under test.
            continue
    return out


def _built(registry) -> list[tuple[str, object]]:
    out = []
    for name in sorted(registry):
        cls = registry.get(name)
        kwargs = _poisoned_kwargs(cls)
        if kwargs is None:
            continue
        try:
            out.append((name, cls(**kwargs)))
        except Exception:
            # Needs a real driver, a live endpoint, or arguments this generic guess cannot
            # supply. The floors below are what stop that quietly emptying the sweep.
            continue
    return out


def _holds_the_secret(connector) -> list[str]:
    """The attributes on `connector` that actually store the secret."""
    slots = [s for c in type(connector).__mro__ for s in getattr(c, "__slots__", ())]
    names = set(slots) | set(getattr(connector, "__dict__", {}))
    held = []
    for attribute in names:
        try:
            if SECRET in str(getattr(connector, attribute, "")):
                held.append(attribute)
        except Exception:
            continue
    return sorted(held)


@pytest.mark.parametrize("registry_name", ["SOURCES", "SINKS"])
def test_no_connector_renders_the_secret_it_was_given(registry_name):
    """The property, over the registry rather than over a list someone maintained."""
    registry = {"SOURCES": SOURCES, "SINKS": SINKS}[registry_name]
    leaking = [
        (name, renderer)
        for name, connector in _built(registry)
        for renderer, text in _rendered(connector).items()
        if SECRET in text
    ]
    assert leaking == [], (
        f"these {registry_name} render a credential they were handed, which puts it in a "
        f"traceback, a plan cache key or an audit record: {leaking}"
    )


@pytest.mark.parametrize(("registry_name", "floor"), [("SOURCES", 15), ("SINKS", 15)])
def test_the_sweep_built_something(registry_name, floor):
    """Without this the assertion above passes when nothing constructs.

    That is not hypothetical for this shape: the first version of the equivalent sweep in
    `test_governed_source_names.py` skipped whatever raised and silently covered none of the
    database, warehouse or document-store family -- thirteen connectors, every one of them
    holding the defect being looked for.
    """
    registry = {"SOURCES": SOURCES, "SINKS": SINKS}[registry_name]
    built = _built(registry)
    assert len(built) >= floor, f"only {len(built)} of {len(registry)} {registry_name} built"


@pytest.mark.parametrize("registry_name", ["SOURCES", "SINKS"])
def test_the_secret_actually_reached_some_connector(registry_name):
    """The control, and the one that makes the sweep mean anything.

    A connector that never received the secret cannot render it, so a sweep where *none*
    received one is a sweep that proves nothing while passing. This requires that several
    constructed connectors genuinely hold the value somewhere reachable -- so the absence
    asserted above is an absence from the *rendering*, not from the object.
    """
    registry = {"SOURCES": SOURCES, "SINKS": SINKS}[registry_name]
    holders = [(name, _holds_the_secret(c)) for name, c in _built(registry)]
    with_secret = [(name, held) for name, held in holders if held]
    assert len(with_secret) >= 3, (
        f"no {registry_name} stored the secret, so 'it is not rendered' is vacuous. "
        f"Constructed: {[n for n, _ in holders]}"
    )


def test_a_connector_that_buries_a_credential_in_a_dict_is_covered():
    """The specific shape the neighbouring static scan cannot see.

    It reads dataclass field *names*; a connector holding `{'password': ...}` inside a
    `_conn_kwargs` dict has no field called `password` and passes that scan while rendering
    the secret. Naming one such connector here means a refactor that stops burying it -- or
    stops constructing it in this sweep -- fails loudly instead of quietly narrowing what is
    checked.
    """
    built = dict(_built(SINKS))
    assert "redis" in built, "the redis sink no longer constructs, so this case is untested"
    held = _holds_the_secret(built["redis"])
    assert held, "the redis sink did not store the poisoned password"
    assert any("conn" in attribute for attribute in held), (
        f"expected the credential inside a connection dict, found it in {held}"
    )
    assert SECRET not in repr(built["redis"])
