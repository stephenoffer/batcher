"""Which expressions in a script are Batcher objects, and which Batcher receiver each one is.

A rename is only safe on the right receiver. `ds.groupby(...)` becomes `ds.group_by(...)` when
`ds` is a Batcher `Dataset` and must stay put when it is a pandas frame, and the two sit side
by side in the same files, often on the same line of a differential test. So before any rule
fires, this module decides what each expression *is*, using nothing but the source and the
generated `returns.toml`: no import of the engine, and no execution of the script.

The inference is deliberately conservative and flow-insensitive:

* **Seeds.** The `batcher` module under whatever alias it was imported as, and names imported
  from it (`from batcher import col`).
* **Chains.** An attribute of a receiver is looked up in `returns.toml`; calling it, or using
  it as the base of another attribute, yields the receiver it returns. `ds["x"]` on a
  `Dataset` and any arithmetic or comparison touching an `Expr` are `Expr`.
* **Bindings.** A name assigned exactly one receiver within a scope has that receiver
  everywhere in the scope; a name assigned two different ones has none. Module-level bindings
  are visible inside functions unless shadowed.
* **Parameters.** An annotation naming a receiver class, or the name of a same-file function
  whose every `return` is one receiver (the pytest fixture pattern), types a parameter.

Anything else is unknown, and an unknown receiver is never rewritten. The rules report those
sites instead, so the residue is visible rather than guessed at.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

from batcher._internal.optional import require

cst = require("libcst", feature="batcher.migrate", provides="libcst", extra="migrate")

__all__ = ["ClassRef", "Inference", "Member", "ReceiverScope", "infer_module"]

_EXPR_RECEIVERS = frozenset({"Expr", "AggExpr", "WindowExpr"})
# Batcher classes, by the name they are defined and imported under, and the receiver an
# instance of each is. A class name only counts when the class came from a `batcher` module
# (or is defined in the file): `pl.Expr` and `ray.data.Dataset` must never match.
_CLASS_RECEIVERS = {
    "Dataset": "Dataset",
    "GroupBy": "GroupBy",
    "MultiLevelGroupBy": "MultiLevelGroupBy",
    "Expr": "Expr",
    "AggExpr": "AggExpr",
    "WindowExpr": "WindowExpr",
    "CaseBuilder": "CaseBuilder",
    "Selector": "Selector",
    "Session": "Session",
    "StreamingQuery": "StreamingQuery",
    "MergeBuilder": "MergeBuilder",
    "Writer": "Dataset.write",
    "Reader": "bt.read",
    "DatasetML": "Dataset.ml",
    "DatasetMeta": "Dataset.meta",
    "_StrNamespace": "Expr.str",
    "_DtNamespace": "Expr.dt",
    "_ListNamespace": "Expr.list",
    "_StructNamespace": "Expr.struct",
    "_JsonNamespace": "Expr.json",
    "_MapNamespace": "Expr.map",
    "_ImageNamespace": "Expr.image",
    "_AudioNamespace": "Expr.audio",
    "_VideoNamespace": "Expr.video",
    "_SeqNamespace": "Expr.seq",
}


@dataclass(frozen=True)
class ClassRef:
    """A Batcher class itself (not an instance), bound by an import or a definition."""

    receiver: str


@dataclass(frozen=True)
class Member:
    """An attribute of a receiver that has not been called or dereferenced yet."""

    receiver: str
    name: str


Value = str | Member | ClassRef | None


@dataclass
class ReceiverScope:
    """Name bindings for one module or function body."""

    parent: ReceiverScope | None = None
    names: dict[str, Value] = field(default_factory=dict)
    conflicted: set[str] = field(default_factory=set)

    def bind(self, name: str, value: Value) -> None:
        """Record one assignment; two different receivers for a name make it unknown.

        Args:
            name: The bound name.
            value: The receiver (or member) the assigned expression evaluates to.
        """
        if name in self.conflicted:
            return
        if name in self.names and self.names[name] != value:
            self.conflicted.add(name)
            self.names.pop(name)
            return
        self.names[name] = value

    def lookup(self, name: str) -> Value:
        """Resolve a name through this scope and its parents.

        Args:
            name: The name to look up.

        Returns:
            The receiver or member, or `None` when unknown.
        """
        if name in self.conflicted:
            return None
        if name in self.names:
            return self.names[name]
        return self.parent.lookup(name) if self.parent else None


class Inference:
    """Evaluate expressions to receivers against one scope and the returns table."""

    def __init__(self, returns: dict[str, dict[str, str]], scope: ReceiverScope) -> None:
        self.returns = returns
        self.scope = scope

    def member_type(self, receiver: str, name: str) -> str | None:
        """The receiver a member returns, looking through `Expr` for aggregate/window nodes.

        Args:
            receiver: The receiver the member is accessed on.
            name: The member name.

        Returns:
            The returned receiver, or `None`.
        """
        found = self.returns.get(receiver, {}).get(name)
        if found is None and receiver in _EXPR_RECEIVERS:
            found = self.returns.get("Expr", {}).get(name)
        return found

    def value(self, node: cst.BaseExpression) -> Value:
        """Evaluate `node` to a receiver, a pending member, or `None`.

        Args:
            node: Any expression.

        Returns:
            What the expression is, as far as the source shows.
        """
        if isinstance(node, cst.Name):
            return self.scope.lookup(node.value)
        if isinstance(node, cst.Attribute):
            base = self.receiver(node.value)
            return Member(base, node.attr.value) if base else None
        if isinstance(node, cst.Call):
            func = self.value(node.func)
            if isinstance(func, Member):
                result = self.member_type(func.receiver, func.name)
                # `bt.read` is a property holding a callable namespace: calling the member
                # calls the namespace, not a method that returns it.
                called = self.returns.get(result or "", {}).get("__call__")
                return called if called and "." in (result or "") else result
            if isinstance(func, str):
                return self.returns.get(func, {}).get("__call__")
            return None
        if isinstance(node, cst.Subscript):
            base = self.receiver(node.value)
            return "Expr" if base == "Dataset" else base if base in _EXPR_RECEIVERS else None
        if isinstance(node, (cst.BinaryOperation, cst.Comparison, cst.UnaryOperation)):
            parts = _operands(node)
            return "Expr" if any(self.receiver(p) in _EXPR_RECEIVERS for p in parts) else None
        return None

    def receiver(self, node: cst.BaseExpression) -> str | None:
        """Evaluate `node` and dereference a pending member (a property access).

        Args:
            node: Any expression.

        Returns:
            The receiver, or `None`.
        """
        found = self.value(node)
        if isinstance(found, Member):
            return self.member_type(found.receiver, found.name)
        return found if isinstance(found, str) else None


def _operands(node: cst.BaseExpression) -> Iterator[cst.BaseExpression]:
    if isinstance(node, cst.BinaryOperation):
        yield node.left
        yield node.right
    elif isinstance(node, cst.UnaryOperation):
        yield node.expression
    elif isinstance(node, cst.Comparison):
        yield node.left
        yield from (c.comparator for c in node.comparisons)


def _dotted(node: cst.BaseExpression) -> str | None:
    if isinstance(node, cst.Name):
        return node.value
    if isinstance(node, cst.Attribute):
        head = _dotted(node.value)
        return f"{head}.{node.attr.value}" if head else None
    return None


def _module_statements(module: cst.Module) -> Iterator[cst.BaseSmallStatement]:
    """Every small statement at module level, including inside `if TYPE_CHECKING:`/`try`."""
    for stmt in _walk_statements(module):  # type: ignore[arg-type]
        yield from stmt.body


def seed_imports(module: cst.Module, scope: ReceiverScope) -> None:
    """Bind the `batcher` module aliases, and the Batcher names imported into the file.

    `import batcher as bt` binds the alias. `from batcher import col` binds a pending member
    of `bt`. A receiver class imported from any `batcher` module (`from batcher.api.dataset
    import Dataset`) binds a `ClassRef`, which is what makes a bare `Dataset` annotation
    trustworthy. A class *defined* in the file under a receiver name binds one too.

    Args:
        module: The parsed script.
        scope: The module scope to bind into.
    """
    for small in _module_statements(module):
        if isinstance(small, cst.Import):
            for alias in small.names:
                if _dotted(alias.name) == "batcher":
                    bound = alias.asname.name if alias.asname else alias.name
                    if isinstance(bound, cst.Name):
                        scope.bind(bound.value, "bt")
        elif isinstance(small, cst.ImportFrom) and small.module is not None:
            source = _dotted(small.module) or ""
            if not (source == "batcher" or source.startswith("batcher.")):
                continue
            if isinstance(small.names, cst.ImportStar):
                continue
            for alias in small.names:
                if not isinstance(alias.name, cst.Name):
                    continue
                name = alias.name.value
                bound = alias.asname.name.value if alias.asname else name  # type: ignore[union-attr]
                if name in _CLASS_RECEIVERS:
                    scope.bind(bound, ClassRef(_CLASS_RECEIVERS[name]))
                elif source == "batcher":
                    scope.bind(bound, Member("bt", name))
    for stmt in module.body:
        if isinstance(stmt, cst.ClassDef) and stmt.name.value in _CLASS_RECEIVERS:
            scope.bind(stmt.name.value, ClassRef(_CLASS_RECEIVERS[stmt.name.value]))


def _annotation_receiver(annotation: cst.Annotation | None, scope: ReceiverScope) -> str | None:
    """The receiver an annotation names, when it provably names a Batcher class."""
    if annotation is None:
        return None
    return _type_expression_receiver(annotation.annotation, scope)


def _type_expression_receiver(node: cst.BaseExpression, scope: ReceiverScope) -> str | None:
    if isinstance(node, cst.BinaryOperation):  # `Dataset | None`
        left = _type_expression_receiver(node.left, scope)
        return left or _type_expression_receiver(node.right, scope)
    if isinstance(node, cst.SimpleString):
        try:
            node = cst.parse_expression(str(node.evaluated_value))
        except Exception:  # an unparseable string annotation types nothing
            return None
    if isinstance(node, cst.Name):
        bound = scope.lookup(node.value)
        return bound.receiver if isinstance(bound, ClassRef) else None
    if isinstance(node, cst.Attribute) and node.attr.value in _CLASS_RECEIVERS:
        head = scope.lookup(node.value.value) if isinstance(node.value, cst.Name) else None
        return _CLASS_RECEIVERS[node.attr.value] if head == "bt" else None
    return None


def _bind_body(body: cst.BaseSuite, inference: Inference) -> None:
    """Bind every simple `name = expr` in a body, iterating until the bindings settle.

    Each pass evaluates every assignment against the previous pass's bindings, so a chain
    (`a = bt.from_pydict(...)`, `b = a.filter(...)`) resolves one link per pass. A name that
    receives two different values, or any value the inference cannot type (a pandas frame),
    is unknown in this scope, and that shadows whatever an enclosing scope says about it.
    """
    assignments: list[tuple[str, cst.BaseExpression]] = []
    for stmt in _walk_statements(body):
        for small in stmt.body:
            if isinstance(small, cst.Assign):
                targets = [t.target for t in small.targets]
            elif isinstance(small, cst.AnnAssign) and small.value is not None:
                targets = [small.target]
            else:
                continue
            value = small.value
            assignments.extend((t.value, value) for t in targets if isinstance(t, cst.Name))
    scope = inference.scope
    fixed = dict(scope.names)  # parameters typed before the body was read
    for _ in range(10):
        seen: dict[str, set[Value]] = {n: {v} for n, v in fixed.items()}
        for name, value in assignments:
            seen.setdefault(name, set()).add(inference.receiver(value))
        names = {n: next(iter(v)) for n, v in seen.items() if len(v) == 1 and None not in v}
        conflicted = {n for n, v in seen.items() if len(v) > 1 or None in v}
        if names == scope.names and conflicted == scope.conflicted:
            return
        scope.names, scope.conflicted = names, conflicted


def _walk_statements(body: cst.BaseSuite) -> Iterator[cst.SimpleStatementLine]:
    if isinstance(body, cst.SimpleStatementSuite):
        return
    for stmt in body.body:
        if isinstance(stmt, cst.SimpleStatementLine):
            yield stmt
        elif isinstance(stmt, (cst.If, cst.For, cst.While, cst.With, cst.Try)):
            yield from _walk_statements(stmt.body)


def _returns_of(func: cst.FunctionDef, inference: Inference) -> str | None:
    found: set[str | None] = set()

    class _Returns(cst.CSTVisitor):
        def visit_FunctionDef(self, node: cst.FunctionDef) -> bool:
            return node is func

        def visit_Return(self, node: cst.Return) -> None:
            found.add(inference.receiver(node.value) if node.value else None)

    func.visit(_Returns())
    return found.pop() if len(found) == 1 else None


def infer_module(
    module: cst.Module, returns: dict[str, dict[str, str]]
) -> dict[cst.FunctionDef | cst.Module, Inference]:
    """Build one `Inference` per scope: the module, and every function in it.

    Args:
        module: The parsed script.
        returns: The generated returns table.

    Returns:
        A map from each scope node to the inference that evaluates expressions inside it.
    """
    top = ReceiverScope()
    seed_imports(module, top)
    top_inference = Inference(returns, top)
    _bind_body(module, top_inference)  # type: ignore[arg-type]
    functions = _all_functions(module)
    # Fixtures and helpers: a same-file function returning one receiver types parameters
    # that share its name.
    typed_functions: dict[str, str] = {}
    for func, owner in functions:
        inference = Inference(returns, ReceiverScope(parent=top))
        _bind_params(func, inference, typed_functions, owner)
        _bind_body(func.body, inference)
        kind = _returns_of(func, inference)
        if kind is not None and owner is None:
            typed_functions[func.name.value] = kind
    out: dict[cst.FunctionDef | cst.Module, Inference] = {module: top_inference}
    for func, owner in functions:
        inference = Inference(returns, ReceiverScope(parent=top))
        _bind_params(func, inference, typed_functions, owner)
        _bind_body(func.body, inference)
        out[func] = inference
    return out


def _bind_params(
    func: cst.FunctionDef,
    inference: Inference,
    fixtures: dict[str, str],
    owner: str | None,
) -> None:
    params = func.params
    positional = [*params.posonly_params, *params.params]
    for i, param in enumerate([*positional, *params.kwonly_params]):
        name = param.name.value
        typed = _annotation_receiver(param.annotation, inference.scope) or fixtures.get(name)
        if typed is None and i == 0 and owner is not None and name == "self":
            typed = owner
        if typed is not None:
            inference.scope.bind(name, typed)


def _all_functions(node: cst.CSTNode) -> list[tuple[cst.FunctionDef, str | None]]:
    """Every function, with the receiver its `self` is when it is a Batcher class's method."""
    found: list[tuple[cst.FunctionDef, str | None]] = []
    classes: list[str | None] = []

    class _Collect(cst.CSTVisitor):
        def visit_ClassDef(self, cls: cst.ClassDef) -> None:
            classes.append(_CLASS_RECEIVERS.get(cls.name.value))

        def leave_ClassDef(self, _original_node: cst.ClassDef) -> None:
            classes.pop()

        def visit_FunctionDef(self, fn: cst.FunctionDef) -> None:
            found.append((fn, classes[-1] if classes else None))

    node.visit(_Collect())
    return found
