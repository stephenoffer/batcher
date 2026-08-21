"""Generative AI table functions: ``AI_GENERATE`` / ``AI_CLASSIFY`` / ``AI_EXTRACT``.

`table_functions` gives SQL the *traditional* model — ``ML_PREDICT`` scores a fitted
scikit-learn, XGBoost, LightGBM, CatBoost or ONNX model over a relation. These are the
generative half: a language model asked to write, label, or pull structure out of a text
column, which a warehouse spells ``ai_query`` (Databricks), ``AI_COMPLETE`` (Snowflake) or
``AI.GENERATE`` (BigQuery).

**Why these are relations rather than scalar functions.** Every one of those warehouses
writes its AI call in the ``SELECT`` list, and this does not. The reason is the same one
that makes `ML_PREDICT` a table function, stated in `translator`'s model catalog: a model
"is not callable from an expression: it is scored over a whole relation, in a stage the
engine schedules". A Batcher scalar function lowers to a `bc_expr::Expr` evaluated per row
in Rust, and a language-model call is neither expressible there nor wanted per row — the
whole point of the inference path is that an engine loads once per worker and sees a batch
at a time. Putting the call in ``FROM`` says that honestly. It is also why the option that
matters, `prompt_column`, is named rather than positional: the relation is the input, and
the column is a setting on it.

All three lower to the matching `Dataset.ml` call, so there is no second inference path
here — only a second way to ask for it, exactly as `ML_PREDICT` is for scoring.
"""

from __future__ import annotations

from typing import Any

from sqlglot import expressions as exp

from batcher._internal.errors import PlanError
from batcher._sql.parser.table_functions import relation_argument
from batcher.api.dataset import Dataset

__all__ = ["ai_table", "is_ai_source"]

#: The AI calls sqlglot parses into a node of their own, mapped to the `Dataset.ml` method
#: each lowers to. sqlglot grew native nodes for these because several warehouses ship them;
#: matching on the type is what makes ``AI_GENERATE`` work without a dialect switch.
_AI_NODES: dict[type, str] = {
    exp.AIGenerate: "generate",
}

#: The rest arrive as a plain function call. ``ai_query`` and ``ai_complete`` are Databricks'
#: and Snowflake's names for free-form generation and are accepted as aliases, so a ported
#: query runs without being reworded first.
_AI_NAMES: dict[str, str] = {
    "ai_generate": "generate",
    "ai_query": "generate",
    "ai_complete": "generate",
    "ai_extract": "extract",
}

#: AI calls sqlglot knows but this does not translate, and where the capability actually
#: lives. Without these a query reaches the table lookup and fails with ``unknown table ''``
#: — the node carries no table name — which says nothing about what went wrong.
_AI_ELSEWHERE: dict[type, str] = {
    # sqlglot models `AI_CLASSIFY` on Snowflake's grammar, which is fixed at
    # ``(input, categories [, config])`` — three arguments, where a relational form needs
    # four: the relation, the engine, the text column and the labels. Rather than fold two
    # of them into a config object and give this one function a shape none of the others
    # have, it is routed to the spelling that does fit.
    exp.AIClassify: (
        "AI_EXTRACT(t, engine, prompt_column => 'col', schema => ['label string']) for a "
        "single labelled column, or ds.ml.classify(engine, labels=[...]) on the Dataset"
    ),
    exp.AIEmbed: "ds.ml.embed(model, ...), which takes an encoder rather than a text engine",
    exp.AISimilarity: "ds.ml.similarity_to(...) or ds.ml.nearest_neighbors(...) over embeddings",
    exp.AIAgg: "ds.ml.generate(...) over a grouped relation",
    exp.AISummarizeAgg: "ds.ml.generate(...) over a grouped relation",
    exp.AIForecast: "the batcher.ml.timeseries forecasting helpers",
}

#: How to name each kind in an error, since a native node carries no function name.
_DISPLAY: dict[str, str] = {
    "generate": "AI_GENERATE",
    "classify": "AI_CLASSIFY",
    "extract": "AI_EXTRACT",
}

#: Settings each function forwards, and nothing else. As with `ML_PREDICT`, execution choices
#: (batch size, GPU count, retry budget) are deliberately absent: they belong to the pipeline
#: running the query, not to the query text. What is here is what changes the answer.
_COMMON_OPTIONS = frozenset({"prompt_column", "template", "output_column"})
_KIND_OPTIONS: dict[str, frozenset[str]] = {
    "generate": _COMMON_OPTIONS,
    "classify": _COMMON_OPTIONS | {"labels"},
    "extract": (_COMMON_OPTIONS - {"output_column"}) | {"schema"},
}

#: Settings without which the call has no meaning, so they are required rather than defaulted.
_REQUIRED: dict[str, tuple[str, ...]] = {
    "generate": ("prompt_column",),
    "classify": ("prompt_column", "labels"),
    "extract": ("prompt_column", "schema"),
}


def is_ai_source(node) -> bool:
    """Whether this FROM item is an AI table function — ``AI_GENERATE(...)`` and friends.

    True for the ones sqlglot gives their own node type, for the ones that arrive as an
    ordinary call, and for the AI calls translated elsewhere — those last are claimed here so
    `ai_table` can say where they live rather than letting them fall through to the table
    lookup and fail as ``unknown table ''``.
    """
    if not isinstance(node, exp.Table):
        return False
    inner = node.this
    if type(inner) in _AI_NODES or type(inner) in _AI_ELSEWHERE:
        return True
    return isinstance(inner, exp.Anonymous) and inner.name.lower() in _AI_NAMES


def ai_table(tr, node) -> Dataset:
    """Build the relation an AI table function denotes.

    Reads ``FROM AI_CLASSIFY(t, grader, prompt_column => 'body', labels => ['a', 'b'])`` as
    the relation `t` with the label column appended, and lowers it to
    `Dataset.ml.classify`. `AI_GENERATE` and `AI_EXTRACT` are the same shape onto
    `Dataset.ml.generate` and `Dataset.ml.extract`.

    Args:
        tr: The translator, holding the table and engine registries.
        node: The `exp.Table` wrapping the call.

    Returns:
        The input relation with the generated column(s) appended.

    Raises:
        PlanError: For an unknown engine or table, a wrong argument count, an unknown or
            missing setting, or a setting of the wrong shape.
    """
    inner = node.this
    elsewhere = _AI_ELSEWHERE.get(type(inner))
    if elsewhere is not None:
        raise PlanError(
            f"{_node_name(inner)} is not translated to SQL here; the capability is "
            f"{elsewhere}. Run that stage on the Dataset and register the result as a table"
        )
    kind = _AI_NODES.get(type(inner)) or _AI_NAMES[inner.name.lower()]
    called = inner.name.upper() if inner.name else _DISPLAY[kind]
    args = list(inner.expressions)
    positional = [a for a in args if not isinstance(a, exp.Kwarg)]
    if len(positional) != 2:
        raise PlanError(
            f"{called}(relation, engine, setting => value, ...) takes exactly two "
            f"positional arguments — the relation and the engine — got {len(positional)}. "
            f"The text column is a named setting: prompt_column => 'your_column'"
        )
    source = relation_argument(tr, positional[0], called)
    engine = _engine(tr, positional[1])
    options = _options(args, kind, called)
    return _generated(source, engine, kind, options)


def _engine(tr, node) -> Any:
    """The engine a call names, resolved against the session's engine catalog.

    Unlike `ML_PREDICT`'s model there is no path spelling, because an engine is not a file:
    it is a callable holding an endpoint, credentials and sampling settings. Naming one in
    SQL text would mean putting an API key there.
    """
    if isinstance(node, exp.Literal) and node.is_string:
        raise PlanError(
            f"the engine argument must be a registered engine name, not the quoted string "
            f"{node.this!r}. An engine carries an endpoint and credentials, so it is built in "
            f"Python and registered: Session.register_engine('name', bt.ml.http_engine(...))"
        )
    engine_name = node.name if isinstance(node, (exp.Column, exp.Table, exp.Identifier)) else None
    if not engine_name:
        raise PlanError(f"the engine argument must be a registered engine name, got {node.sql()!r}")
    engines = getattr(tr, "_engines", {})
    if engine_name not in engines:
        raise PlanError(
            f"unknown engine {engine_name!r}; registered: {sorted(engines)}. Register one with "
            f"Session.register_engine(name, engine), where engine is an EngineFactory such as "
            f"bt.ml.http_engine(...), bt.ml.vllm_engine(...) or bt.ml.anthropic_engine(...)"
        )
    return engines[engine_name]


def _options(args: list, kind: str, called: str) -> dict[str, Any]:
    """The validated ``name => value`` settings of one AI call."""
    allowed = _KIND_OPTIONS[kind]
    out: dict[str, Any] = {}
    for kw in args:
        if not isinstance(kw, exp.Kwarg):
            continue
        key = kw.this.name.lower()
        if key not in allowed:
            raise PlanError(
                f"unknown setting {kw.this.name!r} for {called}; supported: {sorted(allowed)}"
            )
        out[key] = _value(kw.expression, key, called)
    missing = [r for r in _REQUIRED[kind] if r not in out]
    if missing:
        raise PlanError(
            f"{called} needs {' and '.join(repr(m) for m in missing)}; write it as a named "
            f"setting, e.g. {missing[0]} => "
            f"{'[...]' if missing[0] in ('labels', 'schema') else chr(39) + 'column' + chr(39)}"
        )
    return out


def _value(node, key: str, called: str) -> Any:
    """One setting's value: a list for ``labels``/``schema``, a string for the rest."""
    if key in ("labels", "schema"):
        if not isinstance(node, (exp.Array, exp.Tuple)):
            example = "['positive', 'negative']" if key == "labels" else "['sentiment string']"
            raise PlanError(f"{called}'s {key!r} must be a list, e.g. {key} => {example}")
        items = [_string(item, key, called) for item in node.expressions]
        if not items:
            raise PlanError(f"{called}'s {key!r} must not be empty")
        return _schema(items, called) if key == "schema" else items
    return _string(node, key, called)


def _string(node, key: str, called: str) -> str:
    """A string literal, or a rejection naming the setting that wanted one."""
    if isinstance(node, exp.Literal) and node.is_string:
        return node.this
    raise PlanError(f"{called}'s {key!r} must be a quoted string, got {node.sql()!r}")


def _schema(items: list[str], called: str) -> dict[str, str]:
    """``['sentiment string', 'score float64']`` as the ``{name: type}`` `extract` wants.

    Written as a column definition list rather than a nested structure because that is the
    shape SQL already has for "these names with these types", and a `STRUCT` argument would
    be a second spelling of it.
    """
    out: dict[str, str] = {}
    for item in items:
        parts = item.split()
        if len(parts) != 2:
            raise PlanError(
                f"each field in {called}'s 'schema' is written '<name> <type>', e.g. "
                f"'sentiment string' — got {item!r}"
            )
        out[parts[0]] = parts[1]
    return out


def _generated(source: Dataset, engine: Any, kind: str, options: dict[str, Any]) -> Dataset:
    """`source` with the generated column(s) appended — the one place every spelling lands."""
    if kind == "generate":
        return source.ml.generate(engine, **options)
    if kind == "classify":
        return source.ml.classify(engine, **options)
    return source.ml.extract(engine, **options)


def _node_name(inner) -> str:
    """The SQL spelling of an AI node, which carries no function name of its own.

    sqlglot names the classes in CamelCase (``AIClassify``), and quoting that back at someone
    who wrote ``AI_CLASSIFY`` makes them look for a different function than the one they used.
    """
    if inner.name:
        return inner.name.upper()
    letters = type(inner).__name__.removeprefix("AI")
    tail = "".join(f"_{c}" if c.isupper() and i else c for i, c in enumerate(letters))
    return f"AI_{tail.upper()}"
