"""Reading a conversation column — the shape a chat log and an SFT dataset both arrive in.

Fine-tuning data, chat logs, and agent traces all land as a list of `{role, content}` structs
per row. Every question you want to ask of that column is a per-row loop in the obvious
implementation: how many turns, did the assistant actually answer, what was the last thing the
user said, render the whole thing as a prompt.

None of it needs to be. A list of structs is a columnar value, and the `.list` higher-order
functions reach inside it, so each of these is one expression the engine evaluates over a whole
batch. A million conversations is one scan.

The field names are parameters rather than constants because the convention is not universal —
OpenAI-style logs use `role`/`content`, other exports use `speaker`/`text` — and renaming a
struct field to fit a hard-coded assumption is a materialization nobody should have to pay for.
"""

from __future__ import annotations

from batcher._internal.errors import PlanError
from batcher.plan.expr_ir.core import Expr, IntoExpr, Lit
from batcher.plan.functions.aggregate import _as_column
from batcher.plan.functions.collection import element
from batcher.plan.functions.string import concat

__all__ = [
    "conversation_turns",
    "ends_with_role",
    "last_message",
    "render_messages",
]


def _messages(messages: IntoExpr) -> Expr:
    """The message-list column, as an expression."""
    return _as_column(messages)


def _of_role(messages: IntoExpr, role: str | None, role_field: str) -> Expr:
    """The message list, narrowed to one role when asked."""
    column = _messages(messages)
    if role is None:
        return column
    return column.list.filter(element().struct.field(role_field) == Lit(role))


def _require_role(role: str | None, func: str) -> None:
    """Reject an empty role, which would silently match nothing."""
    if role is not None and not role.strip():
        raise PlanError(f"{func}: role must be a non-empty string, got {role!r}")


def conversation_turns(
    messages: IntoExpr,
    role: str | None = None,
    *,
    role_field: str = "role",
) -> Expr:
    """The number of messages in each conversation, optionally counting only one role.

    Turn count is the first thing to look at in a conversation corpus and the cheapest filter
    on it. Single-turn rows are a different training task from long multi-turn ones, and a
    corpus that mixes them without saying so trains a model that is good at neither.

    Counting one role is the more useful form. ``role="assistant"`` is how many replies the
    conversation actually contains, which is not the same as its length: a log full of user
    messages with no answers is a collection failure, not a short conversation.

    Args:
        messages: The conversation column, a list of `{role, content}` structs.
        role: Count only messages with this role. Omit to count every message.
        role_field: The struct field holding the role.

    Returns:
        An Int64 expression: the per-row message count.

    Raises:
        PlanError: If `role` is given but empty.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict(
            ...     {
            ...         "msgs": [
            ...             [
            ...                 {"role": "user", "content": "hi"},
            ...                 {"role": "assistant", "content": "hello"},
            ...                 {"role": "user", "content": "again"},
            ...             ]
            ...         ]
            ...     }
            ... )
            >>> ds.select(
            ...     n=bt.conversation_turns("msgs"),
            ...     replies=bt.conversation_turns("msgs", "assistant"),
            ... ).to_pydict()
            {'n': [3], 'replies': [1]}
    """
    _require_role(role, "conversation_turns")
    return _of_role(messages, role, role_field).list.len()


def last_message(
    messages: IntoExpr,
    role: str | None = None,
    *,
    role_field: str = "role",
    content_field: str = "content",
) -> Expr:
    """The content of the last message, optionally the last from one role.

    The two forms answer different questions. Without a role it is how the conversation ended,
    which is what you filter on to find truncated logs. With ``role="user"`` it is the request
    the final answer was responding to, which is the query half of an evaluation pair; with
    ``role="assistant"`` it is the answer, which is what a generation metric scores.

    A conversation with no message of that role yields null rather than an empty string, so the
    rows that never had one are countable instead of scoring as an empty answer.

    Args:
        messages: The conversation column, a list of `{role, content}` structs.
        role: Take the last message with this role. Omit to take the last message of any role.
        role_field: The struct field holding the role.
        content_field: The struct field holding the text.

    Returns:
        A string expression: the message's content, or null if there is none.

    Raises:
        PlanError: If `role` is given but empty.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict(
            ...     {
            ...         "msgs": [
            ...             [
            ...                 {"role": "user", "content": "what is 2+2?"},
            ...                 {"role": "assistant", "content": "4"},
            ...                 {"role": "user", "content": "and 3+3?"},
            ...             ]
            ...         ]
            ...     }
            ... )
            >>> ds.select(
            ...     ended=bt.last_message("msgs"),
            ...     answer=bt.last_message("msgs", "assistant"),
            ... ).to_pydict()
            {'ended': ['and 3+3?'], 'answer': ['4']}
    """
    _require_role(role, "last_message")
    return _of_role(messages, role, role_field).list.last().struct.field(content_field)


def ends_with_role(
    messages: IntoExpr,
    role: str = "assistant",
    *,
    role_field: str = "role",
) -> Expr:
    """True where the conversation's final message came from `role`.

    The completeness check for a fine-tuning corpus. A conversation ending on a user turn was
    cut off — the export stopped mid-exchange, or the assistant's reply failed and was never
    written — and training on it teaches the model to stop where an answer should start. The
    rows are indistinguishable from complete ones by every other measure, including length.

    Args:
        messages: The conversation column, a list of `{role, content}` structs.
        role: The role the conversation must end on.
        role_field: The struct field holding the role.

    Returns:
        A Boolean expression, true where the last message has that role; null for a null or
        empty conversation.

    Raises:
        PlanError: If `role` is empty.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict(
            ...     {
            ...         "msgs": [
            ...             [
            ...                 {"role": "user", "content": "hi"},
            ...                 {"role": "assistant", "content": "hello"},
            ...             ],
            ...             [{"role": "user", "content": "hi"}],
            ...         ]
            ...     }
            ... )
            >>> ds.select(complete=bt.ends_with_role("msgs")).to_pydict()
            {'complete': [True, False]}
    """
    if not role.strip():
        raise PlanError(f"ends_with_role: role must be a non-empty string, got {role!r}")
    return _messages(messages).list.last().struct.field(role_field) == Lit(role)


def render_messages(
    messages: IntoExpr,
    *,
    role_field: str = "role",
    content_field: str = "content",
    separator: str = "\n",
    role_suffix: str = ": ",
) -> Expr:
    """Render a conversation into one string, one message per line, prefixed by its role.

    The plain-text form a completions endpoint takes, and the readable form for a diff, a spot
    check, or a text-metric pass over a whole conversation rather than its final answer.
    `chatml_prompt` is the alternative when a model expects its own control tokens; this is the
    generic rendering, and it is what you want when the reader is a human or a lexical metric.

    Args:
        messages: The conversation column, a list of `{role, content}` structs.
        role_field: The struct field holding the role.
        content_field: The struct field holding the text.
        separator: The text placed between messages.
        role_suffix: The text between a role and its content.

    Returns:
        A string expression holding the rendered conversation.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict(
            ...     {
            ...         "msgs": [
            ...             [
            ...                 {"role": "user", "content": "hi"},
            ...                 {"role": "assistant", "content": "hello"},
            ...             ]
            ...         ]
            ...     }
            ... )
            >>> print(ds.select(t=bt.render_messages("msgs")).to_pydict()["t"][0])
            user: hi
            assistant: hello
    """
    rendered = _messages(messages).list.transform(
        concat(
            element().struct.field(role_field),
            Lit(role_suffix),
            element().struct.field(content_field),
        )
    )
    return rendered.list.join(separator)
