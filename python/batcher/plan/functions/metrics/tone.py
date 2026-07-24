"""Tone and style metrics — how an LLM sounds, measured as corpus rates.

A model that starts over-hedging, over-exclaiming, or answering a question with a question is
drifting in voice, and the drift is invisible row by row. These measure that voice directly, as a
single mergeable rate over the corpus: the share of outputs that end in a question, end in an
exclamation, hedge, speak in the first person, turn polite, or contain a watched phrase. A jump
between two runs flags a tone regression before a human reads a hundred samples. Each is one
mergeable aggregate over the string primitives, so it stays identical single-node and distributed
and composes inside `group_by`.
"""

from __future__ import annotations

from collections.abc import Iterable

from batcher.plan.expr_ir.constructors import lit
from batcher.plan.expr_ir.core import Expr, IntoExpr
from batcher.plan.functions.aggregate import _as_column, count_if

__all__ = [
    "contains_phrase_rate",
    "exclamation_rate",
    "first_person_rate",
    "hedge_rate",
    "politeness_rate",
    "question_rate",
]

_HEDGE = (
    r"(?i)\b(maybe|perhaps|possibly|might|could be|i think|i believe|"
    r"not sure|it seems|probably)\b"
)
_FIRST_PERSON = r"(?i)\b(i|i'm|i am|my|me|we|our)\b"
_POLITENESS = r"(?i)\b(please|thank you|thanks|sorry|apologize|you're welcome)\b"


def question_rate(text: IntoExpr) -> Expr:
    """The fraction of generations that end in a question mark — the question-back rate.

    On an answer task a question is a deflection: the model asked something back instead of
    answering. A rising rate between runs says the model is punting on prompts it used to answer, so
    it surfaces evasive behavior as a single corpus number rather than one sample at a time.

    Args:
        text: The generated-text column (name or expression).

    Returns:
        The question rate over the corpus, in ``[0, 1]``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict(
            ...     {"o": ["What do you mean?", "The answer is 4.", "Maybe it works?"]}
            ... )
            >>> round(ds.agg(q=bt.question_rate("o")).to_pydict()["q"][0], 4)
            0.6667
    """
    return count_if(_as_column(text).str.is_question()) / count_if(lit(True))


def exclamation_rate(text: IntoExpr) -> Expr:
    """The fraction of generations that end in an exclamation mark — the enthusiasm rate.

    An output ending in ``!`` reads as over-excited or marketing-toned, and a jump in the rate is
    the signature of a model that has drifted breathless. It is the corpus number for that tone, so
    a prompt or checkpoint change that makes the model shout shows up before a reviewer notices.

    Args:
        text: The generated-text column (name or expression).

    Returns:
        The exclamation rate over the corpus, in ``[0, 1]``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"o": ["Wow, amazing!", "It is fine.", "Great job!"]})
            >>> round(ds.agg(e=bt.exclamation_rate("o")).to_pydict()["e"][0], 4)
            0.6667
    """
    return count_if(_as_column(text).str.is_exclamation()) / count_if(lit(True))


def hedge_rate(text: IntoExpr) -> Expr:
    """The fraction of generations containing a hedging word — the uncertainty-tone rate.

    Words such as ``maybe``, ``perhaps``, or ``I think`` mark an output that is qualifying itself
    rather than committing to an answer. A rising rate says the model is growing evasive or
    under-confident across a run, which reads as tone even when the answers stay correct, so it is
    the corpus number for over-hedging. The match is case-insensitive.

    Args:
        text: The generated-text column (name or expression).

    Returns:
        The hedge rate over the corpus, in ``[0, 1]``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"o": ["Maybe it works?", "The answer is 4.", "I think so."]})
            >>> round(ds.agg(h=bt.hedge_rate("o")).to_pydict()["h"][0], 4)
            0.6667
    """
    return count_if(_as_column(text).str.regexp_matches(_HEDGE)) / count_if(lit(True))


def first_person_rate(text: IntoExpr) -> Expr:
    """The fraction of generations using first-person voice — the first-person rate.

    A pronoun such as ``I``, ``we``, or ``my`` marks an output that speaks as itself rather than
    answering impersonally, which for many assistant tasks is exactly the persona drift a team
    watches for. It is the corpus number for that voice, case-insensitive, so a checkpoint that
    starts narrating in the first person shows up as a rate rather than a hunch.

    Args:
        text: The generated-text column (name or expression).

    Returns:
        The first-person rate over the corpus, in ``[0, 1]``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict(
            ...     {"o": ["I can help", "The sky is blue", "We are ready", "my cat"]}
            ... )
            >>> ds.agg(f=bt.first_person_rate("o")).to_pydict()["f"][0]
            0.75
    """
    return count_if(_as_column(text).str.regexp_matches(_FIRST_PERSON)) / count_if(lit(True))


def politeness_rate(text: IntoExpr) -> Expr:
    """The fraction of generations with a politeness marker — the politeness rate.

    A marker such as ``please``, ``thank you``, or ``sorry`` marks an output that has turned
    deferential, and a rising rate is the signature of a model that over-apologizes or over-thanks
    instead of answering plainly. It is the corpus number for that tone, case-insensitive, so an
    over-polite drift is visible before it reaches a reviewer.

    Args:
        text: The generated-text column (name or expression).

    Returns:
        The politeness rate over the corpus, in ``[0, 1]``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict(
            ...     {"o": ["Please help me", "Thank you so much", "Just do it.", "No."]}
            ... )
            >>> ds.agg(p=bt.politeness_rate("o")).to_pydict()["p"][0]
            0.5
    """
    return count_if(_as_column(text).str.regexp_matches(_POLITENESS)) / count_if(lit(True))


def contains_phrase_rate(text: IntoExpr, phrases: Iterable[str]) -> Expr:
    """The fraction of generations containing any of the given literal phrases — a watchlist rate.

    A configurable tone or blocklist monitor: pass the phrases you want to track (a refusal
    boilerplate, a banned slogan, an over-used opener) and get the corpus share of outputs that
    contain any of them. The match is literal substring and case-sensitive, so ``"i cannot"`` does
    not match ``"I cannot"`` — list the casings you care about explicitly.

    Args:
        text: The generated-text column (name or expression).
        phrases: The literal phrases to watch for; an output counts if it contains any of them.

    Returns:
        The watchlist rate over the corpus, in ``[0, 1]``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict(
            ...     {"o": ["i cannot assist with that", "here you go", "i cannot do that"]}
            ... )
            >>> round(ds.agg(c=bt.contains_phrase_rate("o", ["i cannot"])).to_pydict()["c"][0], 4)
            0.6667
    """
    return count_if(_as_column(text).str.contains_any(phrases)) / count_if(lit(True))
