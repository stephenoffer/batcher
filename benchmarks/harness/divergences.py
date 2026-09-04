"""Known semantic differences between two engines — recorded, cited, and never hidden.

The correctness gate compares Batcher against a reference engine and calls any difference
``FAILED``. That is the right default and it is not the whole truth: **a divergence from
DuckDB is evidence about the pair, not about Batcher.** Four cases established that in a
single day of auditing, and they point in three different directions:

* TPC-H q6 — Polars and Daft fold ``0.06 + 0.01`` in IEEE double to the value one ulp below
  ``0.07`` and drop every ``l_discount = 0.07`` row. *They* are wrong; Batcher and DuckDB
  agree with the published TPC-H answer.
* ``var_samp`` over ``{2^53, 2^53+2}`` — Batcher returns 2.0, DuckDB returns 4.0, and the
  exact rational answer is 2.0. *The oracle* is wrong.
* ``SELECT * FROM a JOIN b USING (k)`` — Batcher puts the coalesced key first, per
  SQL:2016 §7.7 (and PostgreSQL); DuckDB keeps the key at its left-table index. *The oracle*
  is off-standard, and a positional column check recorded 49 failures against Batcher for
  being correct before this was understood.
* A tied ``ORDER BY`` feeding ``rank``/``row_number`` — which tied row gets which number is
  undefined in SQL, so two correct engines legitimately disagree. *Neither* is wrong.

Without somewhere to record these, each one costs the same rediscovery, and the pressure at
the end of it is to change Batcher to match a comparator's deviation. That is the failure
this module exists to prevent, and it is not hypothetical: ``compare``'s ``_ORACLE_PREFERENCE``
carries a scar from the same shape.

## This is a disclosure mechanism, not an escape hatch

Adding an entry makes a red row *visible and explained*. It does not make it green:

* the row is reported ``DIVERGENT``, which is its own status and never ``OK``;
* the ``b/<engine>`` ratio is still withheld, because two engines that computed different
  answers were not doing the same work and dividing their times asserts that they were;
* the reason and its citation print beneath the table on every run.

Every entry needs a `reason` naming *which side is right and why*, and a `citation` a reader
can check — a spec clause, a published answer set, or a measurement. An entry that says only
"these differ" is the thing this module is meant to stop, so :func:`_validate` refuses one.

**Never add an entry to make a failing change pass.** If Batcher's answer moved and you do
not know which engine is right, the entry is not the fix — finding out is. `CLAUDE.md` puts
it as "never weaken or delete a differential test to make a change pass", and this is the
same act one level up. The list is a ratchet in the same sense the layer-debt list is: it may
shrink, and it grows only with evidence attached.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["KNOWN_DIVERGENCES", "Divergence", "explain"]


@dataclass(frozen=True)
class Divergence:
    """One recorded semantic difference between two engines on a class of query.

    An entry must match on **three** axes — the case, the engine pair, and the shape of the
    difference itself. The third is what keeps this a disclosure mechanism: an entry keyed
    on the pair alone would silence every future Batcher-vs-DuckDB failure, which is the
    escape hatch this module's docstring promises it is not. The first draft of this file
    did exactly that, which is why the constraint is stated here rather than assumed.

    Attributes:
        case: Substring matched against the benchmark case name (``"tpch-q6"``), or ``""``
            when the difference is a property of the engines rather than of one query — in
            which case `signature` must be correspondingly specific.
        engine: The engine that is the odd one out — the one whose answer differs from the
            others'. Named this way rather than as a pair because the comparison is made
            against whichever oracle `_ORACLE_PREFERENCE` picked, which is not knowable
            when the entry is written: on TPC-H q6 Polars differs from Batcher *and* from
            DuckDB, and an entry keyed on ``("batcher", "polars")`` silently never fires
            because the oracle is DuckDB. That was this module's first bug.
        versus: The specific counterpart, when the difference is genuinely a property of
            one pair (Batcher against DuckDB's `USING` column order), or ``None`` when the
            engine differs from every correct implementation.
        signature: A substring that must appear in the mismatch message `rowsets_match`
            produced. This is what pins the entry to the difference it documents rather
            than to every difference those two engines might ever have.
        verdict: Which side the evidence favours — ``"batcher"``, ``"reference"``, or
            ``"undefined"`` when the query does not determine an answer.
        reason: What differs and why, naming the side that is right.
        citation: Where a reader can check it.
    """

    case: str
    engine: str
    versus: str | None
    signature: str
    verdict: str
    reason: str
    citation: str

    def matches(self, case: str, ref: str, other: str, message: str) -> bool:
        """Whether this entry covers this specific mismatch.

        All three axes must agree. `message` is the diff `rowsets_match` returned, and
        requiring `signature` to appear in it is what stops a recorded difference from
        absorbing an unrelated one involving the same engine.
        """
        pair = {ref, other}
        if self.engine not in pair:
            return False
        if self.versus is not None and self.versus not in pair:
            return False
        return (not self.case or self.case in case) and self.signature in message


_VERDICTS = ("batcher", "reference", "undefined")


def _validate(entries: tuple[Divergence, ...]) -> tuple[Divergence, ...]:
    """Refuse an entry that records a difference without recording what is known about it.

    An entry with no citation, or a one-word reason, is indistinguishable from silencing a
    failure — which is the one thing this module must not become. Checked at import so a
    malformed entry cannot reach a run.
    """
    for d in entries:
        if d.verdict not in _VERDICTS:
            raise ValueError(f"divergence {d.case!r}: verdict must be one of {_VERDICTS}")
        if not d.engine or d.engine == d.versus:
            raise ValueError(
                f"divergence {d.case!r}: `engine` names the odd one out and `versus`, when "
                "given, must be a different engine"
            )
        if len(d.reason.split()) < 5 or not d.citation.strip():
            raise ValueError(
                f"divergence {d.case!r}: needs a reason naming which side is right and a "
                "checkable citation. An entry that only says 'these differ' hides a failure."
            )
        if not d.signature.strip():
            raise ValueError(
                f"divergence {d.case!r}: needs a `signature` pinning it to the specific "
                "difference. Without one the entry matches every mismatch between these "
                "two engines and silences failures nobody has looked at."
            )
        if not d.case and len(d.signature) < 8:
            raise ValueError(
                f"divergence {d.case!r}: an entry with no `case` applies suite-wide, so its "
                f"`signature` must be specific; {d.signature!r} is too broad."
            )
    return entries


#: The recorded differences. Each one costs a reader nothing and saves the next person the
#: rediscovery; each one is also a claim, so each carries its evidence.
KNOWN_DIVERGENCES: tuple[Divergence, ...] = _validate(
    (
        Divergence(
            case="tpch-q6",
            engine="polars",
            versus=None,
            signature="revenue",
            verdict="batcher",
            reason=(
                "Polars folds the SQL literal `0.06 + 0.01` in IEEE double to "
                "0.06999999999999999, one ulp below 0.07, so it drops every "
                "`l_discount = 0.07` row and returns a low revenue. Batcher and DuckDB "
                "both fold to exactly 0.07 and match the published answer."
            ),
            citation="TPC-H spec 2.4.6 validation output; engines/polars.py module docstring",
        ),
        Divergence(
            case="tpch-q6",
            engine="daft",
            versus=None,
            signature="revenue",
            verdict="batcher",
            reason=(
                "Daft folds `0.06 + 0.01` in IEEE double exactly as Polars does, dropping "
                "every `l_discount = 0.07` row. Batcher and DuckDB match the published "
                "TPC-H answer for q6."
            ),
            citation="TPC-H spec 2.4.6 validation output; docs/benchmarks/methodology.md",
        ),
        Divergence(
            case="",
            engine="duckdb",
            versus="batcher",
            signature="column mismatch",
            verdict="batcher",
            reason=(
                "A `USING` join's coalesced key column comes first in Batcher's output, "
                "then the left table's remaining columns, then the right's. DuckDB keeps "
                "the key at its left-table index instead. Batcher follows the standard and "
                "PostgreSQL agrees with Batcher; this shows up only under `SELECT *`."
            ),
            citation="SQL:2016 part 2, section 7.7 <joined table>, syntax rule 1.b",
        ),
        Divergence(
            case="",
            engine="duckdb",
            versus="batcher",
            signature="var_samp",
            verdict="batcher",
            reason=(
                "`var_samp` over values near 2^53 differs: over {2^53, 2^53+2} Batcher "
                "returns 2.0 and DuckDB returns 4.0, where the exact rational answer is "
                "2.0. Batcher's Chan/Welford accumulation is the more accurate one, so the "
                "oracle is the side that is wrong here."
            ),
            citation="Chan, Golub & LeVeque (1983); measured on this tree, 2026-08-26",
        ),
    )
)


def explain(case: str, ref: str, other: str, message: str) -> Divergence | None:
    """The recorded divergence covering a `ref` vs `other` mismatch on `case`, if any.

    Args:
        case: The benchmark case name.
        ref: The reference (oracle) engine's name.
        other: The engine under test.
        message: The mismatch diff, which the entry's `signature` must appear in.

    Returns:
        The matching :class:`Divergence`, or ``None`` when this difference is not a
        recorded one — in which case it is a failure and must be reported as one.
    """
    for entry in KNOWN_DIVERGENCES:
        if entry.matches(case, ref, other, message):
            return entry
    return None
