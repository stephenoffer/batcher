# Data quality and governance

This page covers the scripts that assert contracts on data, and the ones that restrict who
can read which rows and columns.

## The four endings

A `ds.dq` chain ends in one of four ways, and which one you want depends on whether a
violation is a data problem to route or a promise that must hold.

```python
import batcher as bt

people = bt.from_pydict(
    {
        "id": [1, 2, 3, 4],
        "age": [34, 28, 51, 200],
        "country": ["US", "CA", "US", "ZZ"],
    }
)

contract = people.dq.in_range("age", 0, 120).accepted_values("country", ["US", "CA"])

report = contract.validate()          # a report, no raise
clean, rejected = contract.quarantine()  # both sides, for a dead-letter sink

assert not report.ok
assert clean.count() + rejected.count() == people.count()
assert rejected.to_pydict()["id"] == [4]
```

`drop` keeps only the conforming rows and `fail` raises. Profiling first is what makes the
thresholds defensible: writing a contract without looking at the null rate, cardinality and
range is guessing.

## Checks that can actually fail

Several classes of defect are invisible to a row count, so the scripts here check for them
directly. A schema check catches a column that widened upstream while still holding the same
values. A completeness check compares against the *expected* set of groups, because the rows
that would have made the count wrong are the ones that are absent. A referential-integrity
check is an anti join, and it finds exactly the rows an inner join would silently drop.

## Governance is a plan rewrite

Masking and row-level security are injected into the plan rather than applied to the result.
That distinction matters: an aggregate computed by a restricted principal is computed over
the restricted rows, so a count cannot leak the size of the hidden set, and a masked column
stays masked inside a group-by even when the query never projects it.

Residency defaults to `off`, which makes every check pass. That is deliberate, so a fleet can
measure in `advisory` before it blocks in `strict`, and it means setting the mode is the whole
control.

## Every script on this page

The table below lists the quality, governance and security scripts in path order.

<!-- library-table: quality,governance,security -->
| Script | Shows |
| --- | --- |
| `examples/quality/anomaly_detection.py` | Flagging rows that do not look like the rest |
| `examples/quality/completeness_checks.py` | Completeness: did every expected group arrive |
| `examples/quality/contract_lifecycle.py` | The life of a data contract: watch a rule, tolerate it, then enforce it |
| `examples/quality/contracts_on_real_data.py` | Asserting a data contract against a real table |
| `examples/quality/distribution_drift.py` | Detecting that today's data does not look like yesterday's |
| `examples/quality/end_to_end_gate.py` | A release gate: every check a pipeline should pass before it ships |
| `examples/quality/freshness_and_ranges.py` | Checking that data is recent and in range |
| `examples/quality/profiling_columns.py` | Profiling a table before you write any checks |
| `examples/quality/quarantine_workflow.py` | The full quarantine loop: split, write both sides, and reconcile |
| `examples/quality/reconciliation_report.py` | Reconciling a transformed dataset against its source |
| `examples/quality/referential_integrity.py` | Checking that foreign keys point at rows that exist |
| `examples/quality/rule_engine.py` | A rule engine as a projection: one boolean column per rule |
| `examples/quality/schema_contracts.py` | Asserting the schema, not just the values |
| `examples/quality/uniqueness_and_keys.py` | Checking that a key is actually a key |
| `examples/governance/lineage.py` | Column lineage: which inputs does this output column actually depend on? |
| `examples/governance/masking_and_filters.py` | Column masking and row filtering as a plan rewrite, not a wrapper |
| `examples/governance/pii_transforms.py` | Masking, hashing, and encrypting a sensitive column |
| `examples/security/audit_and_lineage.py` | Proving where a governed column went |
| `examples/security/audit_trail.py` | Recording who ran what, and proving the policy applied |
| `examples/security/column_masking.py` | Masking a sensitive column by tag, not by name |
| `examples/security/data_residency.py` | Data residency: refusing to process a dataset in the wrong region |
| `examples/security/masking_functions.py` | The masking functions, and what each preserves |
| `examples/security/row_level_security.py` | Restricting which rows a principal can see |
<!-- /library-table -->
