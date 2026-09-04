"""Per-operator internals benchmarks: one module per stateful operator.

These measure a single operator against the engine it is claimed to beat, at controlled
size and selectivity, rather than as part of a whole query. They live under their own
package because the operator-level questions ("where does this algorithm's cost curve
cross DuckDB's?") are asked per operator and answered by data shapes that no public
benchmark corpus supplies.
"""
