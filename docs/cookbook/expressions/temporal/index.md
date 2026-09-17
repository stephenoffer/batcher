# Dates and times

The `.dt` accessor for timestamp columns. Start with the calendar parts, then durations and shifts, then truncation for rollup keys. The time-zone and business-day recipes come last because they are the two that most often produce a report that looks right and isn't.

| Recipe | What it shows |
|---|---|
| {doc}`/cookbook/expressions/temporal/temporal_parts` | Pulling calendar parts out of a timestamp column |
| {doc}`/cookbook/expressions/temporal/temporal_differences` | Durations between two timestamps, and shifting one |
| {doc}`/cookbook/expressions/temporal/temporal_truncation` | Truncating to a period, or snapping to a boundary |
| {doc}`/cookbook/expressions/temporal/temporal_timezones` | Converting between zones, and the reporting-boundary trap |
| {doc}`/cookbook/expressions/temporal/temporal_business_days` | Weekend and business-day predicates, and output formatting |

```{toctree}
:hidden:

temporal_parts
temporal_differences
temporal_truncation
temporal_timezones
temporal_business_days
```
