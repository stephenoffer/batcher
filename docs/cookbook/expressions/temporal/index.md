# Dates and times

Start with the parts, then the arithmetic, then the two that most often produce a wrong report.

Each page embeds a complete, self-contained script that builds its own in-memory data and asserts on its own output, so a page that stops matching the engine fails the suite instead of drifting.

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
