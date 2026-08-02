# User behavior

These recipes answer questions about what people did over time: who came back, who dropped
out, and where one visit ends and the next begins. All four are window functions over an
event table rather than anything specialised.

| Recipe | The question |
|---|---|
| {doc}`Cohort analysis <cohort-analysis>` | Do customers acquired in January behave differently from customers acquired in March? |
| {doc}`Retention curves <retention-curves>` | Of the people who signed up on Monday, how many came back on day 1, 7, and 30? |
| {doc}`Funnel analysis <funnel-analysis>` | How many people made it from view to cart to checkout, and where they left |
| {doc}`Sessionization <sessionization>` | Turning a flat click stream into the bursts of activity it is made of |

## See also

- {doc}`/user-guide/analyze/window-functions`: the window semantics all four recipes lean on.
- {doc}`/cookbook/analytics/inference/index`: the same event data, asked statistical questions.

```{toctree}
:hidden:

cohort-analysis
retention-curves
funnel-analysis
sessionization
```
