# The relational surface

The verbs, the column language, and the boundary data crosses. Each page leads with a runnable example and then enumerates the surface.

| Page | Covers |
|---|---|
| {doc}`/api/relational/dataset` | Build, transform, aggregate, join, and collect |
| {doc}`/api/relational/expressions` | Column math, predicates, operators, and window methods |
| {doc}`/api/relational/expression-accessors` | Every {py:class}`.str <batcher.plan.expr_ir.namespaces.strings._StrNamespace>`, {py:class}`.dt <batcher.plan.expr_ir.namespaces.temporal._DtNamespace>`, {py:class}`.list <batcher.plan.expr_ir.namespaces.collections._ListNamespace>`, {py:class}`.struct <batcher.plan.expr_ir.namespaces.collections._StructNamespace>`, {py:class}`.json <batcher.plan.expr_ir.namespaces.collections._JsonNamespace>`, {py:class}`.map <batcher.plan.expr_ir.namespaces.collections._MapNamespace>`, {py:class}`.image <batcher.plan.expr_ir.image._ImageNamespace>`, {py:class}`.audio <batcher.plan.expr_ir.audio._AudioNamespace>`, and {py:class}`.video <batcher.plan.expr_ir.video._VideoNamespace>` method |
| {doc}`/api/relational/functions` | Scalar, horizontal, aggregate, and window functions |
| {doc}`/api/relational/geospatial` | Every `ST_*` function: geometry, predicates, measures, grids |
| {doc}`/api/relational/spatial` | Rotations, poses and coordinate frames for robotics and AV |
| {doc}`/api/relational/graph` | Graph analytics and graph-ML features over an edge table |
| {doc}`/api/relational/sql` | The SQL surface, and how it lowers to the DataFrame API |
| {doc}`/api/relational/io` | Every reader and writer, with the optional extras |

```{toctree}
:hidden:

dataset
expressions
expressions-datascience
expression-accessors
functions
geospatial
spatial
graph
sql
io
```
