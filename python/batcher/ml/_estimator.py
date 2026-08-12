"""The scaffolding every fitted estimator in `batcher.ml` shares.

The estimators here (linear, discriminant, naive-Bayes, mixture, cluster, GLM) all follow one
shape: `fit` runs a handful of grouped aggregates and stores per-class parameters, then `predict`
lowers those parameters to an `Expr` the engine evaluates in Rust. Two pieces of that shape were
being written out by hand in every estimator — the "was this fitted?" guard and the argmax over a
set of per-class score expressions — so they live here once instead of once per module.

Neither helper touches a row: `argmax_prediction` builds a nested `when` chain that the engine
evaluates column-wise, which is why an estimator can score a billion rows without the control
plane seeing one of them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, TypeVar, runtime_checkable

from batcher._internal.errors import PlanError
from batcher.plan.expr_ir.constructors import col, lit, when

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from typing import TypeAlias

    from batcher.api.dataset import Dataset
    from batcher.plan.expr_ir.core import Expr

    # The callable shapes the model-selection, interpretation and tuning surfaces exchange.
    #
    # These are `TYPE_CHECKING`-only, and deliberately: an alias naming `Dataset` has to
    # resolve it, and `ml._estimator` is imported by estimators that `api.dataset` itself can
    # reach. Under `from __future__ import annotations` every annotation is already a string,
    # so a checker sees these and the runtime never needs them. They are *not* in `__all__`
    # for the same reason — a star-import would fail on a name that does not exist at runtime.
    #
    # What they replace: three modules each declared their own, as plain `str` values
    # (`Fit = "Callable[[Dataset], Any]"`). A string is not a type alias — a checker reads it
    # as `str` and silently gives up, so every signature annotated with one was effectively
    # `Any`. Two of the three also spelled the *same* concept differently (`Metric` twice,
    # `Predict` against `Predictor` at a different arity), which is what made a fourth
    # spelling the default outcome of adding tuning.

    #: A *bound* predictor: a fitted model already closed over, so it just scores.
    Predictor: TypeAlias = Callable[[Dataset], Dataset]
    #: Scores a prediction against the truth, as ``(dataset, y_true_col, y_pred_col) -> float``.
    #: Lower is better by convention, so a permutation importance is the *rise* under shuffling.
    Scorer: TypeAlias = Callable[[Dataset, str, str], float]
    #: The *unbound* pair, for cross-validation, which refits per fold: `Fit` trains and
    #: returns whatever model object it likes, and `Predict` takes that object back. Currying
    #: `Predict` with a fitted model gives a `Predictor` — which is the relationship the two
    #: separate spellings used to obscure.
    Fit: TypeAlias = Callable[[Dataset], Any]
    Predict: TypeAlias = Callable[[Any, Dataset], Dataset]

T = TypeVar("T")

__all__ = [
    "Estimator",
    "argmax_prediction",
    "linear_score",
    "require_fit_columns",
    "require_fitted",
    "require_numeric",
    "require_rows",
]


@runtime_checkable
class Estimator(Protocol):
    """What `batcher.ml`'s estimators have in common: `fit` learns, `predict` appends a column.

    The shape every native estimator already follows — `LinearRegression`, `KMeans`,
    `GaussianMixture`, the GLMs, the discriminants — stated once so the surfaces built on top
    of them (cross-validation, tuning, interpretation) can name what they accept instead of
    taking `Any`. `fit` returns `self` so `Model(...).fit(ds).predict(ds)` chains, and
    `predict` returns a new `Dataset` with the prediction column appended, executing nothing.

    Runtime-checkable so a caller can reject a mis-shaped object with a clear message rather
    than an `AttributeError` several frames deep. Note that this checks method *presence*
    only, which is all `runtime_checkable` can do — it is a guard, not a proof.
    """

    def fit(self, ds: Dataset) -> Estimator:
        """Learn this estimator's parameters from `ds` and return the fitted estimator."""
        ...

    def predict(self, ds: Dataset) -> Dataset:
        """Append this estimator's prediction column to `ds`, lazily."""
        ...


def require_fitted(estimator: object, state: T | None, method: str = "predict") -> T:
    """Return `state`, or raise if the estimator has not been fitted yet.

    Every estimator stores its learned parameters in a trailing-underscore attribute that is empty
    until `fit` runs. Calling `predict` first is the single most common mistake against this API,
    so it gets one typed error with the class name and the method in it rather than an
    `IndexError` from deep inside expression construction.

    Args:
        estimator: The estimator instance, used for the class name in the message.
        state: The learned-parameter attribute to check for emptiness.
        method: The method being guarded, named in the error message.

    Returns:
        The `state` value, guaranteed non-empty.

    Raises:
        PlanError: If `state` is `None` or empty.
    """
    if not state:
        name = type(estimator).__name__
        raise PlanError(f"{name} must be fitted before {method}().")
    return state


def linear_score(features: Sequence[str], weights: Sequence[float], intercept: float) -> Expr:
    """Build the linear predictor ``intercept + weights . features`` as one expression.

    The shared spine of every linear model here — least squares, ridge, the elastic net, the GLM
    link's inner term, and each class's discriminant score. The weights are folded in as literals
    at plan time, so what reaches the engine is a single arithmetic tree over the feature columns
    rather than a parameter lookup per row.

    Args:
        features: The feature column names, in the order the weights were fitted.
        weights: One coefficient per feature.
        intercept: The constant term.

    Returns:
        An expression evaluating to the linear score for each row.

    Raises:
        ValueError: If `features` and `weights` differ in length.
    """
    expression = lit(intercept)
    for weight, name in zip(weights, features, strict=True):
        expression = expression + lit(weight) * col(name)
    return expression


def argmax_prediction(labels: Sequence[T], score_of: Callable[[T], Expr]) -> Expr:
    """Build the expression selecting whichever label scores highest on each row.

    Folds the labels into a nested `when(score > best).then(label).otherwise(...)` chain, carrying
    the running maximum alongside the running argmax. That keeps the whole decision in one
    expression tree, so a classifier's `predict` is a single projection the JIT can compile rather
    than one pass per class.

    Args:
        labels: The fitted class labels, in a stable order. Ties resolve to the earliest label.
        score_of: Maps a label to its per-row score expression; higher wins.

    Returns:
        An expression evaluating to the highest-scoring label for each row.
    """
    prediction = lit(labels[0])
    best = score_of(labels[0])
    for label in labels[1:]:
        score = score_of(label)
        closer = score > best
        prediction = when(closer).then(lit(label)).otherwise(prediction)
        best = when(closer).then(score).otherwise(best)
    return prediction


def require_rows(estimator: object, rows: int, needed: int, *, because: str) -> None:
    """Raise unless `rows` is enough to fit, naming the count and why that many are needed.

    Every estimator here fits from aggregates, and an aggregate over an empty relation is
    null — so `float(None)` raised ``TypeError: float() argument must be a string or a real
    number, not 'NoneType'`` from inside the solve, a message about a conversion for a
    dataset that simply had nothing in it. The floor differs by estimator (a covariance needs
    two rows, an IRLS fit needs one per term), which is why `because` is the caller's to say.

    Args:
        estimator: The estimator instance, for the class name in the message.
        rows: How many rows the training set actually has.
        needed: The minimum this fit requires.
        because: Why that many, phrased to complete "needs at least N rows because ...".

    Raises:
        PlanError: If `rows` is below `needed`.
    """
    if rows >= needed:
        return
    raise PlanError(
        f"{type(estimator).__name__} cannot fit on {rows} row(s): it needs at least "
        f"{needed} because {because}. Check the filters upstream of fit()."
    )


def require_numeric(
    estimator: object, ds: Dataset, names: Sequence[str], *, role: str = "feature"
) -> None:
    """Raise naming the column unless every one of `names` is something a fit can average.

    The estimators here reach the engine through aggregates and arithmetic, neither of which
    is defined on a string. Without this the failure surfaced from inside the data plane as
    ``aggregate mean is not supported for column type Utf8``, or a cast error, or - worst -
    ``Invalid arithmetic operation: Float64`` with no column named in any of them. Eight
    estimators produced eight different messages for the same mistake, and not one said which
    column was wrong or what to do about it.

    The check reads the schema, which is known when the plan is built, so it costs no pass
    over the data and fires before any work is scheduled.

    Args:
        estimator: The estimator doing the checking, named in the message.
        ds: The dataset whose schema to read.
        names: The columns that must be numeric.
        role: What those columns are to the estimator, named in the message. A regressor
            passes ``"target"`` for the column it predicts; a classifier does not, because
            a class label is legitimately a string.

    Raises:
        PlanError: If a column is present but cannot be used as a numeric feature.
    """
    import pyarrow.types as types

    schema = ds.schema
    for name in names:
        index = schema.get_field_index(name)
        if index < 0:  # absent: the caller's own missing-column check owns that message
            continue
        dtype = schema.field(index).type
        if types.is_floating(dtype) or types.is_integer(dtype) or types.is_decimal(dtype):
            continue
        what = type(estimator).__name__ if not isinstance(estimator, str) else estimator
        if types.is_null(dtype):
            # An untyped column is ambiguous from the schema alone: it is what an all-null
            # column looks like, and equally what an *empty* dataset looks like. Rejecting it
            # broke fitting on an empty frame - a filtered-to-nothing training set, an empty
            # partition - which is a legitimate input, so this defers to the row-count checks
            # rather than guessing. Only the unambiguous types below are rejected.
            continue
        if types.is_boolean(dtype):
            # A boolean flag is a perfectly good 0/1 feature and every estimator here refuses
            # it, because the engine defines neither `mean` nor arithmetic on Boolean. The
            # four messages that produced - a mean failure, two arithmetic failures and a
            # comparison failure - named the type but never the column or the one-line fix.
            raise PlanError(
                f"{what}: {role} {name!r} is boolean, and the engine's aggregates are not "
                "defined on that type. Cast it first: "
                f'ds.with_columns({name}=bt.col("{name}").cast("int64")).'
            )
        raise PlanError(
            f"{what}: {role} {name!r} has type {dtype}, and a fit needs a number. Encode a "
            "categorical column first with OrdinalEncoder, OneHotEncoder or TargetEncoder, "
            "and parse a date or a string of digits into a numeric column."
        )


def require_fit_columns(
    estimator: object,
    ds: Dataset,
    features: Sequence[str],
    target: str | None = None,
    *,
    numeric_target: bool = False,
) -> None:
    """Check everything a `fit` needs of its columns, in the one order that reads well.

    Every estimator here opened `fit` with the same three checks, spelled out in eight lines
    each: that every named column exists, that the features are numeric, and — for a regressor
    but not a classifier — that the target is too. Ten modules carried a copy, and the copies
    were uniform in what they *did* while disagreeing on nothing at all, which is the signature
    of a helper that was never written rather than a difference worth keeping.

    The order matters and is the reason this is one call rather than three. "That column does
    not exist" must come before "that column is not numeric": a typo'd feature name has *no*
    type, so checking numeracy first either passes it through (`require_numeric` skips an
    absent column deliberately) or reports the wrong problem, and either way the message the
    user needs — ``did you mean 'price'?`` — never appears.

    Args:
        estimator: The estimator doing the checking, named in the message.
        ds: The dataset about to be fitted.
        features: The feature columns, which must exist and be numeric.
        target: The column being predicted, which must exist. `None` for an unsupervised fit.
        numeric_target: Whether `target` must also be numeric. True for a regressor; false for
            a classifier, whose label is legitimately a string.

    Raises:
        ColumnNotFoundError: If a named column is not in `ds`.
        PlanError: If a column is present but cannot be used as a number.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml._estimator import require_fit_columns
            >>> ds = bt.from_pydict({"x": [1.0], "y": [2.0]})
            >>> require_fit_columns("Ridge", ds, ["x"], "y", numeric_target=True)
    """
    from batcher.ml.stats._shared import require_columns

    named = [*features, target] if target is not None else list(features)
    require_columns(ds, *named)
    require_numeric(estimator, ds, features)
    if target is not None and numeric_target:
        require_numeric(estimator, ds, [target], role="target")
