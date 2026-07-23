"""`Chain` — a sequence of preprocessors fitted and applied as one (sklearn ``Pipeline``).

Composing preprocessors by hand is easy to get *subtly* wrong: each step must be fitted
on the output of the previous **fitted** step, and the held-out split must then flow
through those same fitted steps in the same order. Written out, that is

    imputer = SimpleImputer(["age"]).fit(train)
    scaler = StandardScaler(["age"]).fit(imputer.transform(train))
    train_x = scaler.transform(imputer.transform(train))
    test_x = scaler.transform(imputer.transform(test))

— four chances to fit on the wrong frame, and every one of them leaks test statistics
into training features without failing. `Chain` is that loop written once:
``chain.fit(train)`` threads each step's output into the next, and
``chain.transform(part)`` replays the fitted steps in order.

A `Chain` is itself a `Preprocessor`, so it nests and can be a step of another chain.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from batcher._internal.errors import PlanError
from batcher.ml.preprocessors.base import Preprocessor

if TYPE_CHECKING:
    from collections.abc import Iterator

    from batcher.api.dataset import Dataset

__all__ = ["Chain"]


class Chain(Preprocessor):
    """Fit and apply preprocessors in order, threading each step's output to the next.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml import Chain, SimpleImputer, StandardScaler
            >>> train = bt.from_pydict({"age": [10.0, 20.0, None, 40.0]})
            >>> chain = Chain(SimpleImputer(["age"]), StandardScaler(["age"]))
            >>> out = chain.fit_transform(train)
            >>> [round(v, 3) for v in out.to_pydict()["age"]]
            [-1.234, -0.309, 0.0, 1.543]
    """

    __slots__ = ("cache", "steps")

    def __init__(self, *steps: Preprocessor, cache: bool = True) -> None:
        """Build a chain from `steps`, applied left to right.

        Args:
            *steps: The preprocessors to run in order. At least one is required.
            cache: Materialize the fit input once instead of re-executing the upstream
                plan for every step. Each step's `fit` runs its own aggregate, so an
                N-step chain otherwise costs N full scans of the source. Set it False
                when the training split is too large to hold in driver memory, and pay
                the rescans instead.

        Raises:
            PlanError: If no steps are given, or a step is not a `Preprocessor`.
        """
        # Accept both spellings: Chain(a, b) and Chain([a, b]) — the latter matches
        # scikit-learn's make_pipeline([...]) habit and is what a user reaches for when
        # the steps are already in a list.
        if len(steps) == 1 and isinstance(steps[0], (list, tuple)):
            steps = tuple(steps[0])
        if not steps:
            raise PlanError("Chain() requires at least one preprocessor")
        for step in steps:
            if not isinstance(step, Preprocessor):
                raise PlanError(
                    f"Chain() steps must be Preprocessor instances, got {type(step).__name__}"
                )
        self.steps = list(steps)
        self.cache = cache

    def _fit_input(self, ds: Dataset) -> Dataset:
        """The dataset the steps fit against — materialized once when `cache` is set.

        Every step's `fit` is a terminal aggregate, and the result cache is keyed on the
        whole plan, so a lazy handle is re-executed from the source once per step. Reading
        the input into memory once turns those N source scans into one.
        """
        if not self.cache or len(self.steps) < 2:
            return ds
        from batcher.api.session import from_arrow

        return from_arrow(ds.collect())

    def __len__(self) -> int:
        """How many steps the chain holds.

        Examples:
            .. doctest::

                >>> from batcher.ml.preprocessors import Chain, SimpleImputer, StandardScaler
                >>> len(Chain(SimpleImputer(["a"]), StandardScaler(["a"])))
                2

        Returns:
            The number of steps.
        """
        return len(self.steps)

    def __iter__(self) -> Iterator[Preprocessor]:
        """Iterate the steps in application order.

        Examples:
            .. doctest::

                >>> from batcher.ml.preprocessors import Chain, SimpleImputer, StandardScaler
                >>> ch = Chain(SimpleImputer(["a"]), StandardScaler(["a"]))
                >>> [type(s).__name__ for s in ch]
                ['SimpleImputer', 'StandardScaler']

        Returns:
            An iterator over the steps, in application order.
        """
        return iter(self.steps)

    def __getitem__(self, index: int) -> Preprocessor:
        """The step at `index` — useful to read a fitted step's learned state.

        Examples:
            .. doctest::

                >>> from batcher.ml.preprocessors import Chain, SimpleImputer, StandardScaler
                >>> ch = Chain(SimpleImputer(["a"]), StandardScaler(["a"]))
                >>> type(ch[0]).__name__
                'SimpleImputer'

        Args:
            index: The position of the step to return.

        Returns:
            The step at `index`.
        """
        return self.steps[index]

    def __repr__(self) -> str:
        """``Chain(SimpleImputer, StandardScaler)`` — the steps by class name."""
        return f"Chain({', '.join(type(s).__name__ for s in self.steps)})"

    def fit(self, ds: Dataset) -> Chain:
        """Fit every step, each on the output of the steps already fitted before it.

        This is the whole point of the class: step *i* must learn its statistics from
        data that steps *0..i-1* have already transformed, or it learns them from a
        distribution that will never be fed to it again.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.preprocessors import Chain, SimpleImputer
                >>> ch = Chain(SimpleImputer(["a"])).fit(bt.from_pydict({"a": [1.0, None, 3.0]}))
                >>> ch[0].statistics_
                {'a': 2.0}

        Args:
            ds: The **training** split. Fitting on anything else leaks into the model.

        Returns:
            ``self``, fitted.
        """
        current = self._fit_input(ds)
        for step in self.steps:
            step.fit(current)
            current = step.transform(current)
        self._fitted = True
        return self

    def transform(self, ds: Dataset) -> Dataset:
        """Apply the fitted steps to `ds` in order, returning a new lazy `Dataset`.

        Lazy: each step contributes an `Expr` projection, so the whole chain is one
        plan the engine optimizes and executes as a unit.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.preprocessors import Chain, SimpleImputer
                >>> ch = Chain(SimpleImputer(["a"])).fit(bt.from_pydict({"a": [1.0, None, 3.0]}))
                >>> ch.transform(bt.from_pydict({"a": [None, 4.0]})).to_pydict()
                {'a': [2.0, 4.0]}

        Args:
            ds: The dataset (e.g. the held-out split) to run the fitted steps over.

        Returns:
            A new lazy `Dataset` with every fitted step applied in order.

        Raises:
            PlanError: If the chain has not been fitted.
        """
        self._require_fitted()
        for step in self.steps:
            ds = step.transform(ds)
        return ds

    def fit_transform(self, ds: Dataset) -> Dataset:
        """`fit(ds)` then `transform(ds)`, reusing each step's already-computed output.

        Overridden so the chain does not transform `ds` twice: `fit` already threaded
        the data through every step, so its final output *is* the answer.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.preprocessors import Chain, SimpleImputer, StandardScaler
                >>> train = bt.from_pydict({"age": [10.0, 20.0, None, 40.0]})
                >>> chain = Chain(SimpleImputer(["age"]), StandardScaler(["age"]))
                >>> [round(v, 3) for v in chain.fit_transform(train).to_pydict()["age"]]
                [-1.234, -0.309, 0.0, 1.543]

        Args:
            ds: The **training** split to fit on and transform in one pass.

        Returns:
            A new lazy `Dataset` with every step fitted and applied.
        """
        current = self._fit_input(ds)
        for step in self.steps:
            current = step.fit_transform(current)
        self._fitted = True
        return current
