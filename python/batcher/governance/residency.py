"""Data residency — where a dataset is allowed to be computed on, not just stored.

Storage residency is the easy half and the half most systems stop at: the bytes live in a
bucket in a named region. The half that actually breaks a sovereignty obligation is *compute*.
A distributed job reads an EU dataset, the scheduler finds spare accelerator capacity in
another region because that is where the queue is shortest, and the rows cross a border inside
a shuffle. Nothing fails, nothing is logged, and the obligation is broken.

This module is the control-plane answer: a catalog of which regions each dataset may be
processed in, and a check that runs before placement rather than after. It sits in
`governance` beside the column masks and row filters because it is the same kind of thing — a
declarative policy over data, resolved to a verdict — and the layer that acts on it is
scheduling, which may read governance but not the other way round.

**Three modes, matching the rest of governance.** `off` (the default) checks nothing. `advisory`
returns a failing verdict a caller logs and proceeds past, which is how an operator finds every
violating placement in a real workload *before* enforcing. `strict` is the one that refuses.
Without the advisory step, enforcement is unadoptable, because the first strict run of a large
pipeline fails somewhere nobody predicted.

**An unregistered dataset is unrestricted.** Residency is an obligation an operator states, not
one this module infers: guessing a region from a bucket name or an endpoint URL would be a
fabricated legal fact, and the failure mode of guessing wrong is a compliance incident in
whichever direction it errs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from batcher._internal.errors import AccessDeniedError

__all__ = [
    "RESIDENCY_MODES",
    "DataResidency",
    "ResidencyCatalog",
    "ResidencyVerdict",
]

#: Enforcement modes, weakest first. `advisory` exists so a fleet can measure before it blocks.
RESIDENCY_MODES = ("off", "advisory", "strict")


@dataclass(frozen=True, slots=True)
class DataResidency:
    """One dataset's permitted processing regions, and the obligation behind them.

    Examples:
        .. doctest::

            >>> from batcher.governance import DataResidency
            >>> rule = DataResidency("s3://eu-customers/", frozenset({"eu-north-1"}), "GDPR")
            >>> sorted(rule.allowed_regions)
            ['eu-north-1']

    Attributes:
        dataset: Dataset name or path prefix the rule governs. Prefix matching is longest-wins,
            so `"s3://eu-customer/"` can carry a default and `"s3://eu-customer/public/"` a
            narrower exception.
        allowed_regions: Regions the data may be processed in. Empty means *no* region is
            permitted, which is a valid and deliberate way to quarantine a dataset — it is not
            the same as unregistered.
        obligation: What requires this, in the operator's own words: a regulation, a contract
            id, a customer commitment. Carried into the verdict so a refusal explains itself.
    """

    dataset: str
    allowed_regions: frozenset[str] = field(default_factory=frozenset)
    obligation: str = ""


@dataclass(frozen=True, slots=True)
class ResidencyVerdict:
    """Whether a placement is permitted, and why not when it is not.

    Examples:
        .. doctest::

            >>> from batcher.governance import ResidencyVerdict
            >>> ResidencyVerdict(allowed=True).message()
            ''

    Attributes:
        allowed: Whether the placement may proceed.
        dataset: The dataset checked.
        region: The region checked against.
        allowed_regions: The regions the rule permits, empty when unregistered.
        obligation: The rule's stated obligation, `""` when unregistered.
        enforced: Whether a refusal would actually block, as opposed to being advisory.
    """

    allowed: bool
    dataset: str = ""
    region: str = ""
    allowed_regions: frozenset[str] = field(default_factory=frozenset)
    obligation: str = ""
    enforced: bool = False

    def message(self) -> str:
        """A one-line explanation suitable for a log line or an exception.

        Examples:
            .. doctest::

                >>> from batcher.governance import ResidencyVerdict
                >>> verdict = ResidencyVerdict(
                ...     allowed=False,
                ...     dataset="s3://eu/orders",
                ...     region="us-east-1",
                ...     allowed_regions=frozenset({"eu-north-1"}),
                ... )
                >>> verdict.message().endswith("permitted in eu-north-1")
                True

        Returns:
            An empty string when allowed, otherwise the dataset, the refused region, the
            permitted set, and the obligation that requires it.
        """
        if self.allowed:
            return ""
        permitted = ", ".join(sorted(self.allowed_regions)) or "no region"
        because = f" ({self.obligation})" if self.obligation else ""
        return (
            f"dataset {self.dataset!r} may not be processed in region {self.region!r}: "
            f"permitted in {permitted}{because}"
        )


@dataclass
class ResidencyCatalog:
    """The registered residency rules, and the checks that resolve against them.

    Examples:
        .. doctest::

            >>> from batcher.governance import DataResidency, ResidencyCatalog
            >>> catalog = ResidencyCatalog(mode="strict")
            >>> _ = catalog.register(DataResidency("s3://eu/", frozenset({"eu-north-1"})))
            >>> catalog.check("s3://eu/orders", "us-east-1").allowed
            False

    Attributes:
        mode: One of `RESIDENCY_MODES`. `off` (the default) makes every check pass.
        rules: Registered rules, keyed by dataset name or path prefix.
    """

    mode: str = "off"
    rules: dict[str, DataResidency] = field(default_factory=dict)

    def register(self, rule: DataResidency) -> ResidencyCatalog:
        """Add or replace a rule, returning self so registrations chain.

        Examples:
            .. doctest::

                >>> from batcher.governance import DataResidency, ResidencyCatalog
                >>> catalog = ResidencyCatalog().register(DataResidency("s3://eu/"))
                >>> len(catalog.rules)
                1

        Args:
            rule: The rule to register; a second rule for the same key replaces the first.

        Returns:
            This catalog.
        """
        self.rules[rule.dataset] = rule
        return self

    def rule_for(self, dataset: str) -> DataResidency | None:
        """The most specific registered rule governing a dataset, or `None`.

        Examples:
            .. doctest::

                >>> from batcher.governance import DataResidency, ResidencyCatalog
                >>> catalog = ResidencyCatalog().register(DataResidency("s3://eu/"))
                >>> catalog.rule_for("s3://eu/orders").dataset
                's3://eu/'
                >>> catalog.rule_for("s3://other/orders") is None
                True

        Longest-prefix wins, so a narrow exception under a broad default is expressible
        without ordering rules by hand.

        Args:
            dataset: Dataset name or path.

        Returns:
            The governing rule, or `None` when the dataset is unregistered.
        """
        best: DataResidency | None = None
        for key, rule in self.rules.items():
            matches = dataset == key or dataset.startswith(key)
            if matches and (best is None or len(key) > len(best.dataset)):
                best = rule
        return best

    def check(self, dataset: str, region: str) -> ResidencyVerdict:
        """Whether a dataset may be processed in a region.

        Examples:
            .. doctest::

                >>> from batcher.governance import DataResidency, ResidencyCatalog
                >>> catalog = ResidencyCatalog(mode="advisory")
                >>> _ = catalog.register(DataResidency("s3://eu/", frozenset({"eu-north-1"})))
                >>> catalog.check("s3://eu/orders", "eu-north-1").allowed
                True
                >>> catalog.check("s3://eu/orders", "us-east-1").enforced
                False

        Args:
            dataset: Dataset name or path.
            region: The region work would run in.

        Returns:
            The verdict. Always allowed when the mode is `off`, when the dataset is
            unregistered, or when the region is unknown (`""`) — an unlabelled fleet must not
            be refused by a policy it cannot be evaluated against, which would take a cluster
            offline the day a node label was dropped.
        """
        if self.mode == "off":
            return ResidencyVerdict(allowed=True, dataset=dataset, region=region)
        rule = self.rule_for(dataset)
        if rule is None or not region:
            return ResidencyVerdict(allowed=True, dataset=dataset, region=region)
        return ResidencyVerdict(
            allowed=region in rule.allowed_regions,
            dataset=dataset,
            region=region,
            allowed_regions=rule.allowed_regions,
            obligation=rule.obligation,
            enforced=self.mode == "strict",
        )

    def enforce(self, dataset: str, region: str) -> ResidencyVerdict:
        """Check a placement and raise in `strict` mode when it is refused.

        Examples:
            .. doctest::

                >>> from batcher.governance import DataResidency, ResidencyCatalog
                >>> catalog = ResidencyCatalog(mode="advisory")
                >>> _ = catalog.register(DataResidency("s3://eu/", frozenset({"eu-north-1"})))
                >>> catalog.enforce("s3://eu/orders", "us-east-1").allowed
                False

        Args:
            dataset: Dataset name or path.
            region: The region work would run in.

        Returns:
            The verdict, which in `advisory` mode may be a refusal the caller logs and
            proceeds past.

        Raises:
            AccessDeniedError: In `strict` mode, when the placement is not permitted.
        """
        verdict = self.check(dataset, region)
        if not verdict.allowed and verdict.enforced:
            raise AccessDeniedError(verdict.message())
        return verdict

    def permitted_regions(self, datasets: list[str] | tuple[str, ...]) -> frozenset[str] | None:
        """Regions where *every* named dataset may be processed.

        Examples:
            .. doctest::

                >>> from batcher.governance import DataResidency, ResidencyCatalog
                >>> catalog = ResidencyCatalog(mode="strict")
                >>> _ = catalog.register(DataResidency("s3://eu/", frozenset({"eu-north-1"})))
                >>> _ = catalog.register(DataResidency("s3://uk/", frozenset({"eu-west-2"})))
                >>> sorted(catalog.permitted_regions(["s3://eu/a", "s3://uk/b"]))
                []

        The figure a multi-input job needs: a join of an EU dataset and a UK dataset may run
        only where both are permitted, and that intersection is frequently empty — which is a
        real answer, not an error, and means the job must be split rather than placed.

        Args:
            datasets: Dataset names or paths the job reads.

        Returns:
            The intersection of the permitted sets, or `None` when no input is registered
            (meaning unrestricted). An empty set means the job cannot run anywhere as written.
        """
        sets = [r.allowed_regions for r in (self.rule_for(d) for d in datasets) if r is not None]
        if not sets or self.mode == "off":
            return None
        out = sets[0]
        for s in sets[1:]:
            out &= s
        return out

    def filter_regions(
        self,
        candidates: list[str] | tuple[str, ...],
        datasets: list[str] | tuple[str, ...],
    ) -> tuple[str, ...]:
        """Candidate regions that every named dataset permits, in the given order.

        Examples:
            .. doctest::

                >>> from batcher.governance import DataResidency, ResidencyCatalog
                >>> catalog = ResidencyCatalog(mode="strict")
                >>> _ = catalog.register(DataResidency("s3://eu/", frozenset({"eu-north-1"})))
                >>> catalog.filter_regions(["us-east-1", "eu-north-1"], ["s3://eu/orders"])
                ('eu-north-1',)

        What a scheduler calls before choosing where to place a stage.

        Args:
            candidates: Regions with capacity, in preference order.
            datasets: Dataset names or paths the stage reads.

        Returns:
            The permitted subset, order preserved. Every candidate when nothing is registered
            or the mode is `off`.
        """
        permitted = self.permitted_regions(datasets)
        if permitted is None:
            return tuple(candidates)
        return tuple(c for c in candidates if c in permitted)
