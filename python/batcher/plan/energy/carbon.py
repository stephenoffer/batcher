"""Turning joules into the two figures a datacenter is actually judged on: cost and carbon.

Energy is the physical quantity; money and emissions are what an operator reports, bills, and
schedules against. Both conversions are one multiplication each, and both depend on facts this
process cannot discover — the facility's power usage effectiveness, the grid's carbon
intensity right now, the contracted price per kilowatt-hour. So none of them are guessed here.

**There is no built-in table of regional carbon intensities.** A grid's intensity varies by
hour, by season, and by contract (a hydro-backed site under a power purchase agreement is not
its national average), so a shipped table would be authoritative-looking and wrong. Every
conversion takes the intensity as an argument, defaulting to `0.0` meaning "not configured",
in which case the reported emissions are `0.0` and callers can tell that apart from a genuine
zero by checking `configured`.

The unit convention throughout: energy in **joules**, intensity in **grams CO2e per
kilowatt-hour**, price in **currency units per kilowatt-hour**, and `pue` as a dimensionless
multiplier at or above 1.0.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "GridProfile",
    "carbon_grams",
    "configured_grid",
    "energy_cost",
    "joules_to_kwh",
    "kwh_to_joules",
]

#: Joules in one kilowatt-hour.
_JOULES_PER_KWH = 3.6e6


def joules_to_kwh(joules: float) -> float:
    """Convert joules to kilowatt-hours.

    Args:
        joules: Energy in joules.

    Returns:
        Kilowatt-hours; `0.0` for a non-positive input.
    """
    return joules / _JOULES_PER_KWH if joules > 0 else 0.0


def kwh_to_joules(kwh: float) -> float:
    """Convert kilowatt-hours to joules.

    Args:
        kwh: Energy in kilowatt-hours.

    Returns:
        Joules; `0.0` for a non-positive input.
    """
    return kwh * _JOULES_PER_KWH if kwh > 0 else 0.0


def carbon_grams(joules: float, gco2e_per_kwh: float, pue: float = 1.0) -> float:
    """Emissions attributable to an amount of IT-load energy, in grams CO2e.

    Args:
        joules: IT-load energy — what the servers drew, before facility overhead.
        gco2e_per_kwh: Grid carbon intensity; `0.0` means not configured and reports `0.0`.
        pue: Power usage effectiveness, the facility's total draw divided by its IT load.
            Values below 1.0 are impossible and are clamped up.

    Returns:
        Grams of CO2e, or `0.0` when the intensity is not configured.
    """
    if joules <= 0 or gco2e_per_kwh <= 0:
        return 0.0
    return joules_to_kwh(joules) * max(1.0, pue) * gco2e_per_kwh


def energy_cost(joules: float, price_per_kwh: float, pue: float = 1.0) -> float:
    """Energy cost of an amount of IT-load energy, in the price's currency units.

    Args:
        joules: IT-load energy in joules.
        price_per_kwh: Contracted energy price; `0.0` means not configured.
        pue: Power usage effectiveness; clamped to at least 1.0.

    Returns:
        Cost in the same currency as `price_per_kwh`, or `0.0` when unpriced.
    """
    if joules <= 0 or price_per_kwh <= 0:
        return 0.0
    return joules_to_kwh(joules) * max(1.0, pue) * price_per_kwh


@dataclass(frozen=True, slots=True)
class GridProfile:
    """The facility and grid facts that convert a workload's joules into cost and carbon.

    Carried as one value rather than three loose floats so a run records the basis of its own
    numbers: an emissions figure without the intensity that produced it cannot be audited or
    compared against another site's.

    Attributes:
        region: Free-form site or region identifier, for attribution in reports.
        gco2e_per_kwh: Grid carbon intensity; `0.0` when not configured.
        price_per_kwh: Energy price; `0.0` when not configured.
        pue: Facility power usage effectiveness; `1.0` means IT load only.
        renewable_fraction: Fraction of supply from carbon-free generation, in [0, 1], for
            reporting only — it does not adjust the intensity, which already accounts for
            the mix.
    """

    region: str = ""
    gco2e_per_kwh: float = 0.0
    price_per_kwh: float = 0.0
    pue: float = 1.0
    renewable_fraction: float = 0.0

    @property
    def configured(self) -> bool:
        """Whether this profile can produce a non-zero cost or carbon figure."""
        return self.gco2e_per_kwh > 0 or self.price_per_kwh > 0

    def carbon_grams(self, joules: float) -> float:
        """Emissions for an amount of IT-load energy under this profile.

        Args:
            joules: IT-load energy in joules.

        Returns:
            Grams CO2e, `0.0` when the intensity is unconfigured.
        """
        return carbon_grams(joules, self.gco2e_per_kwh, self.pue)

    def cost(self, joules: float) -> float:
        """Energy cost for an amount of IT-load energy under this profile.

        Args:
            joules: IT-load energy in joules.

        Returns:
            Cost in the profile's currency, `0.0` when unpriced.
        """
        return energy_cost(joules, self.price_per_kwh, self.pue)

    def facility_joules(self, joules: float) -> float:
        """IT-load energy grossed up by PUE — what the facility drew for this work.

        Args:
            joules: IT-load energy in joules.

        Returns:
            Facility energy in joules.
        """
        return joules * max(1.0, self.pue) if joules > 0 else 0.0


def configured_grid() -> GridProfile:
    """The site's grid profile, read from the active configuration.

    One accessor, so a cost figure and a carbon figure produced in different layers cannot
    disagree about the price, the intensity, or the facility overhead they used.

    Returns:
        A `GridProfile`; unconfigured (and so reporting no cost and no emissions) by default.
    """
    from batcher.config import active_config

    energy = active_config().accelerator.energy
    return GridProfile(
        region=energy.region,
        gco2e_per_kwh=energy.carbon_intensity,
        price_per_kwh=energy.price_per_kwh,
        pue=energy.pue,
        renewable_fraction=energy.renewable_fraction,
    )
