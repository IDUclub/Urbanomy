"""Configuration defaults for SER estimation routines."""

from __future__ import annotations

from typing import Final, Mapping

from blocksnet.enums import LandUse


def _value_map(
    values: Mapping[LandUse, float],
    default: float,
) -> dict[str, float]:
    """Return a string-keyed mapping backed by LandUse values."""
    mapping = {lu.value: float(val) for lu, val in values.items()}
    mapping["default"] = float(default)
    return mapping


DEFAULT_AVG_WAGE_BASE: Final[float] = 70_430
DEFAULT_TAX_RATES: Final[dict[str, float]] = {
    "pit": 0.13,
    "cit": 0.17,
    "prop": 0.02,
    "land": 0.015,
}
DEFAULT_VA_PER_M2_OPS: Final[dict[str, float]] = _value_map(
    {
        LandUse.BUSINESS: 12_000.0,
        LandUse.RESIDENTIAL: 2_000.0,
        LandUse.TRANSPORT: 5_000.0,
        LandUse.AGRICULTURE: 3_000.0,
        LandUse.SPECIAL: 7_000.0,
    },
    default=0.0,
)
DEFAULT_JOBS_PER_M2: Final[dict[str, float]] = _value_map(
    {
        LandUse.BUSINESS: 1 / 18,
        LandUse.RESIDENTIAL: 0.0,
        LandUse.TRANSPORT: 1 / 90,
        LandUse.AGRICULTURE: 1 / 400,
        LandUse.SPECIAL: 1 / 30,
    },
    default=0.0,
)
DEFAULT_WAGE_BY_USE: Final[dict[str, float]] = _value_map(
    {
        LandUse.BUSINESS: 85_000,
        LandUse.RESIDENTIAL: 0,
        LandUse.TRANSPORT: 60_000,
        LandUse.AGRICULTURE: 45_000,
        LandUse.SPECIAL: 75_000,
    },
    default=0,
)
DEFAULT_PROFIT_SHARE_OPS: Final[dict[str, float]] = _value_map(
    {
        LandUse.BUSINESS: 0.18,
        LandUse.TRANSPORT: 0.10,
        LandUse.AGRICULTURE: 0.08,
        LandUse.SPECIAL: 0.15,
        LandUse.RESIDENTIAL: 0.00,
    },
    default=0.12,
)
DEFAULT_CAPEX_CAPITALIZABLE_SHARE: Final[dict[str, float]] = _value_map(
    {
        LandUse.BUSINESS: 0.95,
        LandUse.RESIDENTIAL: 0.95,
        LandUse.TRANSPORT: 0.90,
        LandUse.AGRICULTURE: 0.90,
        LandUse.SPECIAL: 0.95,
    },
    default=0.95,
)
DEFAULT_AMORTIZATION_RATES: Final[dict[str, float]] = _value_map(
    {
        LandUse.BUSINESS: 0.03,
        LandUse.RESIDENTIAL: 0.03,
        LandUse.TRANSPORT: 0.06,
        LandUse.AGRICULTURE: 0.05,
        LandUse.SPECIAL: 0.04,
    },
    default=0.03,
)
DEFAULT_BUILD_WAGE_SHARE: Final[float] = 0.25
DEFAULT_BUILD_PROFIT_MARGIN: Final[float] = 0.05
FALLBACK_CAPEX_CAPITALIZABLE_SHARE: Final[float] = 1.0
DEFAULT_EMPLOYMENT_SHARE: Final[float] = 0.62


DEFAULT_SER_PARAMETERS: Final[dict[str, object]] = {
    "employment_share": DEFAULT_EMPLOYMENT_SHARE,
    "avg_wage_base": DEFAULT_AVG_WAGE_BASE,
    "tax_rates": dict(DEFAULT_TAX_RATES),
    "va_coeff_build": _value_map(
        {
            LandUse.BUSINESS: 0.50,
            LandUse.RESIDENTIAL: 0.45,
            LandUse.TRANSPORT: 0.55,
            LandUse.AGRICULTURE: 0.45,
            LandUse.SPECIAL: 0.50,
        },
        default=0.50,
    ),
    "va_per_m2_ops": DEFAULT_VA_PER_M2_OPS.copy(),
    "jobs_per_m2": DEFAULT_JOBS_PER_M2.copy(),
    "wage_by_use": DEFAULT_WAGE_BY_USE.copy(),
    "profit_share_ops": DEFAULT_PROFIT_SHARE_OPS.copy(),
    "capex_capitalizable_share": DEFAULT_CAPEX_CAPITALIZABLE_SHARE.copy(),
    "amortization_rates": DEFAULT_AMORTIZATION_RATES.copy(),
    "build_wage_share": DEFAULT_BUILD_WAGE_SHARE,
    "build_profit_margin": DEFAULT_BUILD_PROFIT_MARGIN,
}
