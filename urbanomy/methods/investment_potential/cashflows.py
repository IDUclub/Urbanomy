"""Cash-flow calculations for individual and mixed-use development scenarios."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from blocksnet.enums import LandUse

from urbanomy.methods.investment_potential.input import LAND_USE_SHARE_COLUMNS

from .constants import DEFAULT_BENCHMARKS_RU
from .utils_metrics import economic_index, irr, make_cashflow, npv, payback_period, quantize


@dataclass(frozen=True)
class InvestmentMetricsResult:
    """Calculated economic metrics for a cash-flow sequence."""

    npv: float
    irr: float
    profitability_index: float
    payback_years: float
    economic_index: float

    @classmethod
    def from_cashflow(
        cls,
        cashflow: Sequence[float],
        discount_rate: float,
        *,
        compute_ei: bool = True,
    ) -> "InvestmentMetricsResult":
        """Create an instance from a cash-flow sequence.

        Parameters
        ----------
        cashflow : Sequence[float]
            Ordered cash-flow values per period, including the initial
            investment.
        discount_rate : float
            Discount rate per period expressed as a decimal fraction.

        Returns
        -------
        InvestmentMetricsResult
            New instance populated with derived financial indicators.
        """
        cashflow = list(cashflow or [])

        raw_npv = npv(discount_rate, cashflow)
        quantized = quantize(raw_npv)
        npv_value = float(quantized) if quantized is not None else np.nan

        irr_raw = irr(cashflow)
        irr_value = float(irr_raw) if irr_raw is not None else np.nan

        pv_inflows = 0.0
        pv_outflows = 0.0
        for period, cf in enumerate(cashflow):
            discounted = float(cf) / (1.0 + discount_rate) ** period
            if discounted > 0:
                pv_inflows += discounted
            elif discounted < 0:
                pv_outflows += -discounted
        pi_value = (pv_inflows / pv_outflows) if pv_outflows > 0 else np.nan

        payback_raw = payback_period(discount_rate, cashflow)
        payback_value = float(payback_raw) if payback_raw is not None else np.nan

        ei_value = economic_index(raw_npv, irr_raw, discount_rate) if compute_ei else math.nan

        return cls(
            npv=npv_value,
            irr=irr_value,
            profitability_index=pi_value,
            payback_years=payback_value,
            economic_index=ei_value,
        )


@dataclass(frozen=True)
class PreparedProfile:
    """Resolved profile parameters ready for cash-flow computation."""

    params: dict[str, Any]
    land_area: float
    built_area: float
    gross_floor_area: float
    land_cost_total: float


@dataclass(frozen=True)
class RowComputation:
    """Computed financial metrics for an individual polygon."""

    index: Any
    land_use: str
    land_area: float
    built_area: float
    land_cost_total: float
    demolition_cost: float
    construction_cost: float
    investment_need: float
    cashflow: list[float]
    metrics: InvestmentMetricsResult


class _CashflowCalculator:
    """Calculate cash flows and metrics independently of table/report assembly."""

    def __init__(self, benchmarks):
        self._benchmarks_enum = self._normalise_benchmarks(benchmarks or DEFAULT_BENCHMARKS_RU)

    @staticmethod
    def _coerce_land_use(value: str | LandUse) -> LandUse:
        if isinstance(value, LandUse):
            return value
        try:
            return LandUse(str(value))
        except ValueError as exc:
            text = str(value).strip()
            upper = text.upper()
            if upper.startswith("LANDUSE."):
                upper = upper.split(".", 1)[1]
            if upper in LandUse.__members__:
                return LandUse[upper]
            raise KeyError(f"Unknown land-use '{value}'") from exc

    def _normalise_benchmarks(
        self,
        benchmarks: Mapping[str | LandUse, Mapping[str, Any]] | None,
    ) -> dict[LandUse, dict[str, Any]]:
        """Convert benchmark keys to LandUse enum instances."""
        normalised: dict[LandUse, dict[str, Any]] = {}
        for key, profile in (benchmarks or {}).items():
            land_use = self._coerce_land_use(key)
            if not isinstance(profile, Mapping):
                raise ValueError(f"Benchmark profile for '{land_use.value}' must be a mapping")
            normalised[land_use] = dict(profile)
        return normalised

    @staticmethod
    def _to_float(value: Any) -> float:
        """Convert arbitrary input to ``float`` returning ``nan`` on failure.

        Parameters
        ----------
        value : Any
            Value to convert.

        Returns
        -------
        float
            Converted float or ``nan`` if conversion fails.
        """
        try:
            return float(value)
        except (TypeError, ValueError):
            return math.nan

    def _extract_built_area_components(
        self,
        row: pd.Series,
    ) -> tuple[float, float, float]:
        """Read living/non-living areas and derive total built area.

        When both explicit components are present, they take precedence over
        ``build_floor_area`` because they provide the semantic split required
        for mixed-use allocation. When only one component is known, the total
        falls back to ``build_floor_area`` so the residual can be distributed
        across the remaining land uses.
        """
        living = self._to_float(row.get("living_area"))
        if not np.isfinite(living) or living <= 0:
            living = math.nan

        non_living = self._to_float(row.get("non_living_area"))
        if not np.isfinite(non_living) or non_living <= 0:
            non_living = math.nan

        known_components = [value for value in (living, non_living) if np.isfinite(value) and value > 0]
        components_total = float(sum(known_components)) if known_components else math.nan

        build_floor_area = self._to_float(row.get("build_floor_area"))
        if not np.isfinite(build_floor_area) or build_floor_area <= 0:
            build_floor_area = math.nan

        if len(known_components) == 2:
            total_built_area = components_total
        elif np.isfinite(build_floor_area):
            total_built_area = build_floor_area
        else:
            total_built_area = components_total

        if not np.isfinite(total_built_area) or total_built_area <= 0:
            total_built_area = math.nan

        return living, non_living, total_built_area

    @staticmethod
    def _distribute_by_weights(
        total_amount: float,
        weights: Mapping[LandUse, float],
    ) -> dict[LandUse, float]:
        """Distribute a positive amount proportionally to positive weights."""
        if not np.isfinite(total_amount) or total_amount <= 0:
            return {}

        valid_weights = {key: float(value) for key, value in weights.items() if np.isfinite(value) and value > 0}
        total_weight = float(sum(valid_weights.values()))
        if total_weight <= 0:
            return {}

        return {key: total_amount * (value / total_weight) for key, value in valid_weights.items()}

    def _allocate_built_area_by_zone(
        self,
        row: pd.Series,
        zone_areas: Mapping[LandUse, float],
    ) -> tuple[dict[LandUse, float], float]:
        """Allocate built area to land uses using living/non-living semantics.

        Residential zones receive ``living_area`` first. Non-residential zones
        receive ``non_living_area`` first. Any residual from ``build_floor_area``
        is allocated to the complementary group when possible, otherwise across
        all zones proportionally to land-use area.
        """
        zone_weights = {lu: float(area) for lu, area in zone_areas.items() if np.isfinite(area) and area > 0}
        if not zone_weights:
            return {}, math.nan

        living, non_living, total_built_area = self._extract_built_area_components(row)
        allocations: dict[LandUse, float] = {}
        allocated_total = 0.0

        residential_zones = {lu: area for lu, area in zone_weights.items() if lu == LandUse.RESIDENTIAL}
        non_residential_zones = {lu: area for lu, area in zone_weights.items() if lu != LandUse.RESIDENTIAL}

        component_targets = (
            (living, residential_zones),
            (non_living, non_residential_zones),
        )
        for component_area, targets in component_targets:
            distributed = self._distribute_by_weights(component_area, targets)
            if not distributed:
                continue
            for zone_lu, zone_built_area in distributed.items():
                allocations[zone_lu] = allocations.get(zone_lu, 0.0) + zone_built_area
            allocated_total += float(sum(distributed.values()))

        residual_built_area = total_built_area - allocated_total if np.isfinite(total_built_area) else math.nan
        if np.isfinite(residual_built_area) and residual_built_area > 0:
            residual_targets = zone_weights
            if np.isfinite(living) and living > 0 and not (np.isfinite(non_living) and non_living > 0):
                residual_targets = non_residential_zones or zone_weights
            elif np.isfinite(non_living) and non_living > 0 and not (np.isfinite(living) and living > 0):
                residual_targets = residential_zones or zone_weights

            distributed = self._distribute_by_weights(residual_built_area, residual_targets)
            for zone_lu, zone_built_area in distributed.items():
                allocations[zone_lu] = allocations.get(zone_lu, 0.0) + zone_built_area

        return allocations, total_built_area

    def _prepare_profile(self, row: pd.Series, base_profile: dict[str, Any]) -> PreparedProfile:
        """Combine a row with a benchmark profile for cash-flow modelling.

        Parameters
        ----------
        row : pandas.Series
            Row with polygon attributes including geometry, areas and costs.
        base_profile : dict[str, Any]
            Benchmark parameters for the polygon's land-use type.

        Returns
        -------
        PreparedProfile
            Structured values ready for cash-flow generation.

        Raises
        ------
        ValueError
            If a valid land area cannot be derived from inputs.
        """
        params = dict(base_profile or {})

        land_area = self._to_float(row.get("site_area"))
        if not np.isfinite(land_area) or land_area <= 0:
            geom = row.get("geometry")
            land_area = float(getattr(geom, "area", math.nan))
        if not np.isfinite(land_area) or land_area <= 0:
            raise ValueError(f"Polygon {row.name} has no valid land area")

        land_cost_total = self._to_float(row.get("land_value"))
        if np.isfinite(land_cost_total) and land_area > 0:
            params["land_cost"] = land_cost_total / land_area

        _, _, built_area = self._extract_built_area_components(row)

        if np.isfinite(built_area) and built_area > 0:
            params["built_area"] = built_area
        else:
            params.pop("built_area", None)
            built_area = math.nan

        if np.isfinite(built_area):
            gross_floor_area = built_area
        else:
            density = self._to_float(params.get("density"))
            gross_floor_area = land_area * density if np.isfinite(density) and density > 0 else math.nan

        return PreparedProfile(
            params=params,
            land_area=land_area,
            built_area=built_area,
            gross_floor_area=gross_floor_area,
            land_cost_total=land_cost_total,
        )

    def _land_use_areas_from_row(self, row: pd.Series) -> dict[LandUse, float]:
        """Extract positive per-land-use areas from a row.

        Assumes zone columns in ``LAND_USE_SHARE_COLUMNS`` are already expressed
        in square meters by :func:`prepare_investment_input`.
        """
        result: dict[LandUse, float] = {}
        for col in LAND_USE_SHARE_COLUMNS:
            if col not in row.index:
                continue
            area_val = self._to_float(row.get(col))
            if not np.isfinite(area_val) or area_val <= 0:
                continue
            try:
                lu = self._coerce_land_use(col)
            except Exception:
                continue
            if lu not in self._benchmarks_enum:
                continue
            result[lu] = float(area_val)
        return result

    def calculate_row(
        self,
        idx: Any,
        row: pd.Series,
        default_discount_rate: float,
    ) -> RowComputation:
        """Evaluate investment metrics for a single polygon row.

        Parameters
        ----------
        idx : Any
            Index label of the polygon.
        row : pandas.Series
            Polygon attributes enriched by prepared investment input.
        default_discount_rate : float
            Discount rate to use when the benchmark profile omits it.

        Returns
        -------
        RowComputation
            Computed metrics plus intermediate values for the polygon.

        Raises
        ------
        KeyError
            If the polygon's land-use lacks benchmark configuration.
        """
        land_use_raw = row.get("land_use")
        land_use_enum = self._coerce_land_use(land_use_raw)
        if land_use_enum not in self._benchmarks_enum:
            raise KeyError(f"No benchmark settings for land_use='{land_use_enum.value}'")

        zone_areas = self._land_use_areas_from_row(row)
        if not zone_areas:
            zone_areas = {land_use_enum: self._to_float(row.get("site_area"))}
        zone_areas = {lu: area for lu, area in zone_areas.items() if np.isfinite(area) and area > 0}
        if not zone_areas:
            profile = self._prepare_profile(row, self._benchmarks_enum[land_use_enum])
            zone_areas = {land_use_enum: profile.land_area}
        total_zone_area = float(sum(zone_areas.values()))

        land_cost_total = self._to_float(row.get("land_value"))
        land_cost_per_area = (
            land_cost_total / total_zone_area if np.isfinite(land_cost_total) and total_zone_area > 0 else math.nan
        )

        zone_built_areas, total_built_area = self._allocate_built_area_by_zone(row, zone_areas)

        old_build_floor_area = self._to_float(row.get("build_floor_area_before", row.get("build_floor_area")))
        if not np.isfinite(old_build_floor_area) or old_build_floor_area < 0:
            old_build_floor_area = 0.0
        demolition_land_use_raw = row.get("land_use_before")
        if demolition_land_use_raw is None or str(demolition_land_use_raw).strip() == "":
            demolition_land_use_raw = land_use_raw
        demolition_land_use_enum: LandUse | None = None
        try:
            demolition_land_use_enum = self._coerce_land_use(demolition_land_use_raw)
        except Exception:
            demolition_land_use_enum = None
        fallback_demo_lu = next(iter(zone_areas.keys()))
        demolition_profile = (
            self._benchmarks_enum[demolition_land_use_enum]
            if demolition_land_use_enum in self._benchmarks_enum
            else self._benchmarks_enum[fallback_demo_lu]
        )
        cost_demolition_unit = self._to_float(demolition_profile.get("cost_demolition"))
        demolition_cost_total = (
            old_build_floor_area * cost_demolition_unit
            if np.isfinite(cost_demolition_unit) and cost_demolition_unit >= 0
            else 0.0
        )

        zone_cashflows: list[list[float]] = []
        zone_discount_rates: list[float] = []
        construction_cost_total = 0.0
        built_area_total = 0.0
        has_construction = False
        has_built_area = False

        for zone_lu, zone_area in zone_areas.items():
            zone_profile = dict(self._benchmarks_enum[zone_lu])
            if np.isfinite(land_cost_per_area) and land_cost_per_area >= 0:
                zone_profile["land_cost"] = land_cost_per_area
            zone_built_area = zone_built_areas.get(zone_lu, math.nan)
            if np.isfinite(zone_built_area) and zone_built_area > 0:
                zone_profile["built_area"] = zone_built_area
                built_area_total += zone_built_area
                has_built_area = True

            zone_cf = make_cashflow(zone_lu.value, zone_area, zone_profile)
            zone_cashflows.append(zone_cf)

            zone_discount = self._to_float(zone_profile.get("discount_rate", default_discount_rate))
            zone_discount_rates.append(zone_discount if np.isfinite(zone_discount) else float(default_discount_rate))

            zone_cost_build = self._to_float(zone_profile.get("cost_build"))
            zone_gfa = self._to_float(zone_profile.get("built_area"))
            if not np.isfinite(zone_gfa) or zone_gfa <= 0:
                zone_density = self._to_float(zone_profile.get("density"))
                if np.isfinite(zone_density) and zone_density > 0:
                    zone_gfa = zone_area * zone_density
                else:
                    zone_gfa = zone_area
            if np.isfinite(zone_gfa) and zone_gfa > 0 and np.isfinite(zone_cost_build):
                construction_cost_total += zone_gfa * zone_cost_build
                has_construction = True

        cashflow = self._aggregate_cashflows(zone_cashflows)
        if cashflow:
            # Demolition is paid before construction, alongside the initial investment.
            cashflow[0] -= demolition_cost_total
        if zone_discount_rates and total_zone_area > 0:
            weighted_discount_rate = float(np.average(zone_discount_rates, weights=list(zone_areas.values())))
        else:
            weighted_discount_rate = float(default_discount_rate)
        metrics = InvestmentMetricsResult.from_cashflow(cashflow, weighted_discount_rate)

        construction_cost = construction_cost_total if has_construction else math.nan
        demolition_cost = float(demolition_cost_total)
        finite_capex = [value for value in (land_cost_total, demolition_cost, construction_cost) if np.isfinite(value)]
        investment_need = float(sum(finite_capex)) if finite_capex else math.nan
        return RowComputation(
            index=idx,
            land_use=land_use_enum.value if len(zone_areas) == 1 else "mixed",
            land_area=total_zone_area,
            built_area=built_area_total if has_built_area else total_built_area,
            land_cost_total=land_cost_total,
            demolition_cost=demolition_cost,
            construction_cost=construction_cost,
            investment_need=investment_need,
            cashflow=list(cashflow),
            metrics=metrics,
        )

    @staticmethod
    def _aggregate_cashflows(cashflows: Sequence[Sequence[float]]) -> list[float]:
        """Aggregate multiple cash-flow sequences period by period.

        Parameters
        ----------
        cashflows : sequence of sequence of float
            Cash-flow lists aligned per polygon.

        Returns
        -------
        list[float]
            Aggregate cash flow summing each period across sequences.
        """
        sequences = [list(cf) for cf in cashflows if cf]
        if not sequences:
            return []
        max_len = max(len(cf) for cf in sequences)
        return [sum(cf[i] if i < len(cf) else 0.0 for cf in sequences) for i in range(max_len)]
