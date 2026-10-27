"""Utilities for computing investment attractiveness metrics per polygon."""

from __future__ import annotations

import logging
from typing import Any, Mapping, Sequence

import geopandas as gpd
import numpy as np
import pandas as pd
from blocksnet.enums import LandUse

from urbanomy.methods.investment_potential.input import INVESTMENT_NUMERIC_COLUMNS, prepare_investment_input

from .cashflows import InvestmentMetricsResult, PreparedProfile, RowComputation, _CashflowCalculator
from .constants import DEFAULT_DISCOUNT_RATE, DEFAULT_ECON_METRIC, SUMMARY_COLUMNS


logger = logging.getLogger(__name__)


class InvestmentAttractivenessAnalyzer:
    """Prepare input tables, calculate per-polygon metrics, and assemble reports."""

    _aggregate_cashflows = staticmethod(_CashflowCalculator._aggregate_cashflows)

    def __init__(
        self,
        benchmarks: Mapping[str | LandUse, Mapping[str, Any]] | None = None,
        weights_dict: Mapping[str | LandUse, Sequence[float]] | None = None,
        econ_metric: str = DEFAULT_ECON_METRIC,
        discount_rate: float | None = None,
    ) -> None:
        """Initialise the analyzer with reference profiles.

        Parameters
        ----------
        benchmarks : Mapping[str | LandUse, Mapping[str, Any]] or None, optional
            Mapping from land-use codes or ``LandUse`` enums to benchmark profiles describing
            profitability assumptions. If ``None``, uses
            :data:`DEFAULT_BENCHMARKS_RU`.
        weights_dict : Mapping[str | LandUse, Sequence[float]] or None, optional
            Reserved for backward compatibility; ignored.
        econ_metric : str, optional
            Economic metric to emphasise when normalising (``"EI"`` by default).
        discount_rate : float or None, optional
            Discount rate used when the benchmark profile does not specify one.
            If ``None``, ``DEFAULT_DISCOUNT_RATE`` is used.
        """
        self._cashflows = _CashflowCalculator(benchmarks)
        self._benchmarks_enum = self._cashflows._benchmarks_enum
        self.benchmarks = {land_use.value: dict(profile) for land_use, profile in self._benchmarks_enum.items()}
        _ = weights_dict
        self.metric = econ_metric.upper()
        self.discount_rate = discount_rate if discount_rate is not None else DEFAULT_DISCOUNT_RATE

    @staticmethod
    def _round_clean(values: pd.Series | np.ndarray, decimals: int = 2) -> pd.Series:
        """Round numerical data and suppress near-zero artefacts.

        Parameters
        ----------
        values : pandas.Series or numpy.ndarray
            Values to round.
        decimals : int, optional
            Number of decimal places (default is ``2``).

        Returns
        -------
        pandas.Series
            Rounded values with tiny numbers coerced to ``0.0``.
        """
        serie = pd.Series(values, copy=True, dtype=float)
        serie = serie.round(decimals)
        tol = 10 ** (-decimals)
        serie[np.isclose(serie, 0.0, atol=tol, rtol=0.0)] = 0.0
        return serie

    @staticmethod
    def _coerce_numeric_columns(gdf: pd.DataFrame, columns: Sequence[str]) -> None:
        """Cast selected DataFrame columns to numeric values in-place.

        Parameters
        ----------
        gdf : pandas.DataFrame
            DataFrame whose columns are to be converted.
        columns : Sequence[str]
            Column names that should contain numeric data.
        """
        for col in columns:
            if col in gdf.columns:
                gdf[col] = pd.to_numeric(gdf[col], errors="coerce")

    def calculate_investment_metrics(
        self,
        gdf: gpd.GeoDataFrame | pd.DataFrame,
        discount_rate: float | None = None,
        show_project_totals: bool = True,
        show_warning: bool = True,
    ) -> pd.DataFrame:
        """Return a per-land-use summary of investment metrics.

        Parameters
        ----------
        gdf : geopandas.GeoDataFrame or pandas.DataFrame
            Input dataset containing scenario polygons or the output of
            :func:`prepare_investment_input`.
        discount_rate : float or None, optional
            Discount rate to use when benchmark profiles omit one. Falls back to
            the analyzer's configured rate (and ultimately
            ``DEFAULT_DISCOUNT_RATE``) when ``None``.
        show_project_totals : bool, optional
            Whether to print aggregated project totals and project-level metrics.
            Defaults to ``True``.
        show_warning : bool, optional
            Whether to emit warning logs (for example when rows have missing
            ``land_use``). Defaults to ``True``.

        Returns
        -------
        pandas.DataFrame
            Summary table with metrics for each polygon. When ``gdf`` is empty,
            returns an empty DataFrame with ``SUMMARY_COLUMNS``.

        Raises
        ------
        ValueError
            If the requested economic metric is missing from the prepared data.
        """
        if gdf.empty:
            print("No polygons provided; summary is empty.")
            return pd.DataFrame(columns=SUMMARY_COLUMNS)

        needs_preparation = isinstance(gdf, gpd.GeoDataFrame) or "geometry" in gdf.columns
        working = prepare_investment_input(gdf, show_warning=show_warning) if needs_preparation else gdf.copy()
        if "land_use" not in working.columns:
            raise ValueError("Expected 'land_use' column in prepared data.")

        land_use_series = working["land_use"].astype("string")
        normalized_land_use = land_use_series.str.replace(r"^LandUse\.", "", regex=True).str.strip().str.lower()
        missing_land_use = land_use_series.isna() | normalized_land_use.isin({"", "none"})
        missing_count = int(missing_land_use.sum())
        if missing_count and show_warning:
            total = int(len(working))
            logger.warning(
                "calculate_investment_metrics: %s из %s кварталов без land-use; "
                "пропускаем их в расчётах, но оставляем в итоговом отчёте.",
                missing_count,
                total,
            )
        valid_mask = ~missing_land_use

        base_discount_rate = float(discount_rate) if discount_rate is not None else float(self.discount_rate)
        self._coerce_numeric_columns(working, INVESTMENT_NUMERIC_COLUMNS)

        if "land_value" not in working.columns:
            working["land_value"] = np.nan
        working["land_value"] = pd.to_numeric(working["land_value"], errors="coerce")

        row_results = [
            self._cashflows.calculate_row(idx, row, base_discount_rate)
            for idx, row in working.loc[valid_mask].iterrows()
        ]

        raw_to_final = {
            "ECON_NPV": "NPV",
            "ECON_IRR": "IRR",
            "ECON_PI": "PI",
            "ECON_PP_years": "PP_years",
            "ECON_EI": "EI",
        }
        profile_fields = {
            "land_area": "land_area",
            "built_area": "built_area",
            "land_value": "land_cost_total",
            "demolition_cost": "demolition_cost",
            "construction_cost": "construction_cost",
            "investment_need": "investment_need",
        }

        if row_results:
            metrics_records: list[dict[str, Any]] = []
            for result in row_results:
                record: dict[str, Any] = {"_index": result.index}
                record.update(
                    {
                        raw: getattr(result.metrics, attr)
                        for raw, attr in {
                            "ECON_NPV": "npv",
                            "ECON_IRR": "irr",
                            "ECON_PI": "profitability_index",
                            "ECON_PP_years": "payback_years",
                            "ECON_EI": "economic_index",
                        }.items()
                    }
                )
                record.update({column: getattr(result, attr) for column, attr in profile_fields.items()})
                metrics_records.append(record)

            metrics_df = pd.DataFrame.from_records(metrics_records).set_index("_index")
            # Preserve existing columns while enriching with calculated metrics.
            missing_cols = [col for col in metrics_df.columns if col not in working.columns]
            for column in missing_cols:
                working[column] = np.nan
            working.update(metrics_df)
        else:
            for column in (*raw_to_final.keys(), *profile_fields.keys()):
                if column not in working.columns:
                    working[column] = np.nan

        for raw_col, final_col in raw_to_final.items():
            working[final_col] = working[raw_col]

        summary = working.reindex(columns=SUMMARY_COLUMNS).copy()
        columns_to_null = [col for col in ("NPV", "IRR", "PI", "PP_years", "EI") if col in summary.columns]
        if columns_to_null:
            summary.loc[~valid_mask, columns_to_null] = np.nan

        if show_project_totals:
            project_cf = self._aggregate_cashflows(result.cashflow for result in row_results)
            project_metrics = (
                InvestmentMetricsResult.from_cashflow(
                    project_cf,
                    base_discount_rate,
                    compute_ei=False,
                )
                if project_cf
                else None
            )

            land_area_series = (
                summary["land_area"] if "land_area" in summary.columns else pd.Series(0.0, index=summary.index)
            )
            built_area_series = (
                summary["built_area"] if "built_area" in summary.columns else pd.Series(0.0, index=summary.index)
            )
            land_value_series = (
                summary["land_value"] if "land_value" in summary.columns else pd.Series(0.0, index=summary.index)
            )
            demolition_cost_series = (
                summary["demolition_cost"]
                if "demolition_cost" in summary.columns
                else pd.Series(0.0, index=summary.index)
            )
            construction_cost_series = (
                summary["construction_cost"]
                if "construction_cost" in summary.columns
                else pd.Series(0.0, index=summary.index)
            )
            investment_need_series = (
                summary["investment_need"]
                if "investment_need" in summary.columns
                else pd.Series(0.0, index=summary.index)
            )
            total_area = land_area_series.loc[valid_mask].sum(skipna=True)
            total_built = built_area_series.loc[valid_mask].sum(skipna=True)
            total_land_value = land_value_series.loc[valid_mask].sum(skipna=True)
            total_demolition_cost = demolition_cost_series.loc[valid_mask].sum(skipna=True)
            total_construction_cost = construction_cost_series.loc[valid_mask].sum(skipna=True)
            total_investment_need = investment_need_series.loc[valid_mask].sum(skipna=True)

        currency_columns = {"land_value", "demolition_cost", "construction_cost", "investment_need", "NPV"}
        numeric_cols = summary.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            summary[col] = self._round_clean(summary[col], decimals=2)
            if col in currency_columns:
                summary[col] = summary[col].astype(float)

        if show_project_totals:
            print("Project totals:")
            print(f" • Land area:          {total_area:,.2f}")
            print(f" • Built area:         {total_built:,.2f}")
            print(f" • Land value:         {total_land_value:,.2f}")
            print(f" • Demolition cost:    {total_demolition_cost:,.2f}")
            print(f" • Construction cost:  {total_construction_cost:,.2f}")
            print(f" • Investment need:    {total_investment_need:,.2f}")
            if project_metrics is not None:
                print(f" • Project NPV:        {project_metrics.npv:,.2f}")
                print(f" • Project IRR:        {project_metrics.irr:,.2f}")
                print(f" • Project PI:         {project_metrics.profitability_index:,.2f}")
                print(f" • Project PP (yrs):   {project_metrics.payback_years:,.2f}")
            else:
                print(" • Project metrics:    not available (no cashflows)")

        return summary


def calculate_investment_metrics(
    gdf: gpd.GeoDataFrame | pd.DataFrame,
    benchmarks: Mapping[str | LandUse, Mapping[str, Any]] | None = None,
    *,
    weights_dict: Mapping[str | LandUse, Sequence[float]] | None = None,
    econ_metric: str = DEFAULT_ECON_METRIC,
    discount_rate: float | None = None,
    show_project_totals: bool = True,
    show_warning: bool = True,
) -> pd.DataFrame:
    """Convenience wrapper returning investment summary from raw scenario blocks.

    Accepts the original scenario GeoDataFrame (for example ``blocks_full_value``),
    prepares it internally with :func:`prepare_investment_input`, and returns the
    same summary table as :meth:`InvestmentAttractivenessAnalyzer.calculate_investment_metrics`.
    """
    analyzer = InvestmentAttractivenessAnalyzer(
        benchmarks=benchmarks,
        weights_dict=weights_dict,
        econ_metric=econ_metric,
        discount_rate=discount_rate,
    )
    return analyzer.calculate_investment_metrics(
        gdf,
        discount_rate=discount_rate,
        show_project_totals=show_project_totals,
        show_warning=show_warning,
    )
