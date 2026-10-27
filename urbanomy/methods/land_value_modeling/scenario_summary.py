"""Numerical scenario summaries and their console presentation."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
from blocksnet.enums import LandUse
from geopandas import GeoDataFrame

from .constants import BlockColumn


def _summarise_changes(gdf: GeoDataFrame) -> Dict[str, float]:
    """Compute aggregate before/after price statistics for a subset of blocks.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Dataset containing ``land_value_before`` and ``land_value_after`` columns.

    Returns
    -------
    dict[str, float]
        Aggregated sums, deltas, and count information.
    """
    if gdf is None or gdf.empty:
        return {
            "sum_before": 0.0,
            "sum_after": 0.0,
            "delta": 0.0,
            "delta_pct": np.nan,
            "count": 0,
        }

    land_value_before = float(gdf["land_value_before"].sum())
    land_value_after = float(gdf["land_value_after"].sum())
    delta = land_value_after - land_value_before
    delta_pct = (land_value_after / land_value_before - 1.0) * 100 if land_value_before > 0 else np.nan
    return {
        "sum_before": land_value_before,
        "sum_after": land_value_after,
        "delta": delta,
        "delta_pct": delta_pct,
        "count": int(len(gdf)),
    }


def _format_land_use(value: Any) -> str:
    """Render land-use labels for plotting."""
    if isinstance(value, LandUse):
        return value.value
    if value is None:
        return "missing"
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return "missing"
    try:
        return LandUse(text).value
    except ValueError:
        pass
    upper = text.upper()
    if upper.startswith("LANDUSE."):
        upper = upper.split(".", 1)[1]
    return upper


def _print_summary(
    gdf: GeoDataFrame,
    summary: Dict[str, float],
    *,
    eps: float,
    total_summary: Dict[str, float] | None = None,
    total_count: int | None = None,
    target_idx: int | None = None,
    project_column: str = BlockColumn.IS_PROJECT.value,
    print_quarter_stats: bool = True,
) -> None:
    """Print human-readable summaries of scenario-induced price changes.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Blocks whose changes are being highlighted.
    summary : dict[str, float]
        Aggregated metrics for ``gdf`` from :func:`_summarise_changes`.
    eps : float
        Price-change threshold used to filter ``gdf``.
    total_summary : dict[str, float], optional
        Aggregated metrics for the entire buffer.
    total_count : int, optional
        Total number of buffered blocks.
    target_idx : int, optional
        Identifier of the modified block to highlight separately.
    project_column : str, optional
        Column indicating whether a block belongs to the project area.
    """

    def _print_summary_block(title: str, stats: Dict[str, float] | None, *, count_override: int | None = None) -> None:
        print(f"\n{title}:")
        if not stats or stats["count"] == 0:
            print(" Нет данных.")
            return
        print(" • Сумма до:   ", _fmt_rub(stats["sum_before"]))
        print(" • Сумма после:", _fmt_rub(stats["sum_after"]))
        print(" • Изм., ₽:    ", _fmt_rub(stats["delta"], signed=True))
        print(" • Изм., %:    ", _fmt_pct(stats["delta_pct"]))
        print(" • Территорий: ", count_override if count_override is not None else stats["count"])

    def _print_row(idx, row) -> None:
        land_use = _format_land_use(row.get(land_use_key))
        print(
            f"{idx} | {land_use} | {_fmt_rub(row['land_value_before'])} → "
            f"{_fmt_rub(row['land_value_after'])} | {_fmt_rub(row['d_rub'], signed=True)} | "
            f"{_fmt_pct(row['d_pct'])}"
        )

    land_use_key = BlockColumn.LAND_USE.value
    project_mask = pd.Series(False, index=gdf.index, dtype=bool)

    if len(gdf):
        project_mask = pd.Series(gdf.get(project_column, False), index=gdf.index).fillna(False).astype(bool)

    if print_quarter_stats:
        threshold_note = f" (|Δ₽| > {eps:g})" if eps > 0 else ""
        print(f"\nСтатистика по кварталам в контексте{threshold_note}:")

        if len(gdf):
            print("\nИзменяемый квартал:")
            print("index | land_use | до (₽) → после (₽) | Δ₽ | Δ%")
            if target_idx is not None and target_idx in gdf.index:
                _print_row(target_idx, gdf.loc[target_idx])
            else:
                print("—")

            project_rows = gdf[project_mask].drop(index=target_idx, errors="ignore")
            context_rows = gdf[~project_mask].drop(index=target_idx, errors="ignore")

            print("\nКварталы в границах проекта:")
            if len(project_rows):
                print("index | land_use | до (₽) → после (₽) | Δ₽ | Δ%")
                for idx, row in project_rows.iterrows():
                    _print_row(idx, row)
            else:
                print("Нет кварталов проекта с изменениями выше порога.")

            print("\nКварталы в контексте:")
            if len(context_rows):
                print("index | land_use | до (₽) → после (₽) | Δ₽ | Δ%")
                for idx, row in context_rows.iterrows():
                    _print_row(idx, row)
            else:
                print("Нет кварталов контекста с изменениями выше порога.")
        else:
            print("Нет кварталов, у которых стоимость изменилась выше порога.")

    target_rows = gdf.loc[[target_idx]] if target_idx is not None and target_idx in gdf.index else gdf.iloc[0:0]
    project_rows = gdf[project_mask].drop(index=target_idx, errors="ignore")
    context_rows = gdf[~project_mask].drop(index=target_idx, errors="ignore")

    target_summary = _summarise_changes(target_rows)
    project_summary = _summarise_changes(project_rows)
    context_summary = _summarise_changes(context_rows)

    _print_summary_block("Сводка по изменяемому кварталу", target_summary)
    _print_summary_block("Сводка по проектной территории (изменения)", project_summary)
    _print_summary_block("Сводка по контексту (изменения)", context_summary)
    _print_summary_block("Сводка по всем изменениям", summary)


def _fmt_rub(value: float, *, signed: bool = False, digits: int = 0) -> str:
    """Format numeric values as Russian roubles with thin-space separators.

    Parameters
    ----------
    value : float
        Numeric value to format.
    signed : bool, optional
        Display the sign explicitly when ``True``.
    digits : int, optional
        Number of decimal digits to display.

    Returns
    -------
    str
        Formatted currency string or an em dash if value is not finite.
    """
    try:
        value = float(value)
        if not np.isfinite(value):
            return "—"
        formatted = f"{value:+,.{digits}f}" if signed else f"{value:,.{digits}f}"
        return formatted.replace(",", " ") + " ₽"
    except Exception:
        return "—"


def _fmt_pct(value: float, digits: int = 1) -> str:
    """Format percentage values with an explicit sign when finite.

    Parameters
    ----------
    value : float
        Percentage value to display.
    digits : int, optional
        Number of decimal digits to display.

    Returns
    -------
    str
        Signed percentage string or an em dash if value is not finite.
    """
    try:
        value = float(value)
        if not np.isfinite(value):
            return "—"
        return f"{value:+.{digits}f}%"
    except Exception:
        return "—"
