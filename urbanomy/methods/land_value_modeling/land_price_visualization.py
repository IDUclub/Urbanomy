"""Visualization utilities for land price scenarios."""

from __future__ import annotations

from typing import Dict, Tuple

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from geopandas import GeoDataFrame

_FIGSIZE = (18, 14)


def _price_metric_label(price_column: str) -> str:
    """Return a human-friendly label for the selected price column."""
    column = price_column.lower()
    if column == "land_value":
        return "после застройки"
    if column == "land_value_before":
        return "до застройки"
    return price_column


def plot_land_price_maps(
    *,
    blocks_pred: GeoDataFrame,
    price_column: str = "land_value",
    log_price_column: str | None = None,
    area_column: str = "site_area",
    buffer_radius_m: float = 2000.0,
    show: bool = True,
    print_summary: bool = True,
    color_bounds: tuple[float, float] | None = None,
    quantile_bounds: tuple[float, float] = (0.05, 0.95),
) -> Dict[str, object]:
    """Plot land price maps for scenario and context blocks.

    Parameters
    ----------
    blocks_pred : geopandas.GeoDataFrame
        Dataset containing price predictions, block geometries, and the
        ``is_project`` flag. The DataFrame is copied internally to avoid mutating
        the original object.
    price_column : str, default='land_value'
        Name of the column with price predictions in the original scale. Pass
        ``"land_value_before"`` to visualise baseline prices transferred from
        historical blocks. If the column is missing but ``log_price_column`` is
        available, prices are derived by exponentiating the logarithmic predictions.
    log_price_column : str or None, default=None
        Column containing logarithmic prices. Used as a fallback to build
        ``price_column`` when it is absent and ``log_price_column`` is provided.
    area_column : str, default='site_area'
        Column with block areas in square metres. Rows where the value is not
        positive fall back to areas derived from geometry in planar metres.
    buffer_radius_m : float, default=2000.0
        Radius of the buffer (in metres) around the scenario footprint used to
        clip context blocks.
    show : bool, default=True
        Whether to display the generated figures using ``plt.show``.
    print_summary : bool, default=True
        Whether to print aggregate price statistics to stdout.
    color_bounds : tuple[float, float] or None, default=None
        Explicit lower and upper bounds (руб./сотка) for the colour scale. When
        ``None``, bounds are inferred from the data using ``quantile_bounds``.
    quantile_bounds : tuple[float, float], default=(0.05, 0.95)
        Quantile range applied to all available price columns to derive colour
        bounds when ``color_bounds`` is omitted.

    Notes
    -----
    When both ``land_value`` and ``land_value_before`` are present in
    ``blocks_pred``, colour bounds are inferred from the combined distribution
    so that before/after visualisations share the same scale.

    Returns
    -------
    dict
        Dictionary with keys ``'totals'``, ``'figures'`` and ``'color_bounds'``. ``totals`` stores
        aggregate prices for scenario, context, and all blocks. ``figures`` is
        a tuple of Matplotlib figures corresponding to the project and context
        maps. ``color_bounds`` contains the effective colour scale (руб./сотка)
        used for all plots.
    """
    blocks = blocks_pred.copy()

    low_q, high_q = quantile_bounds
    if not (0.0 <= low_q <= high_q <= 1.0):
        raise ValueError("quantile_bounds must be within [0, 1] and low <= high.")

    if price_column not in blocks.columns:
        if log_price_column and log_price_column in blocks.columns:
            blocks[price_column] = np.exp(blocks[log_price_column])
        else:
            raise ValueError(
                f"Neither '{price_column}' nor '{log_price_column}' is available in blocks_pred."
            )

    area_values = _resolve_area_values(blocks, area_column=area_column)
    price_values = blocks[price_column].astype(float).to_numpy(copy=True)
    price_per_sotka = np.full(len(blocks), np.nan, dtype=float)
    valid_area_mask = area_values > 0
    price_per_sotka[valid_area_mask] = price_values[valid_area_mask] / area_values[valid_area_mask] * 100.0
    per_sotka_column = f"{price_column}_per_sotka"
    blocks[per_sotka_column] = price_per_sotka

    candidate_columns = {price_column}
    candidate_columns.update(
        col for col in ("land_value", "land_value_before") if col in blocks.columns
    )
    candidate_per_sotka = [price_per_sotka]
    for col in candidate_columns:
        if col == price_column:
            continue
        other_values = blocks[col].astype(float).to_numpy(copy=True)
        other_per_sotka = np.full(len(blocks), np.nan, dtype=float)
        other_per_sotka[valid_area_mask] = (
            other_values[valid_area_mask] / area_values[valid_area_mask] * 100.0
        )
        candidate_per_sotka.append(other_per_sotka)

    scenario_subset, context_subset = _split_scenario_context(blocks)

    buffer_gdf = _make_buffer_m(scenario_subset, buffer_radius_m, target_crs=blocks.crs)
    context_clipped = gpd.clip(context_subset, buffer_gdf) if len(context_subset) else context_subset

    totals = {
        "scenario": float(np.nansum(scenario_subset[price_column])),
        "context": float(np.nansum(context_subset[price_column])),
        "all": float(np.nansum(blocks[price_column])),
    }

    metric_label = _price_metric_label(price_column)
    if print_summary:
        _print_totals(totals, metric_label)

    if color_bounds is not None:
        vmin, vmax = color_bounds
    else:
        finite_values = [arr[np.isfinite(arr)] for arr in candidate_per_sotka]
        flattened = (
            np.concatenate(finite_values)
            if any(len(vals) for vals in finite_values)
            else np.array([], dtype=float)
        )
        if flattened.size:
            vmin = float(np.nanquantile(flattened, low_q))
            vmax = float(np.nanquantile(flattened, high_q))
            if np.isfinite(vmin) and np.isfinite(vmax) and vmin == vmax:
                vmax = vmin + 1.0
        else:
            vmin = vmax = float("nan")

    figures: Tuple[plt.Figure, ...] = _plot_maps(
        blocks=blocks,
        scenario=scenario_subset,
        context=context_clipped,
        price_column=per_sotka_column,
        vmin=vmin,
        vmax=vmax,
        metric_label=metric_label,
        show=show,
    )

    return {"totals": totals, "figures": figures, "color_bounds": (vmin, vmax)}


def _resolve_area_values(blocks: GeoDataFrame, *, area_column: str) -> np.ndarray:
    """Return per-block areas in square metres with geometry fallback."""
    if blocks.empty:
        return np.array([], dtype=float)

    if area_column in blocks.columns:
        area_series = pd.to_numeric(blocks[area_column], errors="coerce")
    else:
        area_series = pd.Series(np.nan, index=blocks.index, dtype=float)

    area_values = area_series.to_numpy(copy=True)
    invalid_mask = ~np.isfinite(area_values) | (area_values <= 0)

    if invalid_mask.any():
        geometry_areas = _geometry_area_m2(blocks)
        area_values[invalid_mask] = geometry_areas[invalid_mask]

    return area_values


def _geometry_area_m2(blocks: GeoDataFrame) -> np.ndarray:
    """Compute block areas in square metres based on geometry."""
    if blocks.empty:
        return np.array([], dtype=float)
    if blocks.crs is None:
        raise ValueError("GeoDataFrame CRS is required to derive geometry-based areas.")

    if blocks.crs.is_geographic:
        utm_crs = blocks.estimate_utm_crs()
        if utm_crs is None:
            raise ValueError("Unable to estimate a projected CRS for area computation.")
        projected = blocks.to_crs(utm_crs)
        return projected.geometry.area.to_numpy()

    return blocks.geometry.area.to_numpy()


def _split_scenario_context(blocks: GeoDataFrame) -> Tuple[GeoDataFrame, GeoDataFrame]:
    """Separate scenario and context subsets, validating the `is_project` column."""
    if "is_project" not in blocks.columns:
        raise ValueError(
            "Column 'is_project' is required. Run LandDataPreparator or add the column before calling plot_land_price_maps."
        )

    scenario_mask = pd.Series(blocks["is_project"], index=blocks.index).fillna(False).astype(bool)
    if not scenario_mask.any():
        raise ValueError("No scenario blocks detected. Ensure column 'is_project' contains True values.")

    scenario_subset = blocks[scenario_mask].copy()
    context_subset = blocks[~scenario_mask].copy()
    return scenario_subset, context_subset


def _make_buffer_m(scenario: GeoDataFrame, radius_m: float, *, target_crs) -> GeoDataFrame:
    """Generate a buffer around the scenario footprint in metres.

    Parameters
    ----------
    scenario : geopandas.GeoDataFrame
        Scenario blocks whose union defines the buffer centre.
    radius_m : float
        Buffer radius in metres applied to the unified geometry.
    target_crs : Any
        Coordinate reference system for the returned geometry.

    Returns
    -------
    geopandas.GeoDataFrame
        Single-row GeoDataFrame representing the buffered scenario extent.
    """
    if scenario.empty:
        return gpd.GeoDataFrame(geometry=[], crs=target_crs)

    geometry_union = scenario.geometry.unary_union
    buffer_gdf = gpd.GeoDataFrame(geometry=[geometry_union], crs=scenario.crs)

    if buffer_gdf.crs and buffer_gdf.crs.is_geographic:
        projected = buffer_gdf.to_crs(buffer_gdf.estimate_utm_crs())
        projected["geometry"] = projected.buffer(radius_m)
        buffer_gdf = projected.to_crs(target_crs)
    else:
        buffer_gdf["geometry"] = buffer_gdf.buffer(radius_m)
        if buffer_gdf.crs != target_crs:
            buffer_gdf = buffer_gdf.to_crs(target_crs)

    return buffer_gdf


def _plot_maps(
    *,
    blocks: GeoDataFrame,
    scenario: GeoDataFrame,
    context: GeoDataFrame,
    price_column: str,
    vmin: float,
    vmax: float,
    metric_label: str,
    show: bool,
) -> Tuple[plt.Figure, ...]:
    """Create individual scenario and context price maps.

    Parameters
    ----------
    blocks : geopandas.GeoDataFrame
        Dataset containing price-per-sotka values.
    scenario : geopandas.GeoDataFrame
        Scenario subset used for highlighting.
    context : geopandas.GeoDataFrame
        Context blocks within the buffer.
    price_column : str
        Name of the column visualised via colour mapping.
    vmin, vmax : float
        Colour scale bounds shared across plots.
    metric_label : str
        Human-readable description of the price column.
    show : bool
        Whether to display the figures via ``plt.show``.

    Returns
    -------
    tuple[matplotlib.figure.Figure, ...]
        Figures for scenario vs context and context-only views.
    """
    figures = []
    title_base = "Цена за сотку"
    if metric_label:
        title_base = f"{title_base} ({metric_label})"

    figures.append(
        _plot_context_vs_scenario(
            context=context,
            scenario=scenario,
            price_column=price_column,
            vmin=vmin,
            vmax=vmax,
            title=f"{title_base}: проект (контекст ≤6 км серым)",
            show=show,
        )
    )
    figures.append(
        _plot_context_map(
            blocks=blocks,
            scenario=scenario,
            price_column=price_column,
            vmin=vmin,
            vmax=vmax,
            title=f"{title_base}: контекст ≤6 км (сценические серым)",
            show=show,
        )
    )
    return tuple(figures)


def _print_totals(totals: Dict[str, float], metric_label: str) -> None:
    """Display aggregate scenario and context prices in the console.

    Parameters
    ----------
    totals : dict[str, float]
        Mapping containing ``scenario``, ``context``, and ``all`` totals.
    metric_label : str
        Human-readable description of the price column being summarised.
    """
    label = metric_label if metric_label else "выбранная метрика"
    print(f"Суммарная стоимость ({label}):")
    print(" • Сценические кварталы:", _format_currency(totals["scenario"]))
    print(" • Контекст:", _format_currency(totals["context"]))
    print(" • Все кварталы:", _format_currency(totals["all"]))


def _format_currency(value: float) -> str:
    """Return a formatted currency string with thin-space separators."""
    return f"{value:,.0f} ₽".replace(",", " ")


def _format_colorbar(ax) -> None:
    """Apply thousands separators to a Matplotlib colourbar axis."""
    formatter = mticker.FuncFormatter(lambda value, _: f"{int(value):,}".replace(",", " "))
    ax.yaxis.set_major_formatter(formatter)


def _finalize_figure(fig: plt.Figure, *, show: bool) -> plt.Figure:
    """Apply shared layout, optional display, and return the figure."""
    if len(fig.axes) > 1:
        _format_colorbar(fig.axes[-1])
    plt.tight_layout()
    if show:
        plt.show()
    return fig


def _plot_context_vs_scenario(
    *,
    context: GeoDataFrame,
    scenario: GeoDataFrame,
    price_column: str,
    vmin: float,
    vmax: float,
    title: str,
    show: bool,
) -> plt.Figure:
    """Plot scenario blocks atop context blocks using a shared colour scale."""
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    context.plot(ax=ax, color="lightgrey", edgecolor="white", linewidth=0.2, zorder=1)
    if len(scenario):
        scenario.plot(
            ax=ax,
            column=price_column,
            cmap="coolwarm",
            vmin=vmin,
            vmax=vmax,
            legend=True,
            edgecolor="black",
            linewidth=0.3,
            zorder=2,
        )
    ax.set_title(title)
    ax.axis("off")

    return _finalize_figure(fig, show=show)


def _plot_context_map(
    *,
    blocks: GeoDataFrame,
    scenario: GeoDataFrame,
    price_column: str,
    vmin: float,
    vmax: float,
    title: str,
    show: bool,
) -> plt.Figure:
    """Plot the full context map with scenario blocks highlighted in grey."""
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    if len(blocks):
        blocks.plot(
            ax=ax,
            column=price_column,
            cmap="coolwarm",
            vmin=vmin,
            vmax=vmax,
            legend=True,
            edgecolor="black",
            linewidth=0.3,
            zorder=1,
        )
    if len(scenario):
        scenario.plot(ax=ax, color="lightgrey", edgecolor="white", linewidth=0.5, zorder=2)
    ax.set_title(title, pad=10)
    ax.axis("off")

    return _finalize_figure(fig, show=show)
