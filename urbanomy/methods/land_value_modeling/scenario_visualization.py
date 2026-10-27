"""Maps and summaries for already evaluated development scenarios."""

from __future__ import annotations

from typing import Any, Dict

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from geopandas import GeoDataFrame
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.transforms import offset_copy

from .constants import ScenarioResultKey
from .scenario_summary import _fmt_pct, _fmt_rub, _format_land_use, _print_summary, _summarise_changes


def plot_scenario_impact(
    *,
    blocks: GeoDataFrame,
    target_idx: Any,
    target_id_column: str = "id",
    before_column: str = "land_value_before",
    after_column: str = "land_value",
    pct_column: str = "land_value_delta_pct",
    delta_column: str = "d_rub",
    eps: float = 1e-9,
    buffer_radius: float = 4000.0,
    figsize: tuple[float, float] | None = None,
    show: bool = True,
    print_summary: bool = True,
    print_quarter_stats: bool = True,
) -> Dict[str, object]:
    """Визуализировать и вывести статистику по уже рассчитанным изменениям цен.

    Ожидается, что ``blocks`` уже содержит колонки с ценой до/после сценария.
    Функция ничего не предсказывает — только строит карту изменений и печатает
    сводную статистику. На карте используется процентное изменение полной цены
    ``land_value_delta_pct``. ``target_idx`` ожидает
    идентификатор квартала (значение в колонке ``target_id_column``).
    """

    missing = [col for col in (before_column, after_column) if col not in blocks.columns]
    if missing:
        raise KeyError(f"blocks is missing required columns: {', '.join(sorted(missing))}")

    combined = blocks.copy()
    combined["land_value_before"] = combined[before_column].astype(float)
    combined["land_value_after"] = combined[after_column].astype(float)
    combined = _ensure_valid_geoms(combined)

    if target_id_column not in combined.columns:
        raise KeyError(f"blocks is missing target_id_column: {target_id_column}")
    match = combined[combined[target_id_column] == target_idx]
    if match.empty:
        raise KeyError(f"target_id {target_idx!r} not found in column {target_id_column!r}")
    resolved_target_idx = match.index[0]

    if delta_column in combined.columns:
        combined["d_rub"] = combined[delta_column].astype(float)
    else:
        combined["d_rub"] = combined["land_value_after"] - combined["land_value_before"]

    if pct_column in combined.columns:
        combined["d_pct"] = combined[pct_column].astype(float)
    else:
        combined["d_pct"] = (combined["land_value_after"] / combined["land_value_before"] - 1.0) * 100
    combined["d_pct"] = combined["d_pct"].replace([np.inf, -np.inf], np.nan)

    buffer_gdf = _build_buffer(combined, resolved_target_idx, buffer_radius)
    buffer_gdf = _ensure_valid_geoms(buffer_gdf)
    clipped = gpd.clip(combined, buffer_gdf) if len(combined) else combined

    changed = clipped[np.abs(clipped["d_rub"].astype(float)) > eps].copy()
    unchanged = clipped[np.abs(clipped["d_rub"].astype(float)) <= eps].copy()

    plot_column = pct_column if pct_column in combined.columns else "d_pct"
    if plot_column not in combined.columns:
        raise KeyError(f"blocks is missing plot_column: {plot_column}")

    # Use a fixed symmetric percentage scale for consistent maps
    vmin, vmax = -20.0, 20.0

    fig = _plot_change_map(
        changed=changed,
        unchanged=unchanged,
        target_geometry=combined.loc[resolved_target_idx, "geometry"],
        plot_column=plot_column,
        vmin=vmin,
        vmax=vmax,
        figsize=figsize,
        show=show,
    )

    summary_all = _summarise_changes(clipped)
    summary_changed = _summarise_changes(changed)

    if print_summary:
        _print_summary(
            changed,
            summary_changed,
            eps=eps,
            total_summary=summary_all,
            total_count=int(len(clipped)),
            target_idx=resolved_target_idx,
            print_quarter_stats=print_quarter_stats,
        )

    return {
        ScenarioResultKey.MAP.value: changed if len(changed) else clipped.iloc[0:0].copy(),
        ScenarioResultKey.MAP_ALL.value: clipped,
        ScenarioResultKey.FIGURE.value: fig,
        ScenarioResultKey.SUMMARY.value: summary_changed,
        ScenarioResultKey.SUMMARY_ALL.value: summary_all,
    }


def _build_buffer(blocks: GeoDataFrame, target_idx: int, radius_m: float) -> GeoDataFrame:
    """Create a buffer around the target block geometry.

    Parameters
    ----------
    blocks : geopandas.GeoDataFrame
        Source dataset containing the geometry.
    target_idx : int
        Index of the block serving as the buffer centre.
    radius_m : float
        Buffer radius expressed in metres.

    Returns
    -------
    geopandas.GeoDataFrame
        Single-row GeoDataFrame with the buffered geometry.
    """
    tgt_geom = blocks.loc[target_idx, "geometry"]
    buf = gpd.GeoSeries([tgt_geom], crs=blocks.crs)
    if buf.crs and buf.crs.is_geographic:
        utm = buf.estimate_utm_crs()
        buf = buf.to_crs(utm).buffer(radius_m).to_crs(blocks.crs)
    else:
        buf = buf.buffer(radius_m)
    return gpd.GeoDataFrame(geometry=buf, crs=blocks.crs)


def _ensure_valid_geoms(gdf: GeoDataFrame) -> GeoDataFrame:
    """Fix invalid geometries to avoid GEOS TopologyException during overlays/clipping."""
    if gdf.empty or gdf.geometry is None:
        return gdf

    geom = gdf.geometry
    try:
        invalid_mask = ~geom.is_valid
    except Exception:
        invalid_mask = pd.Series(False, index=gdf.index)

    if not invalid_mask.any():
        return gdf

    fixed = geom.copy()
    try:
        from shapely.validation import make_valid  # type: ignore

        fixed.loc[invalid_mask] = geom.loc[invalid_mask].apply(make_valid)
    except Exception:
        fixed.loc[invalid_mask] = geom.loc[invalid_mask].buffer(0)

    return gdf.set_geometry(fixed)


def _plot_change_map(
    *,
    changed: GeoDataFrame,
    unchanged: GeoDataFrame,
    target_geometry,
    plot_column: str,
    vmin: float,
    vmax: float,
    figsize: tuple[float, float] | None,
    show: bool,
) -> plt.Figure:
    """Plot price changes around the target block using the selected metric.

    Parameters
    ----------
    changed : geopandas.GeoDataFrame
        Blocks whose price change magnitude exceeds ``eps``.
    unchanged : geopandas.GeoDataFrame
        Blocks within the buffer that remain below the change threshold.
    target_geometry : shapely.geometry.base.BaseGeometry
        Geometry of the focus block to outline.
    vmin, vmax : float
        Color scale bounds for percentage change.
    plot_column : str
        Column in ``changed``/``unchanged`` to visualise (e.g., ``d_pct`` or ``d_rub``).
    figsize : tuple[float, float] | None
        Optional figure size passed to ``plt.subplots``.
    show : bool
        Whether to render the Matplotlib figure immediately.

    Returns
    -------
    matplotlib.figure.Figure
        Figure displaying the scenario impact map.
    """
    fig, ax = plt.subplots(figsize=figsize or (35, 30))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plot_changed = changed.copy()
    plot_unchanged = unchanged.copy()
    has_colorbar = False

    color_label = "Изменение цены, %"
    formatter = _fmt_pct
    if plot_column.lower().endswith("rub") or plot_column.lower().endswith("value") or plot_column == "d_rub":
        color_label = "Изменение цены, ₽"
        formatter = lambda val: _fmt_rub(val, signed=True)  # noqa: E731

    base_crs = plot_changed.crs or plot_unchanged.crs
    target_gdf = gpd.GeoDataFrame(geometry=[target_geometry], crs=base_crs)
    plotting_crs = plot_changed.crs or plot_unchanged.crs or target_gdf.crs

    if base_crs is not None:
        if plot_changed.crs is None:
            plot_changed = plot_changed.set_crs(base_crs, allow_override=True)
        if plot_unchanged.crs is None:
            plot_unchanged = plot_unchanged.set_crs(base_crs, allow_override=True)
        if target_gdf.crs is None:
            target_gdf = target_gdf.set_crs(base_crs, allow_override=True)

    target_plot_gdf = target_gdf

    if len(plot_unchanged):
        plot_unchanged.plot(ax=ax, color="lightgrey", edgecolor="white", linewidth=0.3, zorder=1)
    if len(plot_changed):
        plot_changed.plot(
            ax=ax,
            column=plot_column,
            cmap="coolwarm",
            vmin=vmin,
            vmax=vmax,
            legend=True,
            legend_kwds={
                "label": color_label,
                "orientation": "vertical",
                "pad": 0.02,
                "shrink": 0.7,
            },
            edgecolor="black",
            linewidth=0.4,
            zorder=2,
        )

        if len(fig.axes) > 1:
            cbar_ax = fig.axes[-1]
            cbar_ax.set_ylabel(color_label, fontsize=20)
            cbar_ax.tick_params(labelsize=14)
            cbar_ax.set_position([0.92, 0.15, 0.025, 0.7])
            has_colorbar = True

        for _, row in plot_changed.iterrows():
            x, y = row.geometry.centroid.coords[0]
            value_text = formatter(row.get(plot_column))
            ax.text(
                x,
                y,
                value_text,
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                color="black",
                zorder=4,
            )
            ax.text(
                x,
                y,
                _format_land_use(row.get("land_use")),
                ha="center",
                va="center",
                fontsize=6,
                color="black",
                transform=offset_copy(ax.transData, fig=fig, y=-10, units="points"),
                zorder=4,
            )

    target_plot_gdf.boundary.plot(
        ax=ax,
        edgecolor="red",
        linewidth=2.0,
        zorder=3,
    )

    legend_handles: list[object] = []
    if len(plot_unchanged):
        legend_handles.append(
            Patch(
                facecolor="lightgrey",
                edgecolor="white",
                linewidth=0.3,
                label="Кварталы без существующих изменений",
            )
        )
    if len(plot_changed):
        legend_handles.append(
            Patch(
                facecolor="none",
                edgecolor="black",
                linewidth=0.4,
                label="Кварталы с изменением цены",
            )
        )
    legend_handles.append(Line2D([0], [0], color="red", linewidth=2.0, label="Границы изменяемого квартала"))
    ax.legend(
        handles=legend_handles,
        loc="upper right",
        frameon=True,
        title="Обозначения",
        title_fontsize=18,
        prop={"size": 16},
    )

    if has_colorbar:
        ax.set_position([0.02, 0.02, 0.88, 0.96])
    else:
        ax.set_position([0.02, 0.02, 0.96, 0.96])

    _add_scale_bar(
        ax=ax,
        crs=plotting_crs,
        label="Масштаб",
    )

    ax.set_title("Изменение цены в процентах (after − before)", pad=12, fontsize=30)
    ax.axis("off")
    if show:
        plt.show()
    return fig


def _add_scale_bar(
    *,
    ax: plt.Axes,
    crs,
    label: str,
    units: str = "м",
    length: float | None = None,
    location: tuple[float, float] = (0.08, 0.08),
    linewidth: float = 4.0,
) -> None:
    """Draw a simple scale bar in the lower corner of the map."""
    if not ax:
        return

    if hasattr(crs, "is_geographic") and crs.is_geographic:
        ax.text(
            0.02,
            0.02,
            "Масштаб недоступен (географическая проекция)",
            transform=ax.transAxes,
            fontsize=10,
            va="bottom",
            ha="left",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
        )
        return

    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    width = float(x_max - x_min)
    height = float(y_max - y_min)
    if width <= 0 or height <= 0:
        return

    candidates = (50, 100, 200, 250, 500, 1000, 2000, 5000, 10000, 20000)
    target = width / 5.0
    if not length:
        length = max((cand for cand in candidates if cand <= target), default=candidates[0])
        if length > target and len(candidates):
            length = min(candidates, key=lambda cand: abs(cand - target))

    x_start = x_min + width * location[0]
    y_start = y_min + height * location[1]
    segment_length = length / 2
    baseline = y_start
    tick_height = height * 0.02

    ax.plot(
        [x_start, x_start + length],
        [baseline, baseline],
        color="black",
        linewidth=linewidth,
        solid_capstyle="butt",
        zorder=5,
    )

    tick_positions = (x_start, x_start + segment_length, x_start + length)
    tick_labels = (
        "0",
        f"{int(round(segment_length))}",
        f"{int(round(length))}",
    )
    for x_pos in tick_positions:
        ax.plot(
            [x_pos, x_pos],
            [baseline, baseline + tick_height],
            color="black",
            linewidth=linewidth / 1.5,
            zorder=6,
        )

    for label_text, x_pos in zip(tick_labels, tick_positions):
        suffix = f" {units}" if x_pos == tick_positions[-1] else ""
        ax.text(
            x_pos,
            baseline + tick_height * 1.4,
            f"{label_text}{suffix}",
            ha="center",
            va="bottom",
            fontsize=13,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
            zorder=7,
        )

    ax.text(
        x_start,
        baseline - tick_height * 1.8,
        label,
        ha="left",
        va="top",
        fontsize=13,
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
        zorder=6,
    )
