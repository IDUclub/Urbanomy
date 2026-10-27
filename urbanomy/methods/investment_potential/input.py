"""Utilities for preparing investment-metrics input datasets."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Mapping, Sequence

import geopandas as gpd
import pandas as pd


logger = logging.getLogger(__name__)

INVESTMENT_NUMERIC_COLUMNS: tuple[str, ...] = (
    "land_value",
    "price_per_sotka",
    "residential",
    "business",
    "recreation",
    "industrial",
    "transport",
    "special",
    "agriculture",
    "site_area",
    "living_area",
    "non_living_area",
    "build_floor_area",
    "build_floor_area_before",
)

LAND_USE_SHARE_COLUMNS: tuple[str, ...] = (
    "residential",
    "business",
    "recreation",
    "industrial",
    "transport",
    "special",
    "agriculture",
)


@dataclass(frozen=True)
class InvestmentInputSpec:
    """Schema describing the columns required for investment attractiveness.

    Parameters
    ----------
    required : Sequence[str]
        Columns that must be present in the input GeoDataFrame.
    optional : Sequence[str]
        Columns that are desirable but can be imputed with ``defaults`` when
        missing.
    defaults : Mapping[str, float]
        Default values to use for optional columns when absent or containing
        nulls.
    geometry_column : str, optional
        Name of the geometry column in the target GeoDataFrame (default:
        ``"geometry"``).
    """

    required: Sequence[str]
    optional: Sequence[str]
    defaults: Mapping[str, float]
    geometry_column: str = "geometry"

    def enforce(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Validate and reorder a GeoDataFrame according to the specification.

        Parameters
        ----------
        gdf : geopandas.GeoDataFrame
            Input GeoDataFrame containing at least the required columns.

        Returns
        -------
        geopandas.GeoDataFrame
            Copy of ``gdf`` with columns ordered as ``geometry`` + required +
            optional. Missing optional columns are filled using ``defaults``.

        Raises
        ------
        TypeError
            If ``gdf`` is not a GeoDataFrame.
        ValueError
            If the geometry column or any required column is missing.
        """
        if not isinstance(gdf, gpd.GeoDataFrame):
            raise TypeError("Expected GeoDataFrame input.")

        if self.geometry_column not in gdf.columns:
            raise ValueError(f"Geometry column '{self.geometry_column}' is missing.")

        missing = [col for col in self.required if col not in gdf.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        ordered_columns: list[str] = [self.geometry_column]
        ordered_columns.extend(self.required)
        ordered_columns.extend([col for col in self.optional if col in gdf.columns])

        trimmed = gdf.loc[:, ordered_columns].copy()
        for col in self.optional:
            default_value = self.defaults.get(col, 0.0)
            if col not in trimmed.columns:
                trimmed[col] = default_value
            else:
                trimmed[col] = trimmed[col].fillna(default_value)

        return trimmed


INPUT_SPEC = InvestmentInputSpec(
    required=("land_use", "land_value"),
    optional=(
        "residential",
        "business",
        "recreation",
        "industrial",
        "transport",
        "special",
        "agriculture",
        "site_area",
        "living_area",
        "non_living_area",
        "build_floor_area",
        "land_use_before",
        "build_floor_area_before",
    ),
    defaults={
        "residential": 0.0,
        "business": 0.0,
        "recreation": 0.0,
        "industrial": 0.0,
        "transport": 0.0,
        "special": 0.0,
        "agriculture": 0.0,
        "site_area": 0.0,
        "living_area": 0.0,
        "non_living_area": 0.0,
        "build_floor_area": 0.0,
        "land_use_before": "",
        "build_floor_area_before": 0.0,
    },
)


def _ensure_geodataframe(data: gpd.GeoDataFrame | pd.DataFrame) -> gpd.GeoDataFrame:
    """Coerce a pandas DataFrame into a GeoDataFrame when necessary.

    Parameters
    ----------
    data : geopandas.GeoDataFrame or pandas.DataFrame
        Input data structure expected to contain a ``geometry`` column.

    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame version of ``data`` preserving the original CRS.

    Raises
    ------
    ValueError
        If a pandas DataFrame lacks the ``geometry`` column.
    TypeError
        If ``data`` is neither a GeoDataFrame nor a DataFrame.
    """
    if isinstance(data, gpd.GeoDataFrame):
        return data
    if isinstance(data, pd.DataFrame):
        if "geometry" not in data.columns:
            raise ValueError("Expected DataFrame with a 'geometry' column.")
        return gpd.GeoDataFrame(data, geometry="geometry", crs=getattr(data, "crs", None))
    raise TypeError("Expected GeoDataFrame or DataFrame input.")


def _convert_land_use_shares_to_area(
    gdf: gpd.GeoDataFrame,
    *,
    area_column: str = "site_area",
    land_use_columns: Sequence[str] = LAND_USE_SHARE_COLUMNS,
) -> None:
    """Convert land-use shares (0..1) to area values in square meters in-place.

    Each value in ``land_use_columns`` is multiplied by ``area_column`` when it
    looks like a share (between 0 and 1 inclusive). Values outside that range
    are treated as already being in area units and are preserved.
    """
    if area_column not in gdf.columns:
        return

    site_area = pd.to_numeric(gdf[area_column], errors="coerce")
    for col in land_use_columns:
        if col not in gdf.columns:
            continue
        values = pd.to_numeric(gdf[col], errors="coerce")
        is_share = values.ge(0.0) & values.le(1.0)
        gdf[col] = values.where(~is_share, values * site_area)


def prepare_investment_input(
    gdf: gpd.GeoDataFrame,
    *,
    land_use_column: str = "land_use",
    scenario_flag_column: str = "is_project",
    show_warning: bool = True,
) -> pd.DataFrame:
    """Prepare scenario data for investment-metrics calculation.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Scenario dataset to be normalised and validated.
    land_use_column : str, optional
        Column containing land-use codes (default ``"land_use"``).
    scenario_flag_column : str, optional
        Column indicating scenario membership (default ``"is_project"``).
    show_warning : bool, optional
        Whether to emit warning logs during preparation (default ``True``).

    Returns
    -------
    pandas.DataFrame
        DataFrame ordered according to :data:`INPUT_SPEC` with the geometry
        column removed.
    """

    polygon_gdf = _ensure_geodataframe(gdf).copy()
    if scenario_flag_column in polygon_gdf.columns:
        scenario_mask = polygon_gdf[scenario_flag_column].fillna(False).astype(bool)
        polygon_gdf = polygon_gdf.loc[scenario_mask].reset_index(drop=True)
    elif show_warning:
        logger.warning(
            "prepare_investment_input: колонка %r не найдена; возвращаем все кварталы.",
            scenario_flag_column,
        )

    polygon_gdf["land_value"] = pd.to_numeric(polygon_gdf.get("land_value"), errors="coerce")

    if "land_value_after" in polygon_gdf.columns:
        polygon_gdf["land_value_after"] = pd.to_numeric(polygon_gdf["land_value_after"], errors="coerce")
    else:
        polygon_gdf["land_value_after"] = polygon_gdf["land_value"]

    if "land_value_before" in polygon_gdf.columns:
        polygon_gdf["land_value_before"] = pd.to_numeric(polygon_gdf["land_value_before"], errors="coerce")
        polygon_gdf["land_value"] = polygon_gdf["land_value_before"].fillna(polygon_gdf["land_value_after"])
    else:
        if show_warning:
            logger.warning(
                "prepare_investment_input: колонка 'land_value_before' не найдена; "
                "значение оставлено пустым, текущая цена записана в 'land_value_after'."
            )
        polygon_gdf["land_value_before"] = float("nan")
        polygon_gdf["land_value"] = polygon_gdf["land_value_after"]

    if "build_floor_area_before" in polygon_gdf.columns:
        polygon_gdf["build_floor_area_before"] = pd.to_numeric(polygon_gdf["build_floor_area_before"], errors="coerce")
    elif "build_floor_area" in polygon_gdf.columns:
        polygon_gdf["build_floor_area_before"] = pd.to_numeric(polygon_gdf["build_floor_area"], errors="coerce")

    if "build_floor_area_before" in polygon_gdf.columns and "build_floor_area" in polygon_gdf.columns:
        polygon_gdf["build_floor_area_before"] = pd.to_numeric(
            polygon_gdf["build_floor_area_before"], errors="coerce"
        ).fillna(pd.to_numeric(polygon_gdf["build_floor_area"], errors="coerce"))

    if "land_use_before" not in polygon_gdf.columns:
        polygon_gdf["land_use_before"] = polygon_gdf.get(land_use_column, "")
    elif land_use_column in polygon_gdf.columns:
        polygon_gdf["land_use_before"] = polygon_gdf["land_use_before"].fillna(polygon_gdf[land_use_column])

    _convert_land_use_shares_to_area(
        polygon_gdf,
        area_column="site_area",
        land_use_columns=LAND_USE_SHARE_COLUMNS,
    )

    prepared = INPUT_SPEC.enforce(polygon_gdf)
    geometry_column = prepared.geometry.name if hasattr(prepared, "geometry") else None
    if geometry_column and geometry_column in prepared.columns:
        prepared = prepared.drop(columns=geometry_column)
    return pd.DataFrame(prepared)


__all__ = [
    "INVESTMENT_NUMERIC_COLUMNS",
    "LAND_USE_SHARE_COLUMNS",
    "InvestmentInputSpec",
    "INPUT_SPEC",
    "prepare_investment_input",
]
