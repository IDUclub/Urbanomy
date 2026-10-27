from __future__ import annotations

import json
from collections.abc import Mapping

import geopandas as gpd
import pandas as pd
from blocksnet.enums import LandUse

from .constants import LAND_USE_CONFIGS, LAND_USE_TO_POTENTIAL_COLUMN, LAND_USE_WEIGHTS


class LandUseScoreAnalyzer:
    """
    Compute spatial investment attractiveness scores for land-use types.

    Takes a GeoDataFrame with raw land-use attributes and outputs
    both wide- and long-format GeoDataFrames of investment scores.
    """

    def __init__(
        self,
        weights: Mapping[str | LandUse, Mapping[str, float]] | None = None,
        weights_path: str | None = None,
    ):
        """
        Initialize the analyzer.

        Parameters
        ----------
        weights : Mapping[str | LandUse, Mapping[str, float]] or None
            Custom per-land-use weighting factors for attributes keyed by strings or ``LandUse``.
            If None, defaults to LAND_USE_WEIGHTS.
        weights_path : str or None
            Path to JSON file containing weights. Used if `weights` is None.

        Raises
        ------
        FileNotFoundError
            If `weights` is None and `weights_path` is provided but file is not found.
        ValueError
            If JSON at `weights_path` is invalid or not a dict of dicts.
        """
        self.configs = LAND_USE_CONFIGS
        if weights is not None:
            self.weights = self._normalise_weights(weights)
        elif weights_path:
            try:
                with open(weights_path, "r", encoding="utf-8") as file_obj:
                    loaded = json.load(file_obj)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Could not decode JSON weights from '{weights_path}'.") from exc
            if not isinstance(loaded, Mapping):
                raise ValueError("Weights JSON must contain an object at the top level.")
            self.weights = self._normalise_weights(loaded)
        else:
            self.weights = {lu: dict(w) for lu, w in LAND_USE_WEIGHTS.items()}
        self.land_use_to_potential: dict[str, str] = LAND_USE_TO_POTENTIAL_COLUMN

    @staticmethod
    def _coerce_land_use(value: str | LandUse) -> LandUse:
        if isinstance(value, LandUse):
            return value
        try:
            return LandUse(str(value))
        except ValueError as exc:
            raise ValueError(f"Unknown land-use key: {value!r}") from exc

    def _normalise_weights(
        self,
        weights: Mapping[str | LandUse, Mapping[str, float]],
    ) -> dict[str, dict[str, float]]:
        normalised: dict[str, dict[str, float]] = {}
        for key, inner in weights.items():
            land_use = self._coerce_land_use(key)
            if not isinstance(inner, Mapping):
                raise ValueError(
                    f"Weights for land-use '{land_use.value}' must be a mapping, " f"got {type(inner).__name__}"
                )
            normalised[land_use.value] = {str(attr): float(weight) for attr, weight in inner.items()}
        return normalised

    def _compute_wide(self, polygon_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Compute wide-format investment scores (no 'ИП_' prefix).

        For each land-use key, computes a weighted average of numeric attributes
        scaled by the corresponding potential column.

        Parameters
        ----------
        polygon_gdf : geopandas.GeoDataFrame
            Input GeoDataFrame containing:
            - numeric attribute columns
            - potential columns as specified in LAND_USE_TO_POTENTIAL_COLUMN
            - geometry column

        Returns
        -------
        geopandas.GeoDataFrame
            Copy of input with additional score columns named by land-use keys,
            each containing a float score or None where not applicable.
        """
        gdf = polygon_gdf.copy()
        pot_cols = list(self.land_use_to_potential.values())
        attrs = [
            col
            for col in gdf.select_dtypes("number").columns
            if col not in pot_cols and ((gdf[col].between(-5, 5) & (gdf[col] != 0)).any())
        ]

        # Read attributes before writing score columns, which can share their names.
        attributes = gdf[attrs].copy()
        valid_counts = attributes.count(axis=1).replace(0, float("nan"))
        for land_use, config in self.configs.items():
            score_col = land_use.value
            weights_for_lu = self.weights.get(score_col, {})
            default_weight = weights_for_lu.get("default", config.indicator_weights.get("default", 1.0))
            weights = pd.Series({attr: weights_for_lu.get(attr, default_weight) for attr in attrs}, dtype=float)
            weighted = attributes.mul(weights, axis=1).sum(axis=1, min_count=1)
            potential = gdf.get(config.potential_column, pd.Series(float("nan"), index=gdf.index))
            gdf[score_col] = (weighted / valid_counts * (potential / 5)).round(1)

        return gdf

    def compute_scores_long(self, polygon_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Convert wide-format scores to long format.

        Takes the output of `_compute_wide` and melts score columns
        into `ip_type`, `spatial_potential`, preserving geometry.

        Parameters
        ----------
        polygon_gdf : geopandas.GeoDataFrame
            Input GeoDataFrame with wide-format score columns.

        Returns
        -------
        geopandas.GeoDataFrame
            Long-format GeoDataFrame with columns:
            - ip_type : str, land-use key
            - spatial_potential: float, computed score
            - geometry: Polygon geometry
        """
        wide = self._compute_wide(polygon_gdf)
        score_cols = list(self.land_use_to_potential.keys())

        df_long = (
            wide[["geometry", *score_cols]]
            .melt(id_vars="geometry", value_vars=score_cols, var_name="ip_type", value_name="spatial_potential")
            .dropna(subset=["spatial_potential"])
            .reset_index(drop=True)
        )

        return gpd.GeoDataFrame(df_long, geometry="geometry", crs=wide.crs)
