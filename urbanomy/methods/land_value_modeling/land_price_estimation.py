"""Tools for estimating land prices using spatial lag features."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import geopandas as gpd
import numpy as np
import pandas as pd
from blocksnet.enums import LandUse
from libpysal.weights import DistanceBand

from ._spatial_features import build_distance_weights, compute_lag_features
from .constants import CATEGORICAL_FEATURES, ORIGINAL_FEATURES, RADIUS_LIST, SERVICE_FEATURES


class LandPriceEstimator:
    """Estimate land prices for blocks using a pretrained regression model.

    Parameters
    ----------
    model : object
        Fitted regression model exposing a ``predict`` method that accepts a
        pandas DataFrame and returns unit-price predictions.
    blocks : geopandas.GeoDataFrame
        Blocks dataset containing the geometry and all required base features.
    radius_list : Sequence[float], optional
        Distance thresholds used to build spatial weights. Defaults to
        ``(300, 500, 1000, 2000, 3000)``.
    orig_features : Sequence[str], optional
        Names of the base features that will be passed to the model. Defaults to
        the feature list used in the original notebook workflow.
    categorical_features : Sequence[str], optional
        Subset of ``orig_features`` that should be treated as categorical.
    """

    DEFAULT_FEATURES: Sequence[str] = ORIGINAL_FEATURES

    DEFAULT_CATEGORICAL: Sequence[str] = CATEGORICAL_FEATURES

    DEFAULT_RADII: Sequence[float] = RADIUS_LIST

    def __init__(
        self,
        *,
        model,
        blocks: gpd.GeoDataFrame,
        radius_list: Sequence[float] | None = None,
        orig_features: Sequence[str] | None = None,
        categorical_features: Sequence[str] | None = None,
        use_service_features: bool = False,
        service_features: Sequence[str] | None = None,
    ) -> None:
        """Initialise the estimator with model, data and feature configuration.

        Parameters
        ----------
        model : object
            Trained estimator exposing a ``predict`` method that accepts a
            pandas DataFrame and returns logarithmic prices.
        blocks : geopandas.GeoDataFrame
            Dataset containing geometries and base features required by
            ``orig_features``.
        radius_list : Sequence[float], optional
            Distance thresholds for spatial lag computation.
        orig_features : Sequence[str], optional
            Feature names supplied to the estimator.
        categorical_features : Sequence[str], optional
            Subset of features that should be treated as categorical.
        use_service_features : bool, optional
            When ``True`` дополнительно включает сервисные признаки вроде
            ``count_*`` и ``osr``/``share_*``. Полезно, если модель обучена с
            такими колонками. По умолчанию выключено для обратной совместимости.
        service_features : Sequence[str], optional
            Явный список сервисных признаков, если нужно переопределить
            стандартный набор ``SERVICE_FEATURES``.
        """
        self.model = model
        self._blocks = blocks.copy()
        self._radii = tuple(radius_list) if radius_list is not None else tuple(self.DEFAULT_RADII)
        base_features = tuple(orig_features) if orig_features is not None else tuple(self.DEFAULT_FEATURES)
        self._service_features = tuple(service_features) if service_features is not None else tuple(SERVICE_FEATURES)
        if use_service_features and self._service_features:
            base_features = tuple(dict.fromkeys([*base_features, *self._service_features]))
        self._orig_features = base_features
        self._categorical_features = (
            tuple(categorical_features) if categorical_features is not None else tuple(self.DEFAULT_CATEGORICAL)
        )
        self._numeric_features = tuple(
            feature for feature in self._orig_features if feature not in self._categorical_features
        )
        self._validate_inputs()
        self._weights = self._build_distance_weights()

    def predict(self, blocks: gpd.GeoDataFrame | None = None) -> gpd.GeoDataFrame:
        """Generate price predictions for each block.

        Parameters
        ----------
        blocks : geopandas.GeoDataFrame or None, optional
            Scenario features. Spatial weights are reused when its geometry,
            index, and CRS match the original blocks.

        Returns
        -------
        geopandas.GeoDataFrame
            Copy of the original blocks with one extra column:
            ``land_value``.
        """
        return self.predict_prices(
            blocks=blocks,
            include_unit_price=False,
            predictions_in_log_scale=True,
        )

    def predict_prices(
        self,
        *,
        blocks: gpd.GeoDataFrame | None = None,
        unit_price_column: str = "land_value_per_sqm",
        total_price_column: str = "land_value",
        area_column: str = "site_area",
        predictions_in_log_scale: bool = False,
        include_unit_price: bool = True,
    ) -> gpd.GeoDataFrame:
        """Generate total and unit land value predictions.

        Parameters
        ----------
        blocks : geopandas.GeoDataFrame or None, optional
            Scenario blocks to evaluate. Defaults to the constructor's dataset.
        unit_price_column : str, optional
            Output column for unit price predictions (rub/sqm), calculated as
            ``total_price_column / area_column``.
        total_price_column : str, optional
            Output column for total block value predicted by the model.
        area_column : str, optional
            Area column used to convert total value to unit price.
        predictions_in_log_scale : bool, optional
            When ``True``, applies ``np.expm1`` to model outputs
            (inverse transform for ``log1p`` targets).
        include_unit_price : bool, optional
            When ``True``, computes ``unit_price_column`` as
            ``total_price_column / area_column``.
        """
        source = self._blocks if blocks is None else blocks
        if blocks is not None:
            self._validate_inputs(source)
        design_matrix = self._design_matrix(source)
        design_matrix = self._align_features_to_model(design_matrix)
        y_pred = np.asarray(self.model.predict(design_matrix)).reshape(-1)
        if predictions_in_log_scale:
            y_pred = np.expm1(y_pred)

        blocks_pred = source.copy()
        blocks_pred[total_price_column] = y_pred

        if include_unit_price:
            if area_column not in blocks_pred.columns:
                raise ValueError(f"Cannot compute '{unit_price_column}': area column '{area_column}' is missing.")
            area = pd.to_numeric(blocks_pred[area_column], errors="coerce")
            total = pd.to_numeric(blocks_pred[total_price_column], errors="coerce")
            blocks_pred[unit_price_column] = np.where(area > 0, total / area, np.nan)
        return blocks_pred

    def _build_distance_weights(self, blocks: gpd.GeoDataFrame | None = None) -> Mapping[float, DistanceBand]:
        """Construct distance-band spatial weights for each configured radius.

        Returns
        -------
        Mapping[float, libpysal.weights.DistanceBand]
            Dictionary mapping radius values to ``DistanceBand`` instances.
        """
        return build_distance_weights(self._blocks if blocks is None else blocks, self._radii)

    def _design_matrix(self, blocks: gpd.GeoDataFrame) -> pd.DataFrame:
        """Assemble the model design matrix including spatial lag features.

        Parameters
        ----------
        blocks : geopandas.GeoDataFrame
            Dataset for which predictions will be generated.

        Returns
        -------
        pandas.DataFrame
            Feature matrix aligned with ``blocks.index``.
        """
        base = blocks[list(self._orig_features)].copy()

        for column in self._categorical_features:
            if column in base.columns:
                base[column] = base[column].apply(self._categorical_token).astype("string")

        lag_features = self._compute_lag_features(blocks)
        return pd.concat([base, lag_features], axis=1)

    @staticmethod
    def _categorical_token(value: Any) -> str:
        """Convert categorical values to stable tokens for model input."""
        if value is None:
            return "missing"
        if isinstance(value, float) and np.isnan(value):
            return "missing"
        if isinstance(value, LandUse):
            return value.name

        text = str(value).strip()
        if not text or text.lower() == "nan":
            return "missing"

        try:
            return LandUse(text).name
        except ValueError:
            pass

        upper = text.upper()
        if upper.startswith("LANDUSE."):
            upper = upper.split(".", 1)[1]
        if upper in LandUse.__members__:
            return upper
        return upper

    def _compute_lag_features(self, blocks: gpd.GeoDataFrame) -> pd.DataFrame:
        """Compute spatial lag features for numeric columns and neighbour counts.

        Parameters
        ----------
        blocks : geopandas.GeoDataFrame
            Dataset whose numeric columns are used to compute lags.

        Returns
        -------
        pandas.DataFrame
            Lag feature frame indexed by ``blocks.index``.
        """
        same_layout = (
            blocks.index.equals(self._blocks.index)
            and blocks.crs == self._blocks.crs
            and blocks.geometry.equals(self._blocks.geometry)
        )
        weights = self._weights if same_layout else self._build_distance_weights(blocks)
        return compute_lag_features(blocks, self._numeric_features, weights)

    def _align_features_to_model(self, design_matrix: pd.DataFrame) -> pd.DataFrame:
        """Reorder/validate columns to match the fitted model's feature layout."""
        model_features = getattr(self.model, "feature_names_", None)
        if model_features is None or len(model_features) == 0:
            return design_matrix

        missing = [name for name in model_features if name not in design_matrix.columns]
        if missing:
            raise ValueError("Design matrix is missing features expected by the model: " + ", ".join(sorted(missing)))
        return design_matrix.reindex(columns=model_features)

    def _validate_inputs(self, blocks: gpd.GeoDataFrame | None = None) -> None:
        """Ensure that all required features are present in the blocks dataset.

        Raises
        ------
        ValueError
            If any of ``orig_features`` are missing from ``self._blocks``.
        """
        blocks = self._blocks if blocks is None else blocks
        missing_categorical = [feature for feature in self._categorical_features if feature not in self._orig_features]
        if missing_categorical:
            raise ValueError(
                "categorical_features must be a subset of orig_features. "
                "Missing from orig_features: " + ", ".join(sorted(missing_categorical))
            )

        missing_columns = [feature for feature in self._orig_features if feature not in blocks.columns]
        if missing_columns:
            raise ValueError("The blocks dataset is missing required features: " + ", ".join(sorted(missing_columns)))


def transfer_baseline_prices(
    after_blocks: gpd.GeoDataFrame,
    before_blocks: gpd.GeoDataFrame,
    *,
    id_column: str = "id",
    price_column: str = "land_value",
    scenario_column: str = "is_project",
    output_column: str = "land_value_before",
    area_column: str = "site_area",
    scenario_mode: bool = True,
    unit_price_column: str = "land_value_per_100m2",
    unit_price_output_column: str = "land_value_before_per_100m2",
) -> gpd.GeoDataFrame:
    """Project baseline land prices from historical blocks onto scenario blocks.

    The function computes a weighted average unit price for each scenario block
    based on the proportional overlap with historical blocks. The resulting
    baseline price is added as a new column to a copy of ``after_blocks``.

    Parameters
    ----------
    after_blocks : geopandas.GeoDataFrame
        Blocks describing the post-development scenario. Must contain geometry,
        ``id_column`` and ``scenario_column``.
    before_blocks : geopandas.GeoDataFrame
        Baseline blocks with historical prices. Must contain geometry,
        ``price_column`` and ``scenario_column``.
    id_column : str, optional
        Unique polygon identifier present in ``after_blocks``. Defaults to ``"id"``.
    price_column : str, optional
        Column in ``before_blocks`` containing baseline total prices. Defaults to ``"land_value"``.
    scenario_column : str, optional
        Boolean column indicating scenario polygons. Defaults to ``"is_project"``.
    output_column : str, optional
        Name of the column to store the transferred baseline price inside the
        returned GeoDataFrame. Defaults to ``"land_value_before"``.
    area_column : str, optional
        Column describing block areas in square metres. When missing or
        containing non-positive values, geometry-derived areas are used.
    scenario_mode : bool, optional
        When ``True`` (по умолчанию) ожидает наличие ``scenario_column`` и
        переносит цены только для проектных кварталов с учетом пересечений
        геометрий. Когда ``False``, работает в упрощенном режиме: не требует
        столбца проекта и просто сопоставляет цены по ``id_column`` без
        пространственного переноса.
    unit_price_column : str, optional
        Имя колонки в ``before_blocks`` с уже рассчитанной удельной ценой
        (например, ``land_value_per_100m2``).
    unit_price_output_column : str, optional
        Имя колонки в возвращаемом GeoDataFrame для удельной цены "до"
        (например, ``land_value_before_per_100m2``).

    Returns
    -------
    geopandas.GeoDataFrame
        Copy of ``after_blocks`` with an additional ``output_column`` describing
        the weighted baseline price for each polygon.

    Raises
    ------
    KeyError
        If required columns are missing from the input GeoDataFrames.
    """
    required_after = {id_column}
    required_before = {id_column, price_column}
    if scenario_mode:
        required_after.add(scenario_column)
        required_before.add(scenario_column)

    for frame_name, frame, required in (
        ("after_blocks", after_blocks, required_after),
        ("before_blocks", before_blocks, required_before),
    ):
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise KeyError(f"{frame_name} is missing required columns: {', '.join(sorted(missing))}")
        if frame.geometry is None:
            raise KeyError(f"{frame_name} must provide a geometry column")

    after_geom_col = after_blocks.geometry.name
    before_geom_col = before_blocks.geometry.name

    result = after_blocks.copy()
    baseline_column = "_baseline_price"
    baseline_unit_column = "_baseline_unit_price"

    baseline_mapping = (
        before_blocks[[id_column, price_column]]
        .drop_duplicates(subset=id_column)
        .rename(columns={price_column: baseline_column})
    )
    if unit_price_column in before_blocks.columns:
        unit_map = (
            before_blocks[[id_column, unit_price_column]]
            .drop_duplicates(subset=id_column)
            .rename(columns={unit_price_column: baseline_unit_column})
        )
        baseline_mapping = baseline_mapping.merge(unit_map, on=id_column, how="left")
    else:
        baseline_mapping[baseline_unit_column] = np.nan

    def _resolve_area(df: gpd.GeoDataFrame) -> np.ndarray:
        if area_column in df.columns:
            area_series = pd.to_numeric(df[area_column], errors="coerce")
        else:
            area_series = pd.Series(np.nan, index=df.index, dtype=float)
        area_values = area_series.to_numpy(copy=True)
        invalid_mask = ~np.isfinite(area_values) | (area_values <= 0)
        if invalid_mask.any():
            geom_area = df.geometry.area.to_numpy()
            area_values[invalid_mask] = geom_area[invalid_mask]
        return area_values

    def _apply_baseline(df: pd.DataFrame) -> pd.DataFrame:
        merged = df.merge(baseline_mapping, on=id_column, how="left")
        if output_column not in merged.columns:
            merged[output_column] = np.nan
        if unit_price_output_column not in merged.columns:
            merged[unit_price_output_column] = np.nan

        if scenario_mode:
            non_scenario_mask = ~merged[scenario_column].fillna(False).astype(bool)
            merged.loc[non_scenario_mask, output_column] = merged.loc[non_scenario_mask, output_column].fillna(
                merged.loc[non_scenario_mask, baseline_column]
            )
            merged.loc[non_scenario_mask, unit_price_output_column] = merged.loc[
                non_scenario_mask, unit_price_output_column
            ].fillna(merged.loc[non_scenario_mask, baseline_unit_column])
        else:
            merged[output_column] = merged[output_column].fillna(merged[baseline_column])
            merged[unit_price_output_column] = merged[unit_price_output_column].fillna(merged[baseline_unit_column])

        return merged.drop(columns=[baseline_column, baseline_unit_column])

    if not scenario_mode:
        if result.empty or before_blocks.empty:
            simplified = gpd.GeoDataFrame(
                _apply_baseline(result),
                geometry=after_geom_col,
                crs=after_blocks.crs,
            )
        else:
            simplified = gpd.GeoDataFrame(
                _apply_baseline(result),
                geometry=after_geom_col,
                crs=after_blocks.crs,
            )
        enriched = simplified
    else:
        if result.empty or before_blocks.empty:
            enriched = gpd.GeoDataFrame(
                _apply_baseline(result),
                geometry=after_geom_col,
                crs=after_blocks.crs,
            )
        else:
            before_blocks = before_blocks.copy()
            before_blocks["area_before"] = before_blocks.geometry.area
            before_blocks["unit_price_before"] = np.where(
                before_blocks["area_before"] > 0,
                before_blocks[price_column] / before_blocks["area_before"],
                np.nan,
            )

            intersections = gpd.overlay(
                result[[id_column, after_geom_col]],
                before_blocks[["unit_price_before", before_geom_col]],
                how="intersection",
                keep_geom_type=False,
            )

            if intersections.empty:
                enriched = gpd.GeoDataFrame(
                    _apply_baseline(result),
                    geometry=after_geom_col,
                    crs=after_blocks.crs,
                )
            else:
                intersections["intersect_area"] = intersections.geometry.area
                area_sum = (
                    intersections.groupby(id_column, as_index=False)["intersect_area"]
                    .sum()
                    .rename(columns={"intersect_area": "id_total_area"})
                )
                intersections = intersections.merge(area_sum, on=id_column, how="left")

                intersections["weight"] = np.divide(
                    intersections["intersect_area"],
                    intersections["id_total_area"],
                    out=np.zeros_like(intersections["intersect_area"]),
                    where=intersections["id_total_area"] > 0,
                )
                intersections["contrib"] = intersections["unit_price_before"] * intersections["weight"]

                price_transfer = (
                    intersections.groupby(id_column, as_index=False)["contrib"]
                    .sum()
                    .rename(columns={"contrib": "unit_price_before_weighted"})
                )

                area_lookup = result[[id_column]].copy()
                area_lookup["_resolved_area"] = _resolve_area(result)
                price_transfer = price_transfer.merge(area_lookup, on=id_column, how="left")
                price_transfer[output_column] = (
                    price_transfer["unit_price_before_weighted"] * price_transfer["_resolved_area"]
                )
                merged = result.merge(price_transfer[[id_column, output_column]], on=id_column, how="left")

                enriched = gpd.GeoDataFrame(
                    _apply_baseline(merged),
                    geometry=after_geom_col,
                    crs=after_blocks.crs,
                )

    if output_column in enriched.columns and "land_value" in enriched.columns:
        enriched["land_value_delta_pct"] = np.where(
            enriched[output_column] > 0,
            (enriched["land_value"] / enriched[output_column] - 1.0) * 100,
            np.nan,
        )

    return enriched
