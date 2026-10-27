from __future__ import annotations

from contextlib import contextmanager
from typing import Any, BinaryIO, Optional, Sequence, Union

import geopandas as gpd
import numpy as np
import pandas as pd

from blocksnet.analysis.network.accessibility import area_accessibility
from blocksnet.analysis.indicators import calculate_development_indicators
from blocksnet.analysis.morphotypes import get_strelka_morphotypes
from blocksnet.config import log_config
from blocksnet.enums import LandUse
from blocksnet.machine_learning.regression import DensityRegressor
from blocksnet.relations import calculate_distance_matrix, generate_adjacency_graph

from . import constants as _constants
from .constants import ACCESSIBILITY_SPEED, BlockColumn

DataFrameLike = Union[pd.DataFrame, gpd.GeoDataFrame]
BlocksInput = Union[BinaryIO, DataFrameLike]


class LandDataPreparator:
    """Provide a reusable interface for preparing block-level geospatial data."""

    DEFAULT_OUTPUT_COLUMNS: Sequence[str] = _constants.DEFAULT_OUTPUT_COLUMNS

    def __init__(
        self,
        scenario_blocks_source: Optional[BlocksInput],
        context_blocks_source: Optional[BlocksInput],
        *,
        adjacency_radius: float = _constants.DEFAULT_ADJACENCY_RADIUS,
        sqm_per_person: float = _constants.DEFAULT_SQM_PER_PERSON,
        output_columns: Optional[Sequence[str]] = None,
        predict_project_only: bool = False,
        log_level: str = 'WARNING',
    ) -> None:
        """Create a preparator configured with scenario, context, and matrices.

        Parameters
        ----------
        scenario_blocks_source : BlocksInput
            Base source representing scenario blocks. When ``None``, all blocks
            are treated as context and marked ``is_project=False``.
        context_blocks_source : BlocksInput or None
            Base source representing context blocks. When ``None`` and
            ``scenario_blocks_source`` is provided, the scenario input is used
            as context and all blocks are marked ``is_project=False``.
        adjacency_radius : float, optional
            Radius (metres) for adjacency graph construction.
        sqm_per_person : float, optional
            Square metres per person used when estimating population.
        output_columns : Sequence[str], optional
            Desired output column ordering. Defaults to
            :data:`~urbanomy.methods.land_value_modeling.constants.DEFAULT_OUTPUT_COLUMNS`.
        predict_project_only : bool, optional
            When True, density indicators are predicted only for project blocks
            (``is_project``). Context blocks keep their existing density values
            (or zeros if missing).
        log_level : str, optional
            Logging level forwarded to ``blocksnet`` utilities.
        """
        self._scenario_source = scenario_blocks_source
        self._context_source = context_blocks_source
        self.adjacency_radius = adjacency_radius
        self.sqm_per_person = sqm_per_person
        self.output_columns = list(output_columns) if output_columns else list(self.DEFAULT_OUTPUT_COLUMNS)
        self._density_regressor = DensityRegressor()
        self._predict_project_only = predict_project_only
        self._last_prepared: Optional[gpd.GeoDataFrame] = None
        self._log_level = log_level

    @contextmanager
    def _temporary_log_level(self):
        """Temporarily set the shared blocksnet logger level for a call."""

        setter = getattr(log_config, "set_logger_level", None)
        if not callable(setter):
            yield
            return

        getter = getattr(log_config, "get_logger_level", None)
        previous_level = None
        if callable(getter):
            try:
                previous_level = getter()
            except Exception:  # pragma: no cover - defensive fallback
                previous_level = None

        try:
            setter(self._log_level)
        except Exception:  # pragma: no cover - defensive fallback
            previous_level = None

        try:
            yield
        finally:
            if previous_level is not None:
                try:
                    setter(previous_level)
                except Exception:  # pragma: no cover - defensive fallback
                    pass

    def prepare(
        self,
        scenario_blocks: Optional[BlocksInput] = None,
        context_blocks: Optional[BlocksInput] = None,
    ) -> gpd.GeoDataFrame:
        """Prepare scenario and context blocks with engineered features.

        Parameters
        ----------
        scenario_blocks : BlocksInput, optional
            Scenario blocks override. Falls back to ``scenario_blocks_source``
            when omitted.
        context_blocks : BlocksInput, optional
            Context blocks override. Falls back to ``context_blocks_source``
            when omitted. When neither scenario nor context is provided, a
            ``ValueError`` is raised.

        Returns
        -------
        geopandas.GeoDataFrame
            Prepared dataset containing engineered indicators and metadata.
        """
        with self._temporary_log_level():
            scenario_source = self._normalize_source(
                scenario_blocks if scenario_blocks is not None else self._scenario_source
            )
            context_source = self._normalize_source(
                context_blocks if context_blocks is not None else self._context_source
            )
            blocks = self._build_blocks(scenario_source, context_source)
            self._ensure_land_use_enum(blocks)
            self._clamp_land_use(blocks)
            adjacency_graph = generate_adjacency_graph(blocks, self.adjacency_radius)
            density_df = self._calculate_density(blocks, adjacency_graph)
            self._attach_development_indicators(blocks, density_df)
            self._append_morphotypes(blocks)
            self._append_accessibility(blocks)
            prepared = self._cleanup(blocks)
            prepared['id'] = prepared.index

        self._last_prepared = prepared.copy()
        return prepared

    def _build_blocks(
        self,
        scenario_source: Optional[BlocksInput],
        context_source: Optional[BlocksInput],
    ) -> gpd.GeoDataFrame:
        """Combine scenario and context blocks into a unified GeoDataFrame.

        Parameters
        ----------
        scenario_source : BlocksInput or None
            Scenario block data or stream. When ``None``, all blocks are treated
            as context.
        context_source : BlocksInput or None
            Context block data or stream. When ``None``, the scenario input is
            reused as context and all blocks are treated as non-project.

        Returns
        -------
        geopandas.GeoDataFrame
            Concatenated blocks with ``site_area`` and ``is_project`` columns.
        """
        if scenario_source is None and context_source is None:
            raise ValueError("At least one of scenario_blocks_source or context_blocks_source must be provided.")

        scenario = self._resolve_blocks_input(scenario_source) if scenario_source is not None else None
        context = self._resolve_blocks_input(context_source) if context_source is not None else None

        if context is None and scenario is not None:
            context = scenario
            scenario = self._empty_geo_like(context)
        elif scenario is None and context is not None:
            scenario = self._empty_geo_like(context)

        if scenario.crs and context.crs and scenario.crs != context.crs:
            context = context.to_crs(scenario.crs)
        geom_name = scenario.geometry.name
        crs = scenario.crs or context.crs
        blocks = pd.concat([scenario, context], ignore_index=True)
        blocks = gpd.GeoDataFrame(blocks, geometry=geom_name, crs=crs)
        blocks[BlockColumn.SITE_AREA.value] = blocks.geometry.area
        blocks[BlockColumn.IS_PROJECT.value] = LandDataPreparator.mark_scenario_blocks(blocks, scenario)
        return blocks

    def _resolve_blocks_input(
        self,
        source: BlocksInput,
    ) -> gpd.GeoDataFrame:
        """Load blocks data from either an in-memory object or binary stream.

        Parameters
        ----------
        source : BlocksInput
            Either a pandas/GeoPandas object or a binary file-like object that
            yields pickled data.

        Returns
        -------
        geopandas.GeoDataFrame
            Validated GeoDataFrame copy derived from the input ``source``.
        """
        loaded = self._load_dataframe_from_source(source)
        return self._ensure_geodataframe(loaded)

    @staticmethod
    def _normalize_source(source: Optional[BlocksInput]) -> Optional[BlocksInput]:
        """Treat explicit ``False`` as a request to disable the source."""
        if source is False:  # type: ignore[comparison-overlap]
            return None
        return source

    @staticmethod
    def _load_dataframe_from_source(source: BlocksInput) -> DataFrameLike:
        """Load a DataFrame or GeoDataFrame from the given input source.

        Parameters
        ----------
        source : BlocksInput
            DataFrame-like object or binary stream containing pickled data.

        Returns
        -------
        pandas.DataFrame or geopandas.GeoDataFrame
            Copy of the loaded structure.

        Raises
        ------
        TypeError
            If ``source`` is neither a DataFrame-like object nor a readable
            binary stream.
        """
        if isinstance(source, (pd.DataFrame, gpd.GeoDataFrame)):
            return source.copy()
        if hasattr(source, 'read'):
            binary_source = LandDataPreparator._reset_stream(source)
            return pd.read_pickle(binary_source)
        raise TypeError("Expected a GeoDataFrame/DataFrame or a binary stream with pickled data.")

    @staticmethod
    def _reset_stream(stream: BinaryIO) -> BinaryIO:
        """Rewind a binary stream to the beginning when supported.

        Parameters
        ----------
        stream : BinaryIO
            File-like object that may expose ``seek``.

        Returns
        -------
        BinaryIO
            The same stream, rewound to the beginning when possible.
        """
        seek = getattr(stream, 'seek', None)
        if callable(seek):
            seek(0)
        return stream

    @staticmethod
    def _ensure_geodataframe(data: DataFrameLike) -> gpd.GeoDataFrame:
        """Validate that input data can be represented as a GeoDataFrame.

        Parameters
        ----------
        data : pandas.DataFrame or geopandas.GeoDataFrame
            Input structure expected to include a ``geometry`` column.

        Returns
        -------
        geopandas.GeoDataFrame
            Copy of the data coerced to GeoDataFrame.

        Raises
        ------
        ValueError
            If the ``geometry`` column is missing.
        """
        if isinstance(data, gpd.GeoDataFrame):
            return data.copy()
        if 'geometry' not in data.columns:
            raise ValueError("The provided DataFrame must contain a 'geometry' column.")
        crs = getattr(data, 'crs', None)
        return gpd.GeoDataFrame(data.copy(), geometry='geometry', crs=crs)

    @staticmethod
    def _empty_geo_like(template: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Create an empty GeoDataFrame matching geometry name/CRS of template."""
        geom_name = template.geometry.name if hasattr(template, "geometry") else "geometry"
        columns = list(template.columns)
        if geom_name not in columns:
            columns.append(geom_name)
        empty = gpd.GeoDataFrame(columns=columns, geometry=geom_name, crs=getattr(template, "crs", None))
        empty[geom_name] = empty[geom_name].astype(object)
        return empty

    @staticmethod
    def mark_scenario_blocks(
        blocks: gpd.GeoDataFrame,
        scenario: gpd.GeoDataFrame,
    ) -> np.ndarray:
        """Compute a boolean mask identifying scenario blocks.

        Parameters
        ----------
        blocks : geopandas.GeoDataFrame
            Combined blocks dataset.
        scenario : geopandas.GeoDataFrame
            Scenario subset used to mark the blocks.

        Returns
        -------
        numpy.ndarray
            Boolean mask aligned to ``blocks.index`` with ``True`` for scenario
            polygons.
        """
        if scenario.empty:
            return np.zeros(len(blocks), dtype=bool)

        scenario_geometry = scenario[['geometry']]
        if scenario_geometry.crs != blocks.crs:
            scenario_geometry = scenario_geometry.to_crs(blocks.crs)

        joined = gpd.sjoin(
            blocks[['geometry']].reset_index().rename(columns={'index': '_idx'}),
            scenario_geometry,
            how='inner',
            predicate='intersects',
        )
        scenario_indices = joined['_idx'].unique()
        return blocks.index.isin(scenario_indices)

    def _clamp_land_use(self, blocks: gpd.GeoDataFrame) -> None:
        """Limit land-use share columns to the [0, 1] interval in-place.

        Parameters
        ----------
        blocks : geopandas.GeoDataFrame
            Blocks containing percentage share columns that correspond to
            :class:`~blocksnet.enums.LandUse` values.
        """
        for land_use in LandUse:
            column = land_use.value
            if column in blocks.columns:
                blocks[column] = blocks[column].clip(upper=1)

    def _ensure_land_use_enum(self, blocks: gpd.GeoDataFrame) -> None:
        """Coerce ``land_use`` column to ``LandUse`` enum values when possible."""
        land_use_column = BlockColumn.LAND_USE.value
        if land_use_column not in blocks.columns:
            return

        def coerce(value: Any) -> Any:
            if isinstance(value, LandUse) or value is None:
                return value

            text = str(value).strip()
            if not text:
                return value

            try:
                return LandUse(text)
            except ValueError:
                pass

            name = text.upper()
            if name.startswith("LANDUSE."):
                name = name.split(".", 1)[1]
            try:
                return LandUse[name]
            except KeyError:
                return value

        blocks[land_use_column] = blocks[land_use_column].apply(coerce)

    def _calculate_density(self, blocks: gpd.GeoDataFrame, adjacency_graph) -> pd.DataFrame:
        """Evaluate density indicators with basic post-processing.

        Parameters
        ----------
        blocks : geopandas.GeoDataFrame
            Blocks enriched with geometric fields.
        adjacency_graph : Any
            Graph describing neighbourhood relations between blocks.

        Returns
        -------
        pandas.DataFrame
            Density indicators aligned to ``blocks.index``.
        """
        project_only = self._predict_project_only
        target_blocks = blocks
        target_graph = adjacency_graph
        density_columns = [
            BlockColumn.FSI.value,
            BlockColumn.GSI.value,
            BlockColumn.MXI.value,
        ]

        if project_only:
            project_mask = blocks[BlockColumn.IS_PROJECT.value]
            target_blocks = blocks.loc[project_mask]
            target_graph = adjacency_graph.subgraph(target_blocks.index)

        if target_blocks.empty:
            predicted = pd.DataFrame(index=target_blocks.index, columns=density_columns, dtype=float)
        else:
            predicted = self._density_regressor.evaluate(target_blocks, target_graph).copy()

        density_df = pd.DataFrame(index=blocks.index, columns=density_columns, dtype=float)
        existing_cols = [col for col in density_columns if col in blocks.columns]
        if existing_cols:
            density_df.loc[:, existing_cols] = blocks[existing_cols]

        density_df.loc[target_blocks.index, :] = predicted
        for col in density_columns:
            density_df[col] = pd.to_numeric(density_df[col], errors='coerce')

        density_df[BlockColumn.FSI.value] = density_df[BlockColumn.FSI.value].clip(lower=0)
        density_df[BlockColumn.GSI.value] = density_df[BlockColumn.GSI.value].clip(lower=0, upper=1)
        density_df[BlockColumn.MXI.value] = density_df[BlockColumn.MXI.value].clip(lower=0, upper=1)
        density_df.loc[blocks[BlockColumn.RESIDENTIAL.value] == 0, BlockColumn.MXI.value] = 0
        density_df = density_df.fillna(0)

        return density_df

    def _attach_development_indicators(self, blocks: gpd.GeoDataFrame, density_df: pd.DataFrame) -> None:
        """Append development indicators derived from density metrics.

        Parameters
        ----------
        blocks : geopandas.GeoDataFrame
            Blocks dataset receiving the indicator columns.
        density_df : pandas.DataFrame
            Output of :meth:`_calculate_density` containing density metrics.
        """
        density_df = density_df.copy()
        density_df[BlockColumn.SITE_AREA.value] = blocks[BlockColumn.SITE_AREA.value]
        indicators = calculate_development_indicators(density_df)
        population = (indicators['living_area'] // self.sqm_per_person).fillna(0)
        indicators['population'] = population.astype(int)
        blocks.loc[:, indicators.columns] = indicators

    def _append_morphotypes(self, blocks: gpd.GeoDataFrame) -> None:
        """Join morphological classifications to the blocks dataset.

        Parameters
        ----------
        blocks : geopandas.GeoDataFrame
            Dataset whose rows will be annotated with morphotype labels.
        """
        morphotypes = get_strelka_morphotypes(blocks)
        blocks.loc[:, morphotypes.columns] = morphotypes

    def _append_accessibility(
        self,
        blocks: gpd.GeoDataFrame,
    ) -> None:
        """Attach area accessibility metrics to blocks in-place.

        Parameters
        ----------
        blocks : geopandas.GeoDataFrame
            Blocks dataset being enriched with accessibility metrics. The
            accessibility matrix is derived from block-to-block travel times.
        """
        matrix = self._compute_accessibility_matrix(blocks)
        area_acc = area_accessibility(matrix, blocks)
        blocks.loc[:, area_acc.columns] = area_acc

    def _compute_accessibility_matrix(self, blocks: gpd.GeoDataFrame) -> pd.DataFrame:
        """Calculate a travel-time accessibility matrix for the given blocks."""
        if blocks.empty:
            return pd.DataFrame(index=blocks.index, columns=blocks.index, dtype=float)

        projected_blocks = blocks
        utm_crs = blocks.estimate_utm_crs()
        if utm_crs:
            projected_blocks = blocks.to_crs(utm_crs)

        distance_matrix = calculate_distance_matrix(projected_blocks)
        if not isinstance(distance_matrix, pd.DataFrame):
            distance_matrix = pd.DataFrame(
                distance_matrix,
                index=blocks.index,
                columns=blocks.index,
            )
        return distance_matrix // ACCESSIBILITY_SPEED

    def _cleanup(self, blocks: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Remove intermediate columns and enforce output ordering.

        Parameters
        ----------
        blocks : geopandas.GeoDataFrame
            Dataset containing intermediate columns to be trimmed.

        Returns
        -------
        geopandas.GeoDataFrame
            Cleaned view limited to the configured ``output_columns``.
        """
        cleaned = blocks.copy()
        for prefix in ('capacity', 'count'):
            drop_cols = [col for col in cleaned.columns if col.startswith(prefix)]
            if drop_cols:
                cleaned = cleaned.drop(columns=drop_cols)
        geom_col = cleaned.geometry.name
        keep_cols = [col for col in self.output_columns if col in cleaned.columns]
        scenario_flag = BlockColumn.IS_PROJECT.value
        if scenario_flag in cleaned.columns and scenario_flag not in keep_cols:
            keep_cols.append(scenario_flag)
        ordered_cols = keep_cols + ([geom_col] if geom_col not in keep_cols else [])
        return cleaned[ordered_cols].copy()
