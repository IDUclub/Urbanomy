"""Scenario tools for adjusting block indicators and analysing impact."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from blocksnet.analysis.morphotypes import get_strelka_morphotypes
from blocksnet.enums import LandUse
from geopandas import GeoDataFrame

from .constants import DEFAULT_SQM_PER_PERSON, BlockColumn


class ScenarioTEPModifier:
    """Apply scenario changes to a single block.

    Parameters
    ----------
    blocks : geopandas.GeoDataFrame
        Source blocks dataset that will be copied during each scenario
        application.
    """

    def __init__(self, blocks: GeoDataFrame) -> None:
        """Store a reference blocks dataset used as the scenario baseline.

        Parameters
        ----------
        blocks : geopandas.GeoDataFrame
            Source blocks that will be copied and modified per scenario.
        """
        self._blocks = blocks

    def apply(
        self,
        target_id: Any,
        changes: Mapping[str, Any],
        *,
        target_id_column: str = BlockColumn.ID.value,
    ) -> GeoDataFrame:
        """Return a modified copy of the blocks with updated TEP values.

        Parameters
        ----------
        target_id : Any
            Value from ``target_id_column`` identifying the block to modify.
        changes : Mapping[str, object]
            Dictionary of field updates (e.g. ``{"land_use": LandUse.RESIDENTIAL}``
            or an equivalent string token).
        target_id_column : str, optional
            Name of the identifier column used to locate the target block.

        Returns
        -------
        geopandas.GeoDataFrame
            Modified copy of the original blocks DataFrame.
        """

        df = self._blocks.copy()
        resolved_index = self._resolve_target_index(df, target_id, target_id_column)

        original_row = df.loc[resolved_index].copy()
        row = original_row.to_dict()
        normalised_changes = dict(changes)

        land_use_key = BlockColumn.LAND_USE.value
        if land_use_key in normalised_changes:
            normalised_changes[land_use_key] = self._coerce_land_use(normalised_changes[land_use_key])

        row.update(normalised_changes)

        site_area_key = BlockColumn.SITE_AREA.value
        build_key = BlockColumn.BUILD_FLOOR_AREA.value
        living_key = BlockColumn.LIVING_AREA.value
        non_living_key = BlockColumn.NON_LIVING_AREA.value
        mxi_key = BlockColumn.MXI.value
        footprint_key = BlockColumn.FOOTPRINT_AREA.value
        population_key = BlockColumn.POPULATION.value
        share_key = BlockColumn.SHARE.value
        residential_key = BlockColumn.RESIDENTIAL.value
        site_area = float(row[site_area_key])

        build = float(row[build_key])
        live = float(row[living_key])
        non = row.get(non_living_key, np.nan)

        build = max(build, 0.0)
        if living_key not in changes and ((mxi_key in changes) or (mxi_key in row)):
            mxi = float(row[mxi_key])
            mxi = float(np.clip(mxi, 0.0, 1.0))
            live = mxi * build
        live = float(np.clip(live, 0.0, build))

        if not np.isfinite(non):
            non = max(build - live, 0.0)
        else:
            non = max(float(non), 0.0)

        if (non_living_key in changes and living_key in changes) and (build_key not in changes):
            build = live + float(non)

        row[build_key] = build
        row[living_key] = live
        row[non_living_key] = float(non)

        footprint = float(row[footprint_key])
        row[population_key] = live / DEFAULT_SQM_PER_PERSON if DEFAULT_SQM_PER_PERSON > 0 else 0.0

        row[BlockColumn.FSI.value] = build / site_area if site_area > 0 else np.nan
        row[BlockColumn.GSI.value] = footprint / site_area if site_area > 0 else np.nan
        row[mxi_key] = (row[living_key] / build) if build > 0 else np.nan
        row[BlockColumn.L.value] = (build / footprint) if footprint > 0 else np.nan
        # row[BlockColumn.OSR.value] = (site_area - footprint) / site_area if site_area > 0 else np.nan

        if residential_key in normalised_changes and share_key not in normalised_changes:
            row[share_key] = float(row[residential_key])

        # Preserve pre-scenario values needed for demolition-cost estimates.
        row["land_use_before"] = original_row.get("land_use_before", original_row.get(land_use_key))
        row["build_floor_area_before"] = original_row.get(
            "build_floor_area_before",
            original_row.get(build_key),
        )

        for key, value in row.items():
            df.at[resolved_index, key] = value

        # Keep morphotype consistent with the updated block TEP.
        try:
            morphotypes = get_strelka_morphotypes(df.loc[[resolved_index]])
            if "morphotype" in morphotypes.columns and resolved_index in morphotypes.index:
                df.at[resolved_index, "morphotype"] = morphotypes.at[resolved_index, "morphotype"]
        except Exception:
            # Do not break scenario application if morphotype recomputation fails.
            pass

        return df

    @staticmethod
    def _resolve_target_index(df: GeoDataFrame, target_id: Any, target_id_column: str) -> Any:
        if target_id_column not in df.columns:
            raise KeyError(f"Target id column {target_id_column!r} is not present in the dataset.")

        matches = df.index[df[target_id_column] == target_id]
        if len(matches) == 0:
            raise KeyError(f"Block with {target_id_column}={target_id!r} is not present in the dataset.")
        if len(matches) > 1:
            raise ValueError(
                f"Multiple blocks found for {target_id_column}={target_id!r}; the identifier must be unique."
            )
        return matches[0]

    @staticmethod
    def _coerce_land_use(value: Any) -> LandUse:
        """Convert user-provided land-use tokens to ``LandUse`` enum values."""
        if isinstance(value, LandUse):
            return value
        if value is None:
            raise ValueError("land_use cannot be None")

        text = str(value).strip()
        if not text:
            raise ValueError("land_use cannot be empty")

        try:
            return LandUse(text)
        except ValueError:
            pass

        upper = text.upper()
        if upper.startswith("LANDUSE."):
            upper = upper.split(".", 1)[1]

        try:
            return LandUse[upper]
        except KeyError as exc:
            raise ValueError(f"Unknown land_use value: {value!r}") from exc


# Keep the former import path available without loading plotting on every scenario.
__all__ = ["ScenarioTEPModifier", "plot_scenario_impact"]


def __getattr__(name: str):
    if name == "plot_scenario_impact":
        from .scenario_visualization import plot_scenario_impact

        return plot_scenario_impact
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
