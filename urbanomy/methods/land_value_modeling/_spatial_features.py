"""Spatial feature calculations shared by training and inference."""

from typing import Iterable, Mapping, Sequence

import pandas as pd
from libpysal.weights import DistanceBand, lag_spatial


def build_distance_weights(df, radii: Iterable[float]) -> dict[float, DistanceBand]:
    weights = {}
    for radius in radii:
        weight = DistanceBand.from_dataframe(df, threshold=radius, binary=True, silence_warnings=True, use_index=True)
        weight.transform = "r"
        weights[radius] = weight
    return weights


def compute_lag_features(
    blocks: pd.DataFrame,
    numeric_features: Sequence[str],
    weights: Mapping[float, DistanceBand],
) -> pd.DataFrame:
    numeric = blocks[list(numeric_features)]
    values = numeric.fillna(numeric.mean(numeric_only=True)).to_numpy(dtype=float, na_value=float("nan"))
    # One sparse matrix multiplication per radius for all numeric features.
    lagged = {radius: lag_spatial(weight, values) for radius, weight in weights.items()}
    columns = {
        f"lag{radius}_{feature}": lagged[radius][:, i]
        for i, feature in enumerate(numeric_features)
        for radius in weights
    }
    columns.update(
        {
            f"n_neighbors_{radius}": [len(weight.neighbors[index]) for index in blocks.index]
            for radius, weight in weights.items()
        }
    )
    return pd.DataFrame(columns, index=blocks.index)
