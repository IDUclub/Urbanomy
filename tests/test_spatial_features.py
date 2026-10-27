from unittest.mock import patch

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from libpysal.weights import DistanceBand, lag_spatial
from pandas.testing import assert_frame_equal
from shapely.geometry import box

from urbanomy.methods.land_value_modeling import DistrictProblem, LandPriceEstimator
from urbanomy.methods.land_value_modeling._spatial_features import build_distance_weights, compute_lag_features
from urbanomy.methods.land_value_modeling.training_utils import add_lags_full_df, build_radii_weights


@pytest.fixture
def blocks():
    return gpd.GeoDataFrame(
        {"id": [86, 87, 88, 89], "value": [1.0, np.nan, 3.0, 4.0], "site_area": [1.0] * 4},
        geometry=[box(x, 0, x + 0.5, 0.5) for x in [0, 1, 2, 10]],
        index=[10, 30, 20, 40],
        crs=3857,
    )


class SumModel:
    feature_names_ = None

    def predict(self, matrix):
        self.matrix = matrix.copy()
        return np.log1p(matrix.sum(axis=1))


def estimator(blocks, model=None, radii=(1.5, 20)):
    return LandPriceEstimator(
        blocks=blocks,
        model=SumModel() if model is None else model,
        orig_features=["value"],
        categorical_features=[],
        radius_list=radii,
    )


def test_batched_lags_match_columnwise_reference_with_missing_values(blocks):
    blocks["empty"] = np.nan
    blocks["nullable"] = pd.array([1, None, 3, 4], dtype="Float64")
    features = ["value", "empty", "nullable"]
    weights = build_distance_weights(blocks, [1.5, 20])
    expected = {}
    for feature in features:
        values = blocks[feature].fillna(blocks[feature].mean()).to_numpy(dtype=float, na_value=np.nan)
        for radius, weight in weights.items():
            expected[f"lag{radius}_{feature}"] = lag_spatial(weight, values)
    for radius, weight in weights.items():
        expected[f"n_neighbors_{radius}"] = [len(weight.neighbors[i]) for i in blocks.index]
    reference = pd.DataFrame(expected, index=blocks.index)
    actual = compute_lag_features(blocks, features, weights)
    assert_frame_equal(actual, reference, check_dtype=False)
    training = add_lags_full_df(blocks, weights=weights, numeric_cols=features + ["absent"])
    assert_frame_equal(training[actual.columns], actual)


def test_training_weight_builder_preserves_keyword_arguments(blocks):
    weights = build_radii_weights(df=blocks, radii=[1.5])
    assert weights[1.5].id_order == list(blocks.index)


def test_lags_without_numeric_features_still_include_neighbor_counts(blocks):
    weights = build_distance_weights(blocks, [1.5])
    actual = compute_lag_features(blocks, [], weights)
    assert actual.columns.tolist() == ["n_neighbors_1.5"]
    assert actual.iloc[:, 0].tolist() == [1, 2, 1, 0]
    assert compute_lag_features(blocks, [], {}).empty


def test_scenario_predictions_reuse_weights_without_stale_features(blocks):
    before = blocks.copy()
    model = SumModel()
    with patch.object(DistanceBand, "from_dataframe", wraps=DistanceBand.from_dataframe) as build:
        predictor = estimator(blocks, model)
        initial = predictor.predict()
        changed = blocks.copy()
        changed.loc[10, "value"] = 10.0
        scenario = predictor.predict(blocks=changed)
        assert build.call_count == 2
    fresh = estimator(changed).predict()
    assert_frame_equal(scenario, fresh)
    assert not np.allclose(initial.land_value, scenario.land_value)
    assert_frame_equal(predictor.predict(), initial)
    assert_frame_equal(blocks, before)
    assert "land_value" not in changed.columns


@pytest.mark.parametrize("change", ["geometry", "index", "crs", "order"])
def test_spatial_weights_rebuild_when_layout_changes(blocks, change):
    predictor = estimator(blocks)
    changed = blocks.copy()
    if change == "geometry":
        changed.loc[10, "geometry"] = box(100, 0, 100.5, 0.5)
    elif change == "index":
        changed.index = changed.index + 100
    elif change == "crs":
        changed = changed.set_crs(3395, allow_override=True)
    else:
        changed = changed.iloc[::-1]
    with patch.object(DistanceBand, "from_dataframe", wraps=DistanceBand.from_dataframe) as build:
        result = predictor.predict(blocks=changed)
        assert build.call_count == 2
    assert_frame_equal(result, estimator(changed).predict())


def test_model_feature_order_and_extra_columns_are_preserved(blocks):
    model = SumModel()
    model.feature_names_ = ["lag20_value", "value"]
    predictor = estimator(blocks, model)
    predictor.predict()
    assert model.matrix.columns.tolist() == model.feature_names_
    model.feature_names_ = ["missing_feature"]
    with pytest.raises(ValueError, match="missing_feature"):
        predictor.predict()
    with pytest.raises(ValueError, match="required features"):
        predictor.predict(blocks=blocks.drop(columns="value"))


def test_optimizer_reuses_estimator_and_invalidates_configuration(blocks):
    model = SumModel()
    problem = DistrictProblem(
        blocks=blocks,
        model=model,
        estimator_kwargs={},
        constraints={"value": {"min": 0, "max": 10}},
        target_id=86,
    )
    with patch.object(DistanceBand, "from_dataframe", wraps=DistanceBand.from_dataframe) as build:
        first = problem.evaluate_catboost(blocks, model, ["value"], [], [1.5])
        changed = blocks.copy()
        changed.loc[10, "value"] = 10.0
        second = problem.evaluate_catboost(changed, model, ["value"], [], [1.5])
        assert build.call_count == 1
        assert first != second
        problem.evaluate_catboost(changed, model, ["value"], [], [20])
        assert build.call_count == 2
        problem.evaluate_catboost(changed, SumModel(), ["value"], [], [20])
        assert build.call_count == 3
