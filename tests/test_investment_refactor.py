from unittest.mock import patch

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from shapely.geometry import box

from urbanomy.methods.investment_potential import InvestmentAttractivenessAnalyzer, LandUseScoreAnalyzer
from urbanomy.methods.investment_potential.constants import DEFAULT_BENCHMARKS_RU, LAND_USE_CONFIGS
from urbanomy.methods.investment_potential.input import prepare_investment_input
from urbanomy.methods.land_value_modeling import ScenarioTEPModifier, transfer_baseline_prices


def test_scores_use_original_attributes_even_when_score_names_overlap():
    frame = gpd.GeoDataFrame(
        {"residential": [0.2, np.nan], "business": [0.3, np.nan], "attribute": [2.0, np.nan]},
        geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)],
        crs=3857,
    )
    for config in LAND_USE_CONFIGS.values():
        frame[config.potential_column] = 5.0
    before = frame.copy()
    analyzer = LandUseScoreAnalyzer(weights={"residential": {"default": 1}, "business": {"default": 2}})
    result = analyzer._compute_wide(frame)
    assert result.loc[0, "residential"] == round((0.2 + 0.3 + 2.0) / 3, 1)
    assert result.loc[0, "business"] == round(2 * (0.2 + 0.3 + 2.0) / 3, 1)
    assert pd.isna(result.loc[1, "residential"])
    assert_frame_equal(frame, before)


def test_scores_handle_missing_potentials_and_zero_weights():
    frame = gpd.GeoDataFrame(
        {"attribute": [2.0, 4.0]},
        geometry=[None, None],
        crs=3857,
    )
    residential = next(config for lu, config in LAND_USE_CONFIGS.items() if lu.value == "residential")
    frame[residential.potential_column] = [5.0, np.nan]
    analyzer = LandUseScoreAnalyzer(weights={"residential": {"default": 0}})
    result = analyzer.compute_scores_long(frame)
    assert result.ip_type.tolist() == ["residential"]
    assert result.spatial_potential.tolist() == [0.0]


def project_blocks():
    return gpd.GeoDataFrame(
        {
            "id": [86, 87],
            "land_use": ["residential", "business"],
            "land_value": [1_000_000.0, 2_000_000.0],
            "site_area": [1000.0, 2000.0],
            "residential": [0.8, 0.0],
            "business": [0.2, 1.0],
            "build_floor_area": [200.0, 300.0],
            "footprint_area": [100.0, 100.0],
            "living_area": [100.0, 0.0],
            "non_living_area": [100.0, 300.0],
            "l": [2.0, 3.0],
            "fsi": [0.2, 0.15],
            "mxi": [0.5, 0.0],
            "morphotype": ["low-rise model", "low-rise non-residential"],
        },
        geometry=[box(0, 0, 10, 100), box(10, 0, 30, 100)],
        index=[10, 30],
        crs=3857,
    )


def test_investment_preparation_never_mutates_input():
    frame = project_blocks()
    original = frame.copy()
    prepare_investment_input(frame, show_warning=False)
    assert_frame_equal(frame, original)


def test_quiet_investment_calculation_skips_unused_totals(capsys):
    frame = project_blocks()
    frame["is_project"] = True
    analyzer = InvestmentAttractivenessAnalyzer()
    with patch.object(analyzer, "_aggregate_cashflows", wraps=analyzer._aggregate_cashflows) as aggregate:
        quiet = analyzer.calculate_investment_metrics(frame, show_project_totals=False, show_warning=False)
        aggregate.assert_not_called()
        aggregate.reset_mock()
        verbose = analyzer.calculate_investment_metrics(frame, show_project_totals=True, show_warning=False)
        aggregate.assert_called_once()
    assert_frame_equal(quiet, verbose)
    assert "Project totals:" in capsys.readouterr().out


def test_scenario_keeps_context_and_baseline_fields(monkeypatch):
    frame = project_blocks()
    original = frame.copy()
    seen = []

    def morphotypes(target):
        seen.append(target.index.tolist())
        return pd.DataFrame({"morphotype": ["updated"]}, index=target.index)

    monkeypatch.setattr(
        "urbanomy.methods.land_value_modeling.scenario_modification.get_strelka_morphotypes", morphotypes
    )
    modified = ScenarioTEPModifier(frame).apply(86, {"build_floor_area": 400.0, "living_area": 200.0})
    assert seen == [[10]]
    assert modified.loc[10, "morphotype"] == "updated"
    assert modified.loc[10, "build_floor_area_before"] == 200.0
    assert modified.loc[10, "land_use_before"] == "residential"
    assert_frame_equal(modified.loc[[30], frame.columns], frame.loc[[30]], check_dtype=False)
    assert_frame_equal(frame, original)


def test_scenario_accepts_required_features_supplied_by_changes():
    frame = project_blocks().drop(columns=["build_floor_area", "living_area"])
    result = ScenarioTEPModifier(frame).apply(86, {"build_floor_area": 400.0, "living_area": 200.0})
    assert result.loc[10, "build_floor_area"] == 400.0
    assert result.loc[10, "living_area"] == 200.0


def test_scenario_plot_computes_percentage_change_when_missing():
    import matplotlib.pyplot as plt

    from urbanomy.methods.land_value_modeling import plot_scenario_impact

    frame = project_blocks()
    frame["land_value_before"] = [1_000_000.0, 1_000_000.0]
    frame["land_value"] = [1_100_000.0, 900_000.0]
    result = plot_scenario_impact(
        blocks=frame,
        target_idx=86,
        buffer_radius=100,
        show=False,
        print_summary=False,
        figsize=(5, 4),
    )
    assert result["summary_all"]["sum_before"] == 2_000_000.0
    assert result["summary_all"]["sum_after"] == 2_000_000.0
    np.testing.assert_allclose(result["map_all"]["d_pct"], [10.0, -10.0])
    plt.close(result["fig"])


def test_investment_uses_predicted_baseline_price_and_existing_buildings():
    before = project_blocks()
    before["land_value"] = [1_000_000.0, 2_000_000.0]
    after = ScenarioTEPModifier(before.drop(columns="land_value")).apply(
        86,
        {
            "residential": 1.0,
            "business": 0.0,
            "build_floor_area": 400.0,
            "living_area": 200.0,
            "non_living_area": 200.0,
        },
    )
    assert "land_value" not in after.columns
    after_pred = after.copy()
    after_pred["land_value"] = [3_000_000.0, 2_000_000.0]
    full_value = transfer_baseline_prices(after_pred, before, scenario_mode=False)
    project = full_value.loc[full_value["id"] == 86].copy()
    project["is_project"] = True

    summary = InvestmentAttractivenessAnalyzer().calculate_investment_metrics(
        project,
        show_project_totals=False,
        show_warning=False,
    )

    assert len(summary) == 1
    assert project["land_value"].iloc[0] == 3_000_000.0
    assert summary["land_value"].iloc[0] == 1_000_000.0
    assert summary["demolition_cost"].iloc[0] == 200.0 * 900.0
    assert summary["construction_cost"].iloc[0] == 400.0 * 45_000.0
    assert summary["investment_need"].iloc[0] == 1_000_000.0 + 180_000.0 + 18_000_000.0


@pytest.mark.parametrize("existing_area", [0.0, 200.0])
def test_demolition_is_charged_once_in_initial_cashflow(existing_area):
    frame = project_blocks().iloc[[0]].copy()
    frame["build_floor_area_before"] = existing_area
    frame["is_project"] = True
    with_demolition = InvestmentAttractivenessAnalyzer().calculate_investment_metrics(
        frame,
        show_project_totals=False,
        show_warning=False,
    )
    zero_demolition = {
        land_use: {**profile, "cost_demolition": 0} for land_use, profile in DEFAULT_BENCHMARKS_RU.items()
    }
    without_demolition = InvestmentAttractivenessAnalyzer(benchmarks=zero_demolition).calculate_investment_metrics(
        frame,
        show_project_totals=False,
        show_warning=False,
    )
    expected = existing_area * 900.0
    assert with_demolition["demolition_cost"].iloc[0] == expected
    assert with_demolition["NPV"].iloc[0] == pytest.approx(without_demolition["NPV"].iloc[0] - expected, abs=0.02)
    assert with_demolition["investment_need"].iloc[0] == without_demolition["investment_need"].iloc[0] + expected
