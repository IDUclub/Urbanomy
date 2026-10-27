"""Public land-value utilities, loaded only when requested."""

from importlib import import_module


_EXPORTS = {
    "CATEGORICAL_FEATURES": "constants",
    "DEFAULT_ADJACENCY_RADIUS": "constants",
    "DEFAULT_OUTPUT_COLUMNS": "constants",
    "DEFAULT_SQM_PER_PERSON": "constants",
    "ORIGINAL_FEATURES": "constants",
    "RADIUS_LIST": "constants",
    "LandDataPreparator": "land_data_preparation",
    "LandPriceEstimator": "land_price_estimation",
    "transfer_baseline_prices": "land_price_estimation",
    "plot_land_price_maps": "land_price_visualization",
    "build_pareto_front_dataframe": "pareto",
    "ScenarioTEPModifier": "scenario_modification",
    "plot_scenario_impact": "scenario_visualization",
    "DistrictProblem": "ga_mc_optimizer",
    "Evaluation": "strategic_alignment",
    "StrategicAlignmentScorer": "strategic_alignment",
    "run_nsga2_with_strategic_alignment": "ga_mc_optimizer",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{_EXPORTS[name]}", __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
