# investment_potential/__init__.py
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _v


try:
    __version__ = _v("investment_potential")
except PackageNotFoundError:  # fallback used during local development
    __version__ = "0.0.1"

from importlib import import_module


_EXPORTS = {
    "LandUseScoreAnalyzer": ".land_use_score",
    "InvestmentAttractivenessAnalyzer": ".investment_metrics",
    "calculate_investment_metrics": ".investment_metrics",
    "DEFAULT_BENCHMARKS_RU": ".constants",
    "LAND_USE_TO_POTENTIAL_COLUMN": ".constants",
    "prepare_investment_input": ".input",
}
__all__ = list(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_EXPORTS[name], __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
