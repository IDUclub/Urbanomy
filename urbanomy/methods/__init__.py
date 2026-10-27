"""Lightweight exports for Urbanomy method modules."""

from importlib import import_module

__all__ = [
    "agent",
    "investment_potential",
    "land_value_modeling",
    "Agent",
    "llm",
    "embedding",
]


def __getattr__(name: str):
    if name in {"agent", "Agent", "llm", "embedding"}:
        module = import_module(".agent", __name__)
        if name == "agent":
            return module
        return getattr(module, name)
    if name == "investment_potential":
        return import_module(".investment_potential", __name__)
    if name == "land_value_modeling":
        return import_module(".land_value_modeling", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
