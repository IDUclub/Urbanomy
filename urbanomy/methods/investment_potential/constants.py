from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

from blocksnet.enums import LandUse


@dataclass(frozen=True)
class LandUseConfig:
    """Immutable container for land-use related parameters."""

    potential_column: str
    indicator_weights: dict[str, float] = field(default_factory=dict)
    investment_weights: tuple[float, float] = (0.0, 0.0)


LAND_USE_CONFIGS: Final[dict[LandUse, LandUseConfig]] = {
    LandUse.RESIDENTIAL: LandUseConfig(
        potential_column="Потенциал развития жилой застройки",
        indicator_weights={
            "Население": 1.3,
            "Социальное обеспечение": 1.4,
            "Экологическая ситуация": 1.5,
            "Средняя доступность до близлежащего крупного населенного пункта": 1.2,
            "Транспортное обеспечение": 1.1,
            "default": 1.0,
        },
        investment_weights=(0.4, 0.6),
    ),
    LandUse.BUSINESS: LandUseConfig(
        potential_column=(
            "Потенциал развития застройки общественно-деловой зоны"
        ),
        indicator_weights={
            "Транспортное обеспечение": 1.5,
            "Население": 1.4,
            "Социальное обеспечение (комфорт)": 1.3,
            "Средняя доступность до близлежащего крупного населенного пункта": 1.2,
            "default": 1.0,
        },
        investment_weights=(0.3, 0.7),
    ),
    LandUse.RECREATION: LandUseConfig(
        potential_column="Потенциал развития застройки рекреационной зоны",
        indicator_weights={
            "Экологическая ситуация": 1.5,
            "Социальное обеспечение (комфорт)": 1.4,
            "Транспортное обеспечение": 1.2,
            "Население": 0.8,
            "default": 1.0,
        },
        investment_weights=(0.6, 0.4),
    ),
    LandUse.SPECIAL: LandUseConfig(
        potential_column=(
            "Потенциал развития застройки зоны специального назначения"
        ),
        indicator_weights={
            "Потенциал размещения порта": 1.5,
            "Транспортное обеспечение": 1.4,
            "Потенциал размещения логистического, складского комплекса": 1.3,
            "default": 1.0,
        },
        investment_weights=(0.3, 0.7),
    ),
    LandUse.INDUSTRIAL: LandUseConfig(
        potential_column="Потенциал развития застройки промышленной зоны",
        indicator_weights={
            "Потенциал размещения логистического, складского комплекса": 1.5,
            "Транспортное обеспечение": 1.4,
            "Экологическая ситуация": 0.8,
            "Население": 0.9,
            "default": 1.0,
        },
        investment_weights=(0.35, 0.65),
    ),
    LandUse.AGRICULTURE: LandUseConfig(
        potential_column="Потенциал развития застройки сельскохозяйственной зоны",
        indicator_weights={
            "Экологическая ситуация": 1.5,
            "Население": 0.8,
            "Транспортное обеспечение": 1.2,
            "Средняя доступность до близлежащего крупного населенного пункта": 1.1,
            "default": 1.0,
        },
        investment_weights=(0.6, 0.4),
    ),
    LandUse.TRANSPORT: LandUseConfig(
        potential_column="Потенциал развития застройки транспортной зоны",
        indicator_weights={
            "Потенциал размещения логистического, складского комплекса": 1.5,
            "Количество аэропортов местного значения": 1.4,
            "Средняя доступность до близлежащего крупного населенного пункта": 1.3,
            "default": 1.0,
        },
        investment_weights=(0.35, 0.65),
    ),
}


LAND_USE_TO_POTENTIAL_COLUMN: Final[dict[str, str]] = {
    land_use.value: config.potential_column for land_use, config in LAND_USE_CONFIGS.items()
}

LAND_USE_WEIGHTS: Final[dict[str, dict[str, float]]] = {
    land_use.value: config.indicator_weights for land_use, config in LAND_USE_CONFIGS.items()
}

INVESTMENT_WEIGHTS: Final[dict[str, tuple[float, float]]] = {
    land_use.value: config.investment_weights for land_use, config in LAND_USE_CONFIGS.items()
}

DEFAULT_BENCHMARKS_RU: Final[dict[LandUse, dict[str, float | int]]] = {
    LandUse.RESIDENTIAL: {
        "cost_build": 45_000,
        "price_sale": 140_000,
        "construction_years": 3,
        "sale_years": 4,
        "opex_rate": 800,
        "cost_demolition": 900,
    },
    LandUse.BUSINESS: {
        "cost_build": 55_000,
        "rent_annual": 25_000,
        "rent_years": 15,
        "construction_years": 3,
        "opex_rate": 1_300,
        "cost_demolition": 900,
    },
    LandUse.RECREATION: {
        "cost_build": 25_000,
        "rent_annual": 7_500,
        "rent_years": 15,
        "construction_years": 3,
        "opex_rate": 1_000,
        "cost_demolition": 900,
    },
    LandUse.SPECIAL: {
        "cost_build": 35_000,
        "rent_annual": 11_000,
        "rent_years": 15,
        "construction_years": 3,
        "opex_rate": 1_500,
        "cost_demolition": 900,
    },
    LandUse.INDUSTRIAL: {
        "cost_build": 38_000,
        "rent_annual": 14_800,
        "rent_years": 15,
        "construction_years": 3,
        "opex_rate": 700,
        "cost_demolition": 900,
    },
    LandUse.AGRICULTURE: {
        "cost_build": 25_000,
        "rent_annual": 6_500,
        "rent_years": 15,
        "construction_years": 3,
        "opex_rate": 300,
        "cost_demolition": 900,
    },
    LandUse.TRANSPORT: {
        "cost_build": 18_000,
        "rent_annual": 8_200,
        "rent_years": 15,
        "construction_years": 3,
        "opex_rate": 600,
        "cost_demolition": 900,
    },
}


DEFAULT_ECON_METRIC: str = "EI"
DEFAULT_DISCOUNT_RATE: float = 0.18
DEFAULT_AREA_COL: str = "Площадь территории"
DEFAULT_IP_TYPE: str = "ip_type"
DEFAULT_IP_VALUE: str = "spatial_potential"


SUMMARY_COLUMNS: Final[tuple[str, ...]] = (
    "land_use",
    "land_area",
    "built_area",
    "land_value",
    "demolition_cost",
    "construction_cost",
    "investment_need",
    "NPV",
    "IRR",
    "PI",
    "PP_years",
    "EI",
)
