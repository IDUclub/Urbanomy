"""Shared configuration for land value modeling workflows."""

from __future__ import annotations

from enum import Enum
from typing import Sequence


ORIGINAL_FEATURES: Sequence[str] = (
    "residential",
    "business",
    "recreation",
    "industrial",
    "transport",
    "special",
    "agriculture",
    "share",
    "footprint_area",
    "build_floor_area",
    "living_area",
    "non_living_area",
    "population",
    "site_area",
    "fsi",
    "gsi",
    "mxi",
    "l",
    "area_accessibility",
    "land_use",
    "morphotype",
)
"""Ordered list of base features used by the land price model."""


CATEGORICAL_FEATURES: Sequence[str] = ("land_use", "morphotype")
"""Subset of features that should be interpreted as categorical."""


RADIUS_LIST: Sequence[float] = (500, 1000, 2000, 3000)
"""Default distance thresholds (in metres) for spatial lag computation."""


DEFAULT_OUTPUT_COLUMNS: Sequence[str] = ORIGINAL_FEATURES
"""Default set of columns produced by land data preparation."""


SERVICE_FEATURES: Sequence[str] = (
    "count_animal_shelter",
    "count_bakery",
    "count_bank",
    "count_bar",
    "count_beach",
    "count_brewery",
    "count_buildings",
    "count_bus_station",
    "count_bus_stop",
    "count_cafe",
    "count_cemetery",
    "count_cinema",
    "count_circus",
    "count_convenience",
    "count_dog_park",
    "count_fuel",
    "count_government",
    "count_greenhouse_complex",
    "count_guest_house",
    "count_hairdresser",
    "count_hospital",
    "count_hostel",
    "count_hotel",
    "count_kindergarten",
    "count_landfill",
    "count_lawyer",
    "count_machine_building_plant",
    "count_mall",
    "count_market",
    "count_multifunctional_center",
    "count_museum",
    "count_park",
    "count_parking",
    "count_pharmacy",
    "count_pier",
    "count_pitch",
    "count_plant_nursery",
    "count_playground",
    "count_police",
    "count_polyclinic",
    "count_post",
    "count_prison",
    "count_recruitment",
    "count_religion",
    "count_reserve",
    "count_restaurant",
    "count_sanatorium",
    "count_school",
    "count_substation",
    "count_subway_entrance",
    "count_supermarket",
    "count_swimming_pool",
    "count_theatre",
    "count_train_building",
    "count_train_station",
    "count_university",
    "count_warehouse",
    "count_wastewater_plant",
    "count_water_works",
    "count_woodworking_plant",
    "count_zoo",
    "osr",
    "share_living",
    "share_non_living",
)
"""Optional service-related features that can be appended to ``ORIGINAL_FEATURES``."""


DEFAULT_ADJACENCY_RADIUS: int = 10
"""Default adjacency radius (metres) used to build block graphs."""


DEFAULT_SQM_PER_PERSON: float = 20.0
"""Default number of square metres per person when estimating population."""


ACCESSIBILITY_SPEED: float = 5 * 1_000 / 60
"""Walking speed (metres per minute) assumed when computing accessibility."""


class BlockColumn(str, Enum):
    """Canonical column identifiers used across land value workflows."""

    ID = "id"
    LAND_USE = "land_use"
    SHARE = "share"
    FOOTPRINT_AREA = "footprint_area"
    BUILD_FLOOR_AREA = "build_floor_area"
    LIVING_AREA = "living_area"
    NON_LIVING_AREA = "non_living_area"
    POPULATION = "population"
    SITE_AREA = "site_area"
    FSI = "fsi"
    GSI = "gsi"
    MXI = "mxi"
    L = "l"
    OSR = "osr"
    SHARE_LIVING = "share_living"
    SHARE_NON_LIVING = "share_non_living"
    RESIDENTIAL = "residential"
    IS_PROJECT = "is_project"


class ScenarioResultKey(str, Enum):
    """Named keys returned by scenario impact helpers."""

    MAP = "map"
    MAP_ALL = "map_all"
    FIGURE = "fig"
    SUMMARY = "summary"
    SUMMARY_ALL = "summary_all"
