"""Competitive Galcon simulator, tactical solver, PlanetTimelineTransformer bot, and gbots adapter.

Cross-pollinates the variable-N entity + arrival-timeline Transformer from ``orbit_wars_bot``
directly to classic Galcon / Phil Hassey's ``gbots`` protocol.
"""

from galcon.fast_sim_cpp import HAS_FAST_SIM_CPP, FastGalconCppSim
from galcon.sim import (
    ITEMS_MAX,
    ITEMS_RESERVED,
    MIN_PLANET_GAP,
    SHIP_RADIUS,
    SHIP_TURN_MIN_SPEED_FACTOR,
    SHIP_TURN_RATE,
    Fleet,
    GalconSim,
    Planet,
    RedirOrder,
    SendOrder,
    generate_galcon_map,
    physical_ship_count,
    planet_radius_from_production,
    settle_planets,
)

__all__ = [
    "FastGalconCppSim",
    "Fleet",
    "GalconSim",
    "HAS_FAST_SIM_CPP",
    "ITEMS_MAX",
    "ITEMS_RESERVED",
    "MIN_PLANET_GAP",
    "Planet",
    "RedirOrder",
    "SHIP_RADIUS",
    "SHIP_TURN_MIN_SPEED_FACTOR",
    "SHIP_TURN_RATE",
    "SendOrder",
    "generate_galcon_map",
    "physical_ship_count",
    "planet_radius_from_production",
    "settle_planets",
]
