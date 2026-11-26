"""Competitive Galcon simulator, tactical solver, PlanetTimelineTransformer bot, and gbots adapter.

Cross-pollinates the variable-N entity + arrival-timeline Transformer from ``orbit_wars_bot``
directly to classic Galcon / Phil Hassey's ``gbots`` protocol.
"""

from galcon.fast_sim_cpp import HAS_FAST_SIM_CPP, FastGalconCppSim
from galcon.sim import (
    DEFAULT_NUM_PLANETS,
    DEFAULT_NUM_PLAYERS,
    ITEMS_MAX,
    ITEMS_RESERVED,
    MIN_PLANET_GAP,
    SHIP_RADIUS,
    SHIP_SEP_MAX_DISP,
    SHIP_TURN_MIN_SPEED_FACTOR,
    SHIP_TURN_RATE,
    SPAWN_AREA_RADIUS,
    SPAWN_SETTLE_ITERS,
    SPAWN_SETTLE_TOL,
    Fleet,
    GalconSim,
    Planet,
    RedirOrder,
    SendOrder,
    generate_galcon_map,
    physical_ship_count,
    planet_radius_from_production,
    settle_planets,
    symmetric_planet_count,
)

__all__ = [
    "DEFAULT_NUM_PLANETS",
    "DEFAULT_NUM_PLAYERS",
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
    "SHIP_SEP_MAX_DISP",
    "SHIP_TURN_MIN_SPEED_FACTOR",
    "SHIP_TURN_RATE",
    "SPAWN_AREA_RADIUS",
    "SPAWN_SETTLE_ITERS",
    "SPAWN_SETTLE_TOL",
    "SendOrder",
    "generate_galcon_map",
    "physical_ship_count",
    "planet_radius_from_production",
    "settle_planets",
    "symmetric_planet_count",
]
