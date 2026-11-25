"""Competitive Galcon simulator, tactical solver, PlanetTimelineTransformer bot, and gbots adapter.

Cross-pollinates the variable-N entity + arrival-timeline Transformer from ``orbit_wars_bot``
directly to classic Galcon / Phil Hassey's ``gbots`` protocol.
"""

from galcon.sim import (
    ITEMS_MAX,
    ITEMS_RESERVED,
    MIN_PLANET_GAP,
    SHIP_RADIUS,
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
    "Fleet",
    "GalconSim",
    "ITEMS_MAX",
    "ITEMS_RESERVED",
    "MIN_PLANET_GAP",
    "Planet",
    "RedirOrder",
    "SHIP_RADIUS",
    "SendOrder",
    "generate_galcon_map",
    "physical_ship_count",
    "planet_radius_from_production",
    "settle_planets",
]
