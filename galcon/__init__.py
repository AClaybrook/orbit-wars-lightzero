"""Competitive Galcon simulator, tactical solver, PlanetTimelineTransformer bot, and gbots adapter.

Cross-pollinates the variable-N entity + arrival-timeline Transformer from ``orbit_wars_bot``
directly to classic Galcon / Phil Hassey's ``gbots`` protocol.
"""

from galcon.sim import (
    MIN_PLANET_GAP,
    SHIP_RADIUS,
    Fleet,
    GalconSim,
    Planet,
    RedirOrder,
    SendOrder,
    generate_galcon_map,
    planet_radius_from_production,
    settle_planets,
)

__all__ = [
    "Fleet",
    "GalconSim",
    "MIN_PLANET_GAP",
    "Planet",
    "RedirOrder",
    "SHIP_RADIUS",
    "SendOrder",
    "generate_galcon_map",
    "planet_radius_from_production",
    "settle_planets",
]
