"""Competitive Galcon simulator, tactical solver, PlanetTimelineTransformer bot, and gbots adapter.

Cross-pollinates the variable-N entity + arrival-timeline Transformer from ``orbit_wars_bot``
directly to classic Galcon / Phil Hassey's ``gbots`` protocol.
"""

from galcon.sim import Fleet, GalconSim, Planet, RedirOrder, SendOrder, generate_galcon_map

__all__ = [
    "Fleet",
    "GalconSim",
    "Planet",
    "RedirOrder",
    "SendOrder",
    "generate_galcon_map",
]
