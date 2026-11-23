"""Unit tests for the 3v1 coalition coordination module."""

from __future__ import annotations

import math

from orbit_wars_bot.coalition import CoalitionCoordinator, create_coalition_team


def _make_obs(step: int = 20, boss_ships: int = 20, ally1_ships: int = 40, ally2_ships: int = 35):
    return {
        "step": step,
        "player": 1,
        "planets": [
            # Boss planet north of the central sun (sun-clear from all 3 allies)
            [0, 0, 60.0, 12.0, 2.5, boss_ships, 4],
            # Ally 1 & Ally 2 aligned vertically at x=5.0 in static outer ring (r_orb + r > 50)
            [1, 1, 5.0, 10.0, 2.5, ally1_ships, 3],
            [2, 2, 5.0, 28.0, 2.5, ally2_ships, 3],
            # Ally 3 planet east of Boss
            [3, 3, 85.0, 12.0, 2.5, 30, 3],
            # Neutral planet south
            [4, -1, 50.0, 82.0, 2.0, 10, 2],
        ],
        "fleets": [],
        "comets": [],
    }


def test_truce_blocks_friendly_fire_between_allies():
    coord = CoalitionCoordinator(boss_id=0, ally_ids=(1, 2, 3), mode="truce", comm_latency=0)
    obs = _make_obs(step=0)
    # Ally 1 base agent tries to attack Ally 2's planet directly below it (angle = +pi/2)
    def hostile_base(o, c=None):
        return [[1, math.pi / 2.0, 20]]

    orders = coord.act(1, obs, None, hostile_base)
    assert orders == []
    assert coord.stats.friendly_fire_blocked == 1


def test_focus_fire_coordinates_strike_against_boss():
    coord = CoalitionCoordinator(
        boss_id=0, ally_ids=(1, 2, 3), mode="focus_fire", comm_latency=0, start_focus_step=5
    )
    obs0 = _make_obs(step=0)
    coord.act(1, obs0, None, lambda o, c=None: [])

    obs20 = _make_obs(step=20, boss_ships=15, ally1_ships=45, ally2_ships=40)
    # Step 20: Allies with idle base agents should coordinate volleys toward boss planet 0
    orders_all = []
    for pid in (1, 2, 3):
        o = dict(obs20, player=pid)
        orders_all.extend(coord.act(pid, o, None, lambda ob, cfg=None: []))

    assert len(orders_all) >= 1
    assert isinstance(orders_all[0], list) and len(orders_all[0]) == 3
    assert coord.stats.sync_volleys_launched >= 1
    msg = coord.get_delayed_message(20)
    assert msg is not None
    assert msg.primary_target_id == 0


def test_comm_latency_delays_coalition_message():
    coord = CoalitionCoordinator(
        boss_id=0, ally_ids=(1, 2, 3), mode="focus_fire", comm_latency=5, start_focus_step=0
    )
    for t in range(7):
        obs = _make_obs(step=t)
        coord.act(1, obs, None, lambda o, c=None: [])

    # Before tick 5, delayed message is None; at tick 6, ally sees tick 1's message
    assert coord.get_delayed_message(3) is None
    msg6 = coord.get_delayed_message(6)
    assert msg6 is not None
    assert msg6.step == 1


def test_coalition_dissolves_when_boss_eliminated():
    coord, team = create_coalition_team(
        lambda: (lambda o, c=None: [[1, math.pi / 2.0, 10]]),
        boss_id=0,
        ally_ids=(1, 2, 3),
        mode="truce",
        comm_latency=0,
    )
    obs0 = _make_obs(step=0)
    assert team[1](obs0, None) == []  # Boss alive -> friendly fire blocked

    # Boss eliminated (planet 0 now owned by ally 3, no boss fleets)
    obs_dead = _make_obs(step=50)
    obs_dead["planets"][0][1] = 3
    orders = team[1](obs_dead, None)
    assert len(orders) == 1  # Truce dissolved -> FFA resumed
    assert coord.stats.boss_eliminated_step == 50


def test_imperfect_compliance_zero_disables_focus_volleys():
    coord = CoalitionCoordinator(
        boss_id=0,
        ally_ids=(1, 2, 3),
        mode="focus_fire",
        comm_latency=0,
        start_focus_step=5,
        compliance=0.0,
        info_noise=0.4,
    )
    obs0 = _make_obs(step=0)
    coord.act(1, obs0, None, lambda o, c=None: [])
    obs20 = _make_obs(step=20, boss_ships=15, ally1_ships=45, ally2_ships=40)
    orders_all = []
    for pid in (1, 2, 3):
        orders_all.extend(coord.act(pid, dict(obs20, player=pid), None, lambda ob, cfg=None: []))
    assert orders_all == []
    assert coord.stats.sync_volleys_launched == 0
