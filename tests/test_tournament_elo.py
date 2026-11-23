"""Unit tests for the local Elo rating estimators in scripts/tournament_elo.py."""

from __future__ import annotations

import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from tournament_elo import (  # noqa: E402
    build_4p_schedule,
    fit_bradley_terry_elo,
    fit_plackett_luce_elo,
)


def test_bradley_terry_elo_ordering_and_anchor():
    bots = ["strong", "medium", "starter"]
    # strong beats medium 75% (6-2) and starter 100% (8-0); medium beats starter 75% (6-2)
    outcomes = (
        [(0, 1, 1.0)] * 6
        + [(0, 1, 0.0)] * 2
        + [(0, 2, 1.0)] * 8
        + [(1, 2, 1.0)] * 6
        + [(1, 2, 0.0)] * 2
    )
    elos = fit_bradley_terry_elo(bots, outcomes, anchor_bot="starter", anchor_elo=1000.0)
    assert math.isclose(elos["starter"], 1000.0, abs_tol=1e-6)
    assert elos["strong"] > elos["medium"] > elos["starter"]
    # Order-independent: reversing outcome order produces identical Elo
    elos_rev = fit_bradley_terry_elo(bots, list(reversed(outcomes)), anchor_bot="starter", anchor_elo=1000.0)
    for b in bots:
        assert math.isclose(elos[b], elos_rev[b], abs_tol=1e-5)


def test_plackett_luce_matches_bradley_terry_for_2p():
    bots = ["a", "b", "c"]
    outcomes = [(0, 1, 1.0)] * 5 + [(0, 1, 0.0)] * 2 + [(1, 2, 1.0)] * 6 + [(0, 2, 1.0)] * 4
    rankings = [[i, j] if s > 0.5 else [j, i] for i, j, s in outcomes]
    bt = fit_bradley_terry_elo(bots, outcomes, anchor_bot="c", prior_strength=0.25)
    pl = fit_plackett_luce_elo(bots, rankings, anchor_bot="c", prior_strength=0.25)
    for b in bots:
        assert math.isclose(bt[b], pl[b], abs_tol=1.0)


def test_plackett_luce_4p_ordering():
    bots = ["p0", "p1", "p2", "starter"]
    # p0 usually 1st, p1 2nd, p2 3rd, starter 4th
    rankings = [[0, 1, 2, 3]] * 8 + [[1, 0, 2, 3]] * 2 + [[0, 2, 1, 3]] * 2
    elos = fit_plackett_luce_elo(bots, rankings, anchor_bot="starter", anchor_elo=1000.0)
    assert math.isclose(elos["starter"], 1000.0, abs_tol=1e-6)
    assert elos["p0"] > elos["p1"] > elos["p2"] > elos["starter"]


def test_build_4p_schedule_balances_games_and_seats():
    bots = ["b0", "b1", "b2", "b3", "b4", "b5", "b6"]
    sched = build_4p_schedule(bots, num_games=28, seed=42)
    assert len(sched) == 28
    counts = {b: 0 for b in bots}
    seats = {b: [0, 0, 0, 0] for b in bots}
    for table in sched:
        assert len(table) == 4
        assert len(set(table)) == 4
        for s, b in enumerate(table):
            counts[b] += 1
            seats[b][s] += 1
    # 28 * 4 / 7 = 16 games per bot exactly
    assert all(c == 16 for c in counts.values())
    # Each bot plays each seat ~4 times (within +/- 1 of mean)
    for b in bots:
        assert max(seats[b]) - min(seats[b]) <= 2
