"""Unified cross-bot Bradley-Terry Elo tournament & ladder on Evan's C++ Galcon 2 physics.

Allows rating heuristics (``random``, ``classic``, ``macro``, ``rusher``, ``tactical``),
unmodified Lua tournament bots (``Sparky v1.6``, ``Waffle3z-GA``, ``EsparanBot v0.2``),
our Pure / Search / Ensemble neural checkpoints, and any custom Python bot callable
(``py:module:factory``) on a single shared real-C++ Bradley-Terry leaderboard.

State is persisted atomically after every completed match pair so tournaments can
be resumed cleanly or extended with new bots via ``--resume``, and re-anchored to any
reference bot (e.g. ``bot:classic=1200`` or ``bot:random=0``) via ``--report-only``.
"""

from __future__ import annotations

import argparse
import atexit
import hashlib
import importlib
import json
import logging
import math
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import torch

from galcon.elo_rating import EloLadder, PlayerEntry
from galcon.fast_sim_cpp import HAS_FAST_SIM_CPP
from galcon.heuristic_bot import (
    GalconClassicBot,
    GalconMacroBot,
    GalconPassBot,
    GalconRandomBot,
    GalconRusherBot,
    GalconTacticalBot,
)
from galcon.lua_bridge import GalconLuaBotBridge
from galcon.sim import GalconSim
from galcon.transformer_bot import (
    EnsembleGalconSearchAgent,
    GalconSearchTransformerAgent,
    GalconTransformerAgent,
)

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent

STANDARD_ROSTER: List[Tuple[str, str, bool]] = [
    # (player_id, spec, is_optional_lua)
    ("bot:random", "heuristic:random", False),
    ("bot:macro", "heuristic:macro", False),
    ("bot:classic", "heuristic:classic", False),
    ("bot:esparan", "lua:external/GalconModVault/esparano/bots/bot_esparanbot_v0.2/bot_esparanbot.lua", True),
    ("bot:rusher", "heuristic:rusher", False),
    ("bot:tactical", "heuristic:tactical", False),
    ("gen3_pure", "pure:checkpoints/galcon_gen3_champion.pt", False),
    ("gen4_pure", "pure:checkpoints/galcon_gen4_ppo_s2.pt", False),
    (
        "bot:waffle",
        "lua:external/GalconModVault/esparano/bots/bot_hivemind/mod_botwar_testing.lua::bots_waffle3z_optimized",
        True,
    ),
    ("w14_search", "search:checkpoints/galcon_wave14_kd.pt", False),
    ("w18_pure", "pure:checkpoints/galcon_gen5_w18_pure.pt", False),
    ("bot:sparky", "lua:external/GalconModVault/esparano/bots/sparky/bot_sparky_v1.6.lua", True),
    ("w18_search", "search:checkpoints/galcon_gen5_w18.pt", False),
    ("w19_search", "search:checkpoints/galcon_gen5_w19.pt", False),
    ("w20_pure", "pure:checkpoints/galcon_gen5_w20_pure.pt", False),
    ("w20_search", "search:checkpoints/galcon_gen5_w20.pt", False),
    ("w21_pure", "pure:checkpoints/galcon_gen5_w21_pure.pt", False),
    ("w22_search", "search:checkpoints/galcon_gen5_w21.pt", False),
]

_BOT_CACHE: Dict[str, Any] = {}


def _cleanup_bots() -> None:
    for bot in _BOT_CACHE.values():
        close_fn = getattr(bot, "close", None)
        if callable(close_fn):
            try:
                close_fn()
            except Exception:
                pass
    _BOT_CACHE.clear()


atexit.register(_cleanup_bots)


def _init_worker_threads() -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    torch.set_num_threads(1)


def _resolve_repo_path(raw: str) -> Path:
    p = Path(raw)
    if p.is_file():
        return p.resolve()
    cand = REPO_ROOT / raw
    if cand.is_file():
        return cand.resolve()
    raise FileNotFoundError(f"Bot asset not found: {raw} (also checked {cand})")


def validate_spec(spec: str) -> None:
    """Validate that a bot spec refers to a supported kind and existing files."""
    if ":" not in spec:
        raise ValueError(f"Invalid bot spec {spec!r}; expected KIND:VALUE")
    kind, rest = spec.split(":", 1)
    if kind == "heuristic":
        if rest not in {"random", "pass", "classic", "macro", "rusher", "tactical"}:
            raise ValueError(f"Unknown heuristic bot {rest!r}")
    elif kind == "lua":
        lua_path = rest.split("::", 1)[0]
        _resolve_repo_path(lua_path)
    elif kind in {"search", "pure"}:
        _resolve_repo_path(rest)
    elif kind == "ens":
        parts = [s.strip() for s in rest.split("+") if s.strip()]
        if not parts:
            raise ValueError(f"Ensemble spec {spec!r} requires at least one checkpoint")
        for ck in parts:
            _resolve_repo_path(ck)
    elif kind == "py":
        if ":" not in rest:
            raise ValueError(f"Python spec {spec!r} must be py:<module.path>:<callable>")
    else:
        raise ValueError(f"Unsupported bot spec kind {kind!r} in {spec!r}")


def _infer_kind(spec: str) -> str:
    kind = spec.split(":", 1)[0]
    return "checkpoint" if kind in {"search", "pure", "ens"} else "bot"


def build_bot(spec: str):
    """Instantiate any bot (heuristic, Lua, Pure/Search/Ensemble neural, or py callable)."""
    if spec in _BOT_CACHE:
        return _BOT_CACHE[spec]
    _init_worker_threads()
    if ":" not in spec:
        raise ValueError(f"Invalid bot spec {spec!r}")
    kind, rest = spec.split(":", 1)

    if kind == "heuristic":
        mapping = {
            "random": GalconRandomBot,
            "pass": GalconPassBot,
            "classic": GalconClassicBot,
            "macro": GalconMacroBot,
            "rusher": GalconRusherBot,
            "tactical": GalconTacticalBot,
        }
        if rest not in mapping:
            raise ValueError(f"Unknown heuristic bot {rest!r}")
        bot = mapping[rest]()
    elif kind == "lua":
        if "::" in rest:
            lua_raw, fn_name = rest.split("::", 1)
            lua_path = _resolve_repo_path(lua_raw)
            bot = GalconLuaBotBridge(f"{lua_path}::{fn_name}")
        else:
            lua_path = _resolve_repo_path(rest)
            bot = GalconLuaBotBridge(str(lua_path))
    elif kind == "search":
        ckpt = _resolve_repo_path(rest)
        bot = GalconSearchTransformerAgent(ckpt, use_d4_canonical=True, use_d4_tta=False)
    elif kind == "pure":
        ckpt = _resolve_repo_path(rest)
        bot = GalconTransformerAgent(ckpt, use_d4_canonical=True, use_d4_tta=False)
    elif kind == "ens":
        ckpts = [_resolve_repo_path(p.strip()) for p in rest.split("+") if p.strip()]
        bot = EnsembleGalconSearchAgent(ckpts, use_d4_canonical=True, use_d4_tta=False)
    elif kind == "py":
        mod_path, fn_name = rest.split(":", 1)
        mod = importlib.import_module(mod_path)
        factory = getattr(mod, fn_name)
        bot = factory()
    else:
        raise ValueError(f"Unsupported bot spec kind {kind!r} in {spec!r}")

    _BOT_CACHE[spec] = bot
    return bot


def _arena_params(arena: str, pair_seed_idx: int) -> Dict[str, Any]:
    if arena == "standard18":
        return {
            "num_planets": 18,
            "width": 400.0,
            "height": 240.0,
            "symmetric": True,
            "act_every": 1,
            "max_ticks": 200,
        }
    if arena == "large24":
        return {
            "num_planets": 24,
            "width": 560.0,
            "height": 340.0,
            "symmetric": False,
            "act_every": 1,
            "max_ticks": 240,
        }
    if arena == "mixed":
        if pair_seed_idx % 4 == 3:
            return _arena_params("large24", pair_seed_idx)
        return _arena_params("standard18", pair_seed_idx)
    if arena == "compact6":
        return {
            "num_planets": 6,
            "width": 200.0,
            "height": 200.0,
            "symmetric": False,
            "act_every": 1,
            "max_ticks": 100,
        }
    raise ValueError(f"Unknown arena preset {arena!r}")


def _pair_key(a: str, b: str) -> Tuple[str, str]:
    return (a, b) if a < b else (b, a)


def _derive_map_seeds(base_seed: int, player_a: str, player_b: str, count: int) -> List[int]:
    a, b = _pair_key(player_a, player_b)
    digest = hashlib.sha256(f"{base_seed}:{a}|{b}".encode("utf-8")).digest()
    seed0 = int.from_bytes(digest[:4], "little") % 1_000_000_000
    return [seed0 + i * 7919 for i in range(count)]


def _run_single_game(task: Dict[str, Any]) -> Dict[str, Any]:
    assert HAS_FAST_SIM_CPP, "Real Galcon 2 C++ engine (galcon/sim_core.cpp) is required"
    _init_worker_threads()

    bot_a = build_bot(task["spec_a"])
    bot_b = build_bot(task["spec_b"])
    seat_a = int(task["seat_a"])
    seat_b = 2 if seat_a == 1 else 1
    act_every = int(task["act_every"])

    sim = GalconSim.create(
        seed=int(task["map_seed"]),
        num_planets=int(task["num_planets"]),
        width=float(task["width"]),
        height=float(task["height"]),
        max_ticks=int(task["max_ticks"]),
        symmetric=bool(task["symmetric"]),
        use_cpp=True,
    )

    lat_a: List[float] = []
    lat_b: List[float] = []

    while not sim.done:
        if sim.tick % act_every == 0:
            t0 = time.perf_counter()
            orders_a = bot_a.act(sim, seat_a)
            lat_a.append((time.perf_counter() - t0) * 1000.0)

            t1 = time.perf_counter()
            orders_b = bot_b.act(sim, seat_b)
            lat_b.append((time.perf_counter() - t1) * 1000.0)

            sim.step({seat_a: orders_a, seat_b: orders_b})
        else:
            sim.step({})

    winner = sim.winner()
    if winner == seat_a:
        res_a = "win"
    elif winner == seat_b:
        res_a = "loss"
    else:
        res_a = "draw"

    return {
        "res_a": res_a,
        "end_tick": int(sim.tick),
        "lat_a": lat_a,
        "lat_b": lat_b,
    }


def _played_pair_counts(ladder: EloLadder) -> Dict[Tuple[str, str], int]:
    counts: Dict[Tuple[str, str], int] = {}
    for m in ladder.matches:
        if m.total_games <= 0 or m.player_a == m.player_b:
            continue
        key = _pair_key(m.player_a, m.player_b)
        counts[key] = counts.get(key, 0) + 1
    return counts


def plan_round_robin_pairs(
    player_ids: Sequence[str],
    played_counts: Dict[Tuple[str, str], int],
) -> List[Tuple[str, str]]:
    """Return all unordered pairs not yet played."""
    pairs: List[Tuple[str, str]] = []
    for i in range(len(player_ids)):
        for j in range(i + 1, len(player_ids)):
            a, b = player_ids[i], player_ids[j]
            if played_counts.get(_pair_key(a, b), 0) == 0:
                pairs.append((a, b))
    return pairs


def plan_ladder_initial_pairs(
    ordered_player_ids: Sequence[str],
    anchor_id: str,
    played_counts: Dict[Tuple[str, str], int],
) -> List[Tuple[str, str]]:
    """Build a connected initial chain connecting adjacent players."""
    del anchor_id
    ids = list(ordered_player_ids)
    pairs: List[Tuple[str, str]] = []
    seen: Set[Tuple[str, str]] = set()
    for i in range(len(ids) - 1):
        a, b = ids[i], ids[i + 1]
        key = _pair_key(a, b)
        if key not in seen and played_counts.get(key, 0) == 0:
            pairs.append((a, b))
            seen.add(key)
    return pairs


def plan_swiss_round_pairs(
    ladder: EloLadder,
    player_ids: Sequence[str],
    played_counts: Dict[Tuple[str, str], int],
    require_member_of: Optional[Set[str]] = None,
) -> List[Tuple[str, str]]:
    """Pair players with closest current Elo who have not yet played each other."""
    sorted_ids = sorted(player_ids, key=lambda pid: (-ladder.get_elo(pid), pid))
    if require_member_of is not None:
        ordered = [pid for pid in sorted_ids if pid in require_member_of] + [
            pid for pid in sorted_ids if pid not in require_member_of
        ]
    else:
        ordered = list(sorted_ids)

    unpaired = list(ordered)
    pairs: List[Tuple[str, str]] = []

    while len(unpaired) >= 2:
        p = unpaired.pop(0)
        if require_member_of is not None and p not in require_member_of:
            break
        p_elo = ladder.get_elo(p)
        best_idx: Optional[int] = None
        best_diff: Optional[float] = None
        for idx, q in enumerate(unpaired):
            if played_counts.get(_pair_key(p, q), 0) > 0:
                continue
            diff = abs(p_elo - ladder.get_elo(q))
            if best_diff is None or diff < best_diff:
                best_diff = diff
                best_idx = idx
        if best_idx is not None:
            q = unpaired.pop(best_idx)
            pairs.append((p, q))
    return pairs


def _atomic_json_save(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp_path, path)


def _update_player_latency(
    lat_state: Dict[str, Dict[str, Any]],
    entry: PlayerEntry,
    new_samples: Sequence[float],
) -> None:
    if not new_samples:
        return
    rec = lat_state.setdefault(entry.player_id, {"sum": 0.0, "count": 0, "reservoir": []})
    arr = np.asarray(new_samples, dtype=np.float64)
    rec["sum"] = float(rec["sum"]) + float(arr.sum())
    rec["count"] = int(rec["count"]) + int(arr.size)

    # Strided downsample into bounded reservoir (max 2048 floats) so p95 stays accurate & JSON stays compact
    res: List[float] = list(rec.get("reservoir", []))
    stride = max(1, len(new_samples) // 64)
    res.extend(round(float(x), 3) for x in new_samples[::stride])
    if len(res) > 2048:
        keep_stride = max(2, len(res) // 1024)
        res = res[-2048::keep_stride] if len(res[::keep_stride]) < 512 else res[::keep_stride]
        res = res[-2048:]
    rec["reservoir"] = res

    entry.mean_ms = round(float(rec["sum"]) / max(1, int(rec["count"])), 2)
    entry.p95_ms = round(float(np.percentile(np.asarray(res, dtype=np.float64), 95)), 2)


def _resolve_roster(
    preset: str,
    explicit_bots: Sequence[str],
) -> List[Tuple[str, str]]:
    roster: List[Tuple[str, str]] = []
    seen: Set[str] = set()

    if preset == "standard":
        for pid, spec, optional_lua in STANDARD_ROSTER:
            if optional_lua:
                try:
                    validate_spec(spec)
                except FileNotFoundError as exc:
                    log.warning("Skipping optional Lua preset bot %s: %s", pid, exc)
                    continue
            else:
                validate_spec(spec)
            roster.append((pid, spec))
            seen.add(pid)
    elif preset != "none":
        raise ValueError(f"Unknown preset {preset!r}")

    for raw in explicit_bots:
        if "=" not in raw:
            raise ValueError(f"--bots entries must be NAME=SPEC, got {raw!r}")
        pid, spec = raw.split("=", 1)
        pid = pid.strip()
        spec = spec.strip()
        validate_spec(spec)
        if pid in seen:
            roster = [(p, spec if p == pid else s) for p, s in roster]
        else:
            roster.append((pid, spec))
            seen.add(pid)

    return roster


def _run_pair(
    *,
    ex: Optional[ProcessPoolExecutor],
    player_a: PlayerEntry,
    player_b: PlayerEntry,
    arena: str,
    games_per_pair: int,
    base_seed: int,
) -> Dict[str, Any]:
    assert player_a.spec and player_b.spec
    num_seeds = games_per_pair // 2
    seeds = _derive_map_seeds(base_seed, player_a.player_id, player_b.player_id, num_seeds)

    tasks: List[Dict[str, Any]] = []
    for idx, map_seed in enumerate(seeds):
        ap = _arena_params(arena, idx)
        for seat_a in (1, 2):
            tasks.append(
                {
                    "spec_a": player_a.spec,
                    "spec_b": player_b.spec,
                    "map_seed": map_seed,
                    "seat_a": seat_a,
                    **ap,
                }
            )

    if ex is None:
        outcomes = [_run_single_game(t) for t in tasks]
    else:
        outcomes = [f.result() for f in as_completed([ex.submit(_run_single_game, t) for t in tasks])]

    wins_a = sum(1 for o in outcomes if o["res_a"] == "win")
    draws = sum(1 for o in outcomes if o["res_a"] == "draw")
    losses_a = sum(1 for o in outcomes if o["res_a"] == "loss")
    lat_a = [x for o in outcomes for x in o["lat_a"]]
    lat_b = [x for o in outcomes for x in o["lat_b"]]
    avg_ticks = round(sum(o["end_tick"] for o in outcomes) / max(1, len(outcomes)), 1)

    return {
        "wins_a": wins_a,
        "draws": draws,
        "losses_a": losses_a,
        "lat_a": lat_a,
        "lat_b": lat_b,
        "avg_ticks": avg_ticks,
        "seeds": seeds,
    }


def _save_state(
    state_path: Path,
    ladder: EloLadder,
    ordered_ids: Sequence[str],
    meta: Dict[str, Any],
    lat_state: Dict[str, Dict[str, Any]],
) -> None:
    payload = {
        "ladder": ladder.to_dict(),
        "ordered_player_ids": list(ordered_ids),
        "meta": meta,
        "latency_stats": lat_state,
        "estimates": {
            pid: est.to_dict() for pid, est in ladder.estimates.items()
        },
    }
    _atomic_json_save(state_path, payload)


def _execute_pairs(
    pairs: Sequence[Tuple[str, str]],
    *,
    label: str,
    ex: Optional[ProcessPoolExecutor],
    ladder: EloLadder,
    ordered_ids: Sequence[str],
    meta: Dict[str, Any],
    lat_state: Dict[str, Dict[str, Any]],
    state_path: Path,
    arena: str,
    games_per_pair: int,
    base_seed: int,
    secondary_anchor: Optional[str],
    secondary_anchor_elo: float,
) -> None:
    if not pairs:
        return
    print(f"\n[{label}] Scheduling {len(pairs)} paired matches ({games_per_pair}g each)...", flush=True)
    for idx, (pa_id, pb_id) in enumerate(pairs, start=1):
        t0 = time.perf_counter()
        pa = ladder.players[pa_id]
        pb = ladder.players[pb_id]
        res = _run_pair(
            ex=ex,
            player_a=pa,
            player_b=pb,
            arena=arena,
            games_per_pair=games_per_pair,
            base_seed=base_seed,
        )
        ladder.record_match(
            player_a=pa_id,
            player_b=pb_id,
            wins_a=res["wins_a"],
            draws=res["draws"],
            losses_a=res["losses_a"],
            metadata={
                "round_label": label,
                "arena": arena,
                "avg_ticks": res["avg_ticks"],
                "seeds": res["seeds"],
            },
        )
        _update_player_latency(lat_state, pa, res["lat_a"])
        _update_player_latency(lat_state, pb, res["lat_b"])
        ladder.fit_ratings()
        _save_state(state_path, ladder, ordered_ids, meta, lat_state)
        dt = time.perf_counter() - t0
        ea = ladder.get_elo(pa_id)
        eb = ladder.get_elo(pb_id)
        print(
            f"  ({idx}/{len(pairs)}) {pa_id} vs {pb_id}: "
            f"{res['wins_a']}-{res['draws']}-{res['losses_a']} "
            f"(avg_t={res['avg_ticks']:.1f}, {dt:.1f}s) -> Elos {ea:.1f} / {eb:.1f}",
            flush=True,
        )
    print("\n" + ladder.format_leaderboard(secondary_anchor=secondary_anchor, secondary_anchor_elo=secondary_anchor_elo), flush=True)


def _parse_combine_spec(spec: str) -> Tuple[Optional[str], Path]:
    raw = str(spec).strip()
    p_direct = Path(raw)
    if p_direct.is_file() or "=" not in raw:
        return None, p_direct
    label, path_str = raw.split("=", 1)
    return label.strip(), Path(path_str.strip())


def combine_ladders(
    specs: Sequence[str],
    *,
    anchor_id: str,
    anchor_elo: float,
    secondary_anchor: Optional[str] = None,
    secondary_anchor_elo: float = 0.0,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Load multiple per-size tournament JSONs and aggregate an equal-weight Overall leaderboard."""
    labels: List[str] = []
    ladders: Dict[str, EloLadder] = {}
    sec_estimates: Dict[str, Optional[Dict[str, Any]]] = {}
    player_order: List[str] = []
    seen_players: Set[str] = set()

    for spec in specs:
        explicit_label, path = _parse_combine_spec(spec)
        if not path.is_file():
            raise FileNotFoundError(f"Tournament state file not found for --combine: {path}")
        raw = json.loads(path.read_text(encoding="utf-8"))
        ladder = EloLadder.from_dict(raw["ladder"])
        if anchor_id in ladder.players:
            ladder.reanchor(anchor_id, anchor_elo)

        label = explicit_label or str(raw.get("meta", {}).get("arena") or path.stem)
        if label in ladders:
            raise ValueError(f"Duplicate --combine label {label!r}; pass explicit LABEL=PATH")
        labels.append(label)
        ladders[label] = ladder
        sec_estimates[label] = (
            ladder._secondary_estimates(secondary_anchor, secondary_anchor_elo)
            if secondary_anchor and secondary_anchor in ladder.players
            else None
        )
        for pid in raw.get("ordered_player_ids", list(ladder.players.keys())):
            if pid not in seen_players:
                seen_players.add(pid)
                player_order.append(pid)
        for pid in ladder.players:
            if pid not in seen_players:
                seen_players.add(pid)
                player_order.append(pid)

    rows: List[Dict[str, Any]] = []
    for pid in player_order:
        per_elo: Dict[str, Optional[float]] = {}
        per_std: Dict[str, Optional[float]] = {}
        per_games: Dict[str, int] = {}
        played_elos: List[float] = []
        played_stds: List[float] = []
        played_alt_elos: List[float] = []
        tot_g = tot_w = tot_d = tot_l = 0

        for label in labels:
            lad = ladders[label]
            est = lad.estimates.get(pid)
            if est is not None and est.games > 0:
                per_elo[label] = float(est.elo)
                per_std[label] = float(est.elo_std)
                per_games[label] = int(est.games)
                played_elos.append(float(est.elo))
                played_stds.append(float(est.elo_std))
                sec_map = sec_estimates[label]
                if sec_map is not None and pid in sec_map:
                    played_alt_elos.append(float(sec_map[pid].elo))
                tot_g += int(est.games)
                tot_w += int(est.wins)
                tot_d += int(est.draws)
                tot_l += int(est.losses)
            else:
                per_elo[label] = None
                per_std[label] = None
                per_games[label] = 0

        sizes_played = len(played_elos)
        if sizes_played == 0:
            continue
        overall_elo = sum(played_elos) / sizes_played
        if all(math.isfinite(s) for s in played_stds):
            overall_std = math.sqrt(sum(s * s for s in played_stds)) / sizes_played
        else:
            overall_std = float("inf")
        overall_alt = (
            sum(played_alt_elos) / len(played_alt_elos)
            if len(played_alt_elos) == sizes_played and sizes_played > 0
            else None
        )
        score_rate = (tot_w + 0.5 * tot_d) / tot_g if tot_g > 0 else 0.0

        rows.append(
            {
                "player_id": pid,
                "per_size_elo": per_elo,
                "per_size_std": per_std,
                "per_size_games": per_games,
                "sizes_played": sizes_played,
                "overall_elo": overall_elo,
                "overall_std": overall_std,
                "overall_alt_elo": overall_alt,
                "total_games": tot_g,
                "total_wins": tot_w,
                "total_draws": tot_d,
                "total_losses": tot_l,
                "score_rate": score_rate,
            }
        )

    rows.sort(key=lambda r: (r["sizes_played"] < len(labels), -r["overall_elo"], r["player_id"]))
    return labels, rows


def format_combined_ascii(
    rows: Sequence[Dict[str, Any]],
    labels: Sequence[str],
    anchor_id: str,
    anchor_elo: float,
    secondary_anchor: Optional[str] = None,
    secondary_anchor_elo: float = 0.0,
) -> str:
    del anchor_id, anchor_elo, secondary_anchor_elo
    has_sec = secondary_anchor is not None and any(r["overall_alt_elo"] is not None for r in rows)
    widths = {lbl: max(8, len(lbl)) for lbl in labels}
    size_hdr = "".join(f"  {lbl:>{widths[lbl]}}" for lbl in labels)
    sec_hdr = f"  {'AltElo':>8}" if has_sec else ""
    hdr = f"{'Rank':>4}  {'Bot':<16}  {'Overall':>8}  {'±Std':>6}{size_hdr}{sec_hdr}  {'Sizes':>5}  {'Games':>5}  {'Score%':>6}"
    lines = [hdr, "-" * len(hdr)]
    for rank, r in enumerate(rows, start=1):
        std_s = f"{r['overall_std']:6.1f}" if math.isfinite(r["overall_std"]) else "   inf"
        size_vals = "".join(
            f"  {r['per_size_elo'][lbl]:{widths[lbl]}.1f}"
            if r["per_size_elo"][lbl] is not None
            else f"  {'-':>{widths[lbl]}}"
            for lbl in labels
        )
        sec_val = (
            f"  {r['overall_alt_elo']:8.1f}"
            if (has_sec and r["overall_alt_elo"] is not None)
            else (f"  {'-':>8}" if has_sec else "")
        )
        lines.append(
            f"{rank:>4}  {r['player_id']:<16}  {r['overall_elo']:>8.1f}  {std_s}{size_vals}{sec_val}  "
            f"{r['sizes_played']:>2}/{len(labels):<2}  {r['total_games']:>5}  {100.0 * r['score_rate']:>5.1f}%"
        )
    return "\n".join(lines)


def format_combined_markdown(
    rows: Sequence[Dict[str, Any]],
    labels: Sequence[str],
    anchor_id: str,
    anchor_elo: float,
    secondary_anchor: Optional[str] = None,
    secondary_anchor_elo: float = 0.0,
) -> str:
    has_sec = secondary_anchor is not None and any(r["overall_alt_elo"] is not None for r in rows)
    size_cols = "".join(f" {lbl} |" for lbl in labels)
    size_seps = "---:|" * len(labels)
    sec_col = f" Overall vs `{secondary_anchor}` ({secondary_anchor_elo:.0f}) |" if has_sec else ""
    sec_sep = "---:|" if has_sec else ""
    lines = [
        f"| Rank | Bot | Overall Elo (`{anchor_id}={anchor_elo:.0f}`) | ±Std |{size_cols}{sec_col} Sizes | Total Games | Score% |",
        f"|---:|---|---:|---:|{size_seps}{sec_sep}:---:|---:|---:|",
    ]
    for rank, r in enumerate(rows, start=1):
        std_s = f"±{r['overall_std']:.1f}" if math.isfinite(r["overall_std"]) else "±inf"
        size_vals = "".join(
            f" `{r['per_size_elo'][lbl]:.1f}` |" if r["per_size_elo"][lbl] is not None else " — |"
            for lbl in labels
        )
        sec_val = (
            f" `{r['overall_alt_elo']:.1f}` |"
            if (has_sec and r["overall_alt_elo"] is not None)
            else (" — |" if has_sec else "")
        )
        lines.append(
            f"| {rank} | `{r['player_id']}` | **{r['overall_elo']:.1f}** | {std_s} |{size_vals}{sec_val} "
            f"{r['sizes_played']}/{len(labels)} | {r['total_games']} | {100.0 * r['score_rate']:.1f}% |"
        )
    return "\n".join(lines)


def run_tournament(args: argparse.Namespace) -> Optional[EloLadder]:
    if getattr(args, "combine", None):
        labels, rows = combine_ladders(
            args.combine,
            anchor_id=args.anchor,
            anchor_elo=args.anchor_elo,
            secondary_anchor=args.secondary_anchor,
            secondary_anchor_elo=args.secondary_anchor_elo,
        )
        print(
            format_combined_ascii(
                rows,
                labels,
                anchor_id=args.anchor,
                anchor_elo=args.anchor_elo,
                secondary_anchor=args.secondary_anchor,
                secondary_anchor_elo=args.secondary_anchor_elo,
            )
        )
        if args.markdown_out:
            md = format_combined_markdown(
                rows,
                labels,
                anchor_id=args.anchor,
                anchor_elo=args.anchor_elo,
                secondary_anchor=args.secondary_anchor,
                secondary_anchor_elo=args.secondary_anchor_elo,
            )
            Path(args.markdown_out).write_text(md + "\n", encoding="utf-8")
        return None

    state_path = Path(args.state)
    games_per_pair = max(2, int(args.games_per_pair))
    if games_per_pair % 2 != 0:
        raise ValueError(f"--games-per-pair must be even for paired seat swaps, got {games_per_pair}")

    if args.report_only:
        if not state_path.is_file():
            raise FileNotFoundError(f"State file not found for --report-only: {state_path}")
        raw = json.loads(state_path.read_text(encoding="utf-8"))
        ladder = EloLadder.from_dict(raw["ladder"])
        ladder.reanchor(args.anchor, args.anchor_elo)
        print(
            ladder.format_leaderboard(
                secondary_anchor=args.secondary_anchor,
                secondary_anchor_elo=args.secondary_anchor_elo,
            )
        )
        if args.markdown_out:
            md = ladder.format_markdown_leaderboard(
                secondary_anchor=args.secondary_anchor,
                secondary_anchor_elo=args.secondary_anchor_elo,
            )
            Path(args.markdown_out).write_text(md + "\n", encoding="utf-8")
        return ladder

    assert HAS_FAST_SIM_CPP, "Real Galcon 2 C++ simulator (galcon/sim_core.cpp) must be compiled"

    cli_roster = _resolve_roster(args.preset, args.bots or [])

    if args.resume and state_path.is_file():
        raw = json.loads(state_path.read_text(encoding="utf-8"))
        ladder = EloLadder.from_dict(raw["ladder"])
        if ladder.anchor_id != args.anchor or abs(ladder.anchor_elo - args.anchor_elo) > 1e-9:
            ladder.reanchor(args.anchor, args.anchor_elo)
        ordered_ids: List[str] = list(raw.get("ordered_player_ids", list(ladder.players.keys())))
        meta: Dict[str, Any] = dict(raw.get("meta", {}))
        lat_state: Dict[str, Dict[str, Any]] = dict(raw.get("latency_stats", {}))
        newly_added: Set[str] = set()

        for pid, spec in cli_roster:
            if pid not in ladder.players:
                ladder.register_player(
                    PlayerEntry(
                        player_id=pid,
                        kind=_infer_kind(spec),
                        spec=spec,
                        ckpt_path=spec.split(":", 1)[1] if _infer_kind(spec) == "checkpoint" else None,
                        bot_type=pid.split(":", 1)[-1] if _infer_kind(spec) == "bot" else None,
                        active=True,
                    )
                )
                ordered_ids.append(pid)
                newly_added.add(pid)
            else:
                ladder.players[pid].spec = spec

        for pid, spec in cli_roster:
            validate_spec(spec)
        for pid, entry in ladder.players.items():
            if entry.spec:
                validate_spec(entry.spec)

        meta.setdefault("chain_completed", False)
        meta.setdefault("completed_swiss_rounds", 0)
        if newly_added:
            meta["chain_completed"] = False
            catchup = set(meta.get("catchup_player_ids", []))
            catchup.update(newly_added)
            meta["catchup_player_ids"] = sorted(catchup)
            target_swiss = max(0, int(args.rounds) - 1)
            meta["catchup_rounds_remaining"] = max(int(meta.get("catchup_rounds_remaining", 0)), target_swiss)
    else:
        if len(cli_roster) < 2:
            raise ValueError("Need at least 2 bots (via --preset standard and/or --bots NAME=SPEC)")
        anchor_id = args.anchor if any(pid == args.anchor for pid, _ in cli_roster) else cli_roster[0][0]
        ladder = EloLadder(anchor_id=anchor_id, anchor_elo=args.anchor_elo)
        ladder.players.clear()
        ordered_ids = []
        for pid, spec in cli_roster:
            ladder.register_player(
                PlayerEntry(
                    player_id=pid,
                    kind=_infer_kind(spec),
                    spec=spec,
                    ckpt_path=spec.split(":", 1)[1] if _infer_kind(spec) == "checkpoint" else None,
                    bot_type=pid.split(":", 1)[-1] if _infer_kind(spec) == "bot" else None,
                    active=True,
                )
            )
            ordered_ids.append(pid)
        ladder.fit_ratings()
        meta = {
            "mode": args.mode,
            "arena": args.arena,
            "games_per_pair": games_per_pair,
            "rounds": int(args.rounds),
            "seed": int(args.seed),
            "chain_completed": False,
            "completed_swiss_rounds": 0,
            "catchup_player_ids": [],
            "catchup_rounds_remaining": 0,
        }
        lat_state = {}
        newly_added = set()

    workers = max(1, int(args.workers))
    ctx = mp.get_context("spawn")
    ex: Optional[ProcessPoolExecutor] = (
        ProcessPoolExecutor(max_workers=workers, mp_context=ctx) if workers > 1 else None
    )

    try:
        played_counts = _played_pair_counts(ladder)

        if args.mode == "round_robin":
            rr_pairs = plan_round_robin_pairs(ordered_ids, played_counts)
            _execute_pairs(
                rr_pairs,
                label="Round-Robin",
                ex=ex,
                ladder=ladder,
                ordered_ids=ordered_ids,
                meta=meta,
                lat_state=lat_state,
                state_path=state_path,
                arena=args.arena,
                games_per_pair=games_per_pair,
                base_seed=int(args.seed),
                secondary_anchor=args.secondary_anchor,
                secondary_anchor_elo=args.secondary_anchor_elo,
            )
        else:
            # 1. Chain pass (or reconnect newly resumed bots into the graph)
            if not meta.get("chain_completed", False):
                if newly_added:
                    # Connect each new player to the median-Elo existing neighbor + last added neighbor
                    existing_sorted = sorted(
                        [pid for pid in ordered_ids if pid not in newly_added],
                        key=lambda pid: ladder.get_elo(pid),
                    )
                    chain_pairs: List[Tuple[str, str]] = []
                    seen_p: Set[Tuple[str, str]] = set()
                    for npid in sorted(newly_added):
                        if existing_sorted:
                            mid = existing_sorted[len(existing_sorted) // 2]
                            k1 = _pair_key(npid, mid)
                            if k1 not in seen_p and played_counts.get(k1, 0) == 0:
                                chain_pairs.append((npid, mid))
                                seen_p.add(k1)
                            top = existing_sorted[-1]
                            k2 = _pair_key(npid, top)
                            if k2 not in seen_p and played_counts.get(k2, 0) == 0:
                                chain_pairs.append((npid, top))
                                seen_p.add(k2)
                else:
                    chain_pairs = plan_ladder_initial_pairs(ordered_ids, ladder.anchor_id, played_counts)

                _execute_pairs(
                    chain_pairs,
                    label="Round 1 / Chain",
                    ex=ex,
                    ladder=ladder,
                    ordered_ids=ordered_ids,
                    meta=meta,
                    lat_state=lat_state,
                    state_path=state_path,
                    arena=args.arena,
                    games_per_pair=games_per_pair,
                    base_seed=int(args.seed),
                    secondary_anchor=args.secondary_anchor,
                    secondary_anchor_elo=args.secondary_anchor_elo,
                )
                meta["chain_completed"] = True
                _save_state(state_path, ladder, ordered_ids, meta, lat_state)

            # 2. Catch-up Swiss rounds for any newly resumed players
            catchup_ids = set(meta.get("catchup_player_ids", []))
            while catchup_ids and int(meta.get("catchup_rounds_remaining", 0)) > 0:
                played_counts = _played_pair_counts(ladder)
                c_pairs = plan_swiss_round_pairs(
                    ladder, ordered_ids, played_counts, require_member_of=catchup_ids
                )
                if not c_pairs:
                    meta["catchup_rounds_remaining"] = 0
                    break
                r_num = int(args.rounds) - int(meta["catchup_rounds_remaining"]) + 1
                _execute_pairs(
                    c_pairs,
                    label=f"Catch-up Swiss {r_num}",
                    ex=ex,
                    ladder=ladder,
                    ordered_ids=ordered_ids,
                    meta=meta,
                    lat_state=lat_state,
                    state_path=state_path,
                    arena=args.arena,
                    games_per_pair=games_per_pair,
                    base_seed=int(args.seed),
                    secondary_anchor=args.secondary_anchor,
                    secondary_anchor_elo=args.secondary_anchor_elo,
                )
                meta["catchup_rounds_remaining"] = int(meta["catchup_rounds_remaining"]) - 1
                if meta["catchup_rounds_remaining"] <= 0:
                    meta["catchup_player_ids"] = []
                _save_state(state_path, ladder, ordered_ids, meta, lat_state)

            # 3. Remaining full Swiss rounds
            target_swiss = max(0, int(args.rounds) - 1)
            while int(meta.get("completed_swiss_rounds", 0)) < target_swiss:
                played_counts = _played_pair_counts(ladder)
                s_pairs = plan_swiss_round_pairs(ladder, ordered_ids, played_counts)
                r_idx = int(meta.get("completed_swiss_rounds", 0)) + 2
                if not s_pairs:
                    meta["completed_swiss_rounds"] = target_swiss
                    break
                _execute_pairs(
                    s_pairs,
                    label=f"Round {r_idx} / Swiss",
                    ex=ex,
                    ladder=ladder,
                    ordered_ids=ordered_ids,
                    meta=meta,
                    lat_state=lat_state,
                    state_path=state_path,
                    arena=args.arena,
                    games_per_pair=games_per_pair,
                    base_seed=int(args.seed),
                    secondary_anchor=args.secondary_anchor,
                    secondary_anchor_elo=args.secondary_anchor_elo,
                )
                meta["completed_swiss_rounds"] = int(meta.get("completed_swiss_rounds", 0)) + 1
                _save_state(state_path, ladder, ordered_ids, meta, lat_state)
    finally:
        if ex is not None:
            ex.shutdown(wait=True)
        _cleanup_bots()

    _save_state(state_path, ladder, ordered_ids, meta, lat_state)
    if args.markdown_out:
        md = ladder.format_markdown_leaderboard(
            secondary_anchor=args.secondary_anchor,
            secondary_anchor_elo=args.secondary_anchor_elo,
        )
        Path(args.markdown_out).write_text(md + "\n", encoding="utf-8")

    return ladder


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Unified cross-bot Bradley-Terry Elo tournament on Evan's C++ Galcon 2 physics."
    )
    p.add_argument(
        "--preset",
        choices=("none", "standard"),
        default="none",
        help="Built-in roster preset ('standard' = 18 cross-generation + Lua + heuristic bots).",
    )
    p.add_argument(
        "--bots",
        nargs="+",
        default=[],
        help="Explicit NAME=SPEC bot entries (heuristic:*, lua:*, search:*, pure:*, ens:*, py:mod:fn).",
    )
    p.add_argument(
        "--state",
        default="docs/galcon_unified_ladder.json",
        help="Path to atomic tournament state JSON.",
    )
    p.add_argument(
        "--mode",
        choices=("ladder", "round_robin"),
        default="ladder",
        help="Scheduling mode ('ladder' = chain + Swiss, 'round_robin' = all pairs).",
    )
    p.add_argument(
        "--arena",
        choices=("compact6", "standard18", "large24", "mixed"),
        default="standard18",
        help="Real-C++ map arena preset.",
    )
    p.add_argument(
        "--games-per-pair",
        type=int,
        default=16,
        help="Even number of seat-swapped games per paired match.",
    )
    p.add_argument(
        "--rounds",
        type=int,
        default=7,
        help="Total ladder rounds (1 initial chain + rounds-1 Swiss rounds).",
    )
    p.add_argument("--seed", type=int, default=42000, help="Base deterministic map seed.")
    p.add_argument("--workers", type=int, default=14, help="Parallel worker processes.")
    p.add_argument("--anchor", default="bot:classic", help="Primary anchor bot ID.")
    p.add_argument("--anchor-elo", type=float, default=1200.0, help="Pinned Elo for --anchor.")
    p.add_argument(
        "--secondary-anchor",
        default="bot:random",
        help="Optional secondary anchor bot ID printed alongside primary Elo.",
    )
    p.add_argument(
        "--secondary-anchor-elo",
        type=float,
        default=0.0,
        help="Pinned Elo for --secondary-anchor.",
    )
    p.add_argument("--resume", action="store_true", help="Resume or extend an existing --state JSON.")
    p.add_argument("--report-only", action="store_true", help="Re-fit and print leaderboard from --state.")
    p.add_argument(
        "--combine",
        nargs="+",
        default=[],
        help="Combine multiple per-size state JSONs (LABEL=PATH or PATH) into an Overall leaderboard.",
    )
    p.add_argument("--markdown-out", default=None, help="Optional path to write Markdown leaderboard table.")
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = build_arg_parser().parse_args(argv)
    run_tournament(args)


if __name__ == "__main__":
    main()
