"""Strong tactical Galcon bots (Classic greedy baseline + Timeline-aware Tactical Solver).

``GalconTacticalBot`` acts both as:
1. A strong standalone rule-based competitor ready for ``gbots``, and
2. A high-skill analytical teacher for bootstrapping & DAgger-training ``GalconPlanetTransformer``.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from galcon.sim import (
    PRODUCTION_DIVISOR,
    GalconSim,
    Order,
    Planet,
    RedirOrder,
    SendOrder,
    project_planet_future,
)


def _ships_to_percent(avail: float, desired: float) -> int:
    if avail <= 1.0 or desired <= 0.0:
        return 0
    if desired >= avail - 0.5:
        return 100
    pct = int(math.ceil((desired / avail) * 100.0))
    return max(5, min(100, pct))


class GalconClassicBot:
    """Fast baseline similar to Phil Hassey's built-in ``classic`` bot."""

    def __init__(self, send_threshold: float = 15.0) -> None:
        self.send_threshold = float(send_threshold)

    def act(self, sim: GalconSim, player: int) -> List[Order]:
        orders: List[Order] = []
        mine = [p for p in sim.planets.values() if p.owner == player]
        others = [p for p in sim.planets.values() if p.owner != player]
        if not mine or not others:
            return orders
        for src in mine:
            if src.ships < self.send_threshold:
                continue
            best_t: Optional[Planet] = None
            best_score = -1e9
            for dst in others:
                dist = math.hypot(dst.x - src.x, dst.y - src.y)
                cost = dst.ships + 1.0
                score = (dst.production + 25.0) / (cost + 0.18 * dist + 10.0)
                if dst.owner != 0:
                    score *= 1.15
                if score > best_score:
                    best_score = score
                    best_t = dst
            if best_t is not None and src.ships > best_t.ships * 0.65:
                pct = _ships_to_percent(src.ships, best_t.ships + 4.0)
                if pct >= 10:
                    orders.append(SendOrder(src.id, best_t.id, min(85, pct)))
        return orders


class GalconTacticalBot:
    """High-tier timeline-aware Galcon solver with exact defense, snipe steals, ROI expansion, and fleet redirects."""

    def __init__(self, *, aggression: float = 1.0, enable_redirect: bool = True) -> None:
        self.aggression = float(aggression)
        self.enable_redirect = bool(enable_redirect)

    def act(self, sim: GalconSim, player: int) -> List[Order]:
        orders: List[Order] = []
        planets = list(sim.planets.values())
        mine = [p for p in planets if p.owner == player]
        if not mine:
            return orders

        horizon = 28
        futures: Dict[int, Tuple] = {
            p.id: project_planet_future(sim, p.id, player, horizon=horizon) for p in planets
        }

        # 1. Exact safe surplus per owned planet (protect against incoming waves AND enemy home-rush!)
        enemies = [p for p in planets if p.owner != 0 and p.owner != player]
        opp_id = 2 if player == 1 else 1
        my_tot_ships = sim.total_ships(player)
        en_tot_ships = sim.total_ships(opp_id)

        surplus: Dict[int, float] = {}
        threatened: Dict[int, Tuple[float, int]] = {}  # pid -> (deficit_ships, urgent_eta)
        for p in mine:
            _, en_arr, po, ps = futures[p.id]
            lost_idx = -1
            min_hold = p.ships
            for t in range(horizon):
                if po[t] <= 0:
                    lost_idx = t
                    break
                if ps[t] < min_hold:
                    min_hold = float(ps[t])
            if lost_idx >= 0:
                deficit = float(ps[lost_idx]) + 2.0
                surplus[p.id] = 0.0
                threatened[p.id] = (deficit, lost_idx + 1)
            else:
                # Light opening home-rush guard (only when we own 1 planet and haven't expanded yet)
                garrison = 1.0
                if len(mine) == 1 and enemies:
                    nearest_en = min(enemies, key=lambda e: sim.planet_eta_ticks(e, p))
                    eta_en = sim.planet_eta_ticks(nearest_en, p)
                    growth_during = (p.production / PRODUCTION_DIVISOR) * sim.tick_dt * eta_en
                    garrison = max(1.0, min(p.ships * 0.35, nearest_en.ships * 0.45 - growth_during))
                surplus[p.id] = max(0.0, min_hold - garrison)

        rem_surplus = dict(surplus)
        sim_ships: Dict[int, float] = {p.id: float(p.ships) for p in mine}
        planned_to: Dict[int, float] = {}
        launches_per_src: Dict[int, int] = {}

        def issue_send(src_p: Planet, dst_id: int, desired: float, *, allow_full: bool = False) -> float:
            cur_base = sim_ships.get(src_p.id, 0.0)
            avail = cur_base - 1.0 if allow_full else rem_surplus.get(src_p.id, 0.0)
            amt = min(avail, desired)
            if amt < 1.5 or cur_base < 2.0:
                return 0.0
            pct = _ships_to_percent(cur_base, amt)
            if pct < 5:
                return 0.0
            sent = math.floor(cur_base * (pct / 100.0) + 1e-9)
            if sent < 1.0:
                return 0.0
            orders.append(SendOrder(src_p.id, dst_id, pct))
            sim_ships[src_p.id] = max(0.0, cur_base - sent)
            rem_surplus[src_p.id] = max(0.0, rem_surplus.get(src_p.id, 0.0) - sent)
            launches_per_src[src_p.id] = launches_per_src.get(src_p.id, 0) + 1
            planned_to[dst_id] = planned_to.get(dst_id, 0.0) + sent
            return sent

        # 2. Mid-flight fleet redirects: rescue wasted fleets or redirect to threatened/stealable targets
        if self.enable_redirect and sim.allow_redirect:
            for fl in sim.fleets.values():
                if fl.owner != player:
                    continue
                cur_dst = sim.planets.get(fl.target)
                if cur_dst is None:
                    continue
                _, _, po, ps = futures[cur_dst.id]
                if cur_dst.owner == player and po[-1] > 0 and ps[-1] > fl.ships + 18.0:
                    best_alt = self._best_redirect_target(sim, fl, player, futures, threatened)
                    if best_alt is not None and best_alt != cur_dst.id:
                        orders.append(RedirOrder(fl.id, best_alt))

        # 3. Urgent defense of threatened planets (using full safe ships from non-threatened helpers)
        for pid_def, (deficit, urgent_eta) in sorted(
            threatened.items(), key=lambda kv: -sim.planets[kv[0]].production
        ):
            dst = sim.planets[pid_def]
            need = deficit - planned_to.get(pid_def, 0.0)
            if need <= 0.0:
                continue
            helpers = [
                p
                for p in mine
                if p.id != pid_def
                and p.id not in threatened
                and sim_ships.get(p.id, 0.0) >= 3.0
                and sim.planet_eta_ticks(p, dst) <= urgent_eta
            ]
            helpers.sort(key=lambda p: sim.planet_eta_ticks(p, dst))
            if sum(max(0.0, sim_ships[h.id] - 1.0) for h in helpers) >= need:
                for h in helpers:
                    if need <= 0.0:
                        break
                    sent = issue_send(h, dst.id, need + 1.0, allow_full=True)
                    need -= sent
            elif urgent_eta <= 4 and sim_ships.get(pid_def, 0.0) >= 3.0:
                # Planet is doomed -> counter-capture a weak enemy planet if possible, else evacuate to safest ally!
                killable_en = [
                    e
                    for e in enemies
                    if e.ships + (e.production / PRODUCTION_DIVISOR) * sim.tick_dt * sim.planet_eta_ticks(dst, e) + 2.0
                    < sim_ships[pid_def]
                ]
                if killable_en:
                    evac_target = max(killable_en, key=lambda e: e.production / max(5.0, e.ships + sim.planet_eta_ticks(dst, e)))
                else:
                    safe_dests = [p for p in mine if p.id != pid_def and p.id not in threatened]
                    evac_target = (
                        min(safe_dests, key=lambda p: sim.planet_eta_ticks(dst, p))
                        if safe_dests
                        else (min(enemies, key=lambda e: sim.planet_eta_ticks(dst, e)) if enemies else None)
                    )
                if evac_target is not None:
                    issue_send(dst, evac_target.id, sim_ships[pid_def], allow_full=True)

        # 4. Target-centric coordinated multi-planet attacks & PROJECTED army-parity neutral expansion!
        # Count ships AFTER current in-flight neutral captures settle so we never double-spend into neutrals!
        proj_my_ships = sum(float(futures[p.id][3][-1]) for p in planets if futures[p.id][2][-1] > 0)
        proj_en_ships = sum(float(futures[p.id][3][-1]) for p in planets if futures[p.id][2][-1] < 0)
        neutral_spend_budget = max(0.0, proj_my_ships - proj_en_ships * 0.88)
        if len(mine) == 1 and proj_my_ships >= 85.0 and proj_my_ships >= proj_en_ships * 0.88:
            neutral_spend_budget = max(neutral_spend_budget, 34.0)

        enemy_cx = sum(p.x for p in enemies) / max(1, len(enemies)) if enemies else sim.width * 0.5
        enemy_cy = sum(p.y for p in enemies) / max(1, len(enemies)) if enemies else sim.height * 0.5

        scored_targets: List[Tuple[float, Planet, List[Tuple[int, Planet]], float]] = []
        for dst in planets:
            _, en_arr, po, ps = futures[dst.id]
            if po[-1] > 0 and dst.owner == player:
                continue
            attackers = [
                (sim.planet_eta_ticks(p, dst), p)
                for p in mine
                if p.id != dst.id and rem_surplus.get(p.id, 0.0) >= 2.0 and launches_per_src.get(p.id, 0) < 2
            ]
            if not attackers:
                continue
            attackers.sort(key=lambda kv: kv[0])

            closest_eta = attackers[0][0]
            wave: List[Tuple[int, Planet]] = []
            wave_ships = 0.0
            need = 0.0
            for eta_p, p in attackers:
                if eta_p > closest_eta + 12 and wave:
                    break
                t_idx = min(horizon - 1, eta_p - 1)
                owner_eta = po[t_idx]
                if owner_eta > 0:
                    continue
                extra = (
                    (dst.production / PRODUCTION_DIVISOR) * sim.tick_dt * max(0, eta_p - horizon)
                    if owner_eta < 0
                    else 0.0
                )
                need = float(ps[t_idx]) + extra + 2.0 - planned_to.get(dst.id, 0.0)
                wave.append((eta_p, p))
                wave_ships += rem_surplus[p.id]
                if wave_ships >= need:
                    break

            if not wave or need <= 0.0 or wave_ships < need:
                continue

            avg_eta = sum(e for e, _ in wave) / len(wave)
            travel_sec = avg_eta * sim.tick_dt
            prod_rate = dst.production / PRODUCTION_DIVISOR
            nearest_en_eta = min((sim.planet_eta_ticks(e, dst) for e in enemies), default=30)

            t_main = min(horizon - 1, wave[-1][0] - 1)
            owner_at_land = po[t_main]

            if dst.owner == 0 and owner_at_land == 0:
                if need > neutral_spend_budget:
                    continue
                if nearest_en_eta + 4 < avg_eta:
                    continue
                mult = 1.0
            elif dst.owner == 0 and owner_at_land < 0:
                if avg_eta > nearest_en_eta + 5:
                    continue
                mult = 2.40
            else:
                # Direct enemy capture: massive priority (especially punishing lightly-defended high-prod planets!)
                mult = 3.00 * self.aggression
                if proj_my_ships > proj_en_ships * 1.05:
                    mult *= 1.35

            denom = max(3.5, need + 2.5 * travel_sec * max(0.5, prod_rate))
            score = (prod_rate * 35.0 * mult) / denom
            scored_targets.append((score, dst, wave, need))

        scored_targets.sort(key=lambda x: -x[0])
        for score, dst, wave, _ in scored_targets:
            if score < 0.45:
                break
            # Re-verify wave can still cover dst's exact needed ships
            t_main = min(horizon - 1, wave[-1][0] - 1)
            _, _, po, ps = futures[dst.id]
            extra = (
                (dst.production / PRODUCTION_DIVISOR) * sim.tick_dt * max(0, wave[-1][0] - horizon)
                if po[t_main] < 0
                else 0.0
            )
            need = float(ps[t_main]) + extra + 2.0 - planned_to.get(dst.id, 0.0)
            if need <= 0.0:
                continue
            is_pure_neutral = (dst.owner == 0 and po[t_main] == 0)
            if is_pure_neutral and need > neutral_spend_budget:
                continue
            avail_wave = sum(rem_surplus.get(p.id, 0.0) for _, p in wave if launches_per_src.get(p.id, 0) < 2)
            if avail_wave < need:
                continue

            rem_need = need
            spent_total = 0.0
            for _, p in wave:
                if rem_need <= 0.0 or launches_per_src.get(p.id, 0) >= 2:
                    continue
                sent = issue_send(p, dst.id, rem_need)
                rem_need -= sent
                spent_total += sent
            if is_pure_neutral:
                neutral_spend_budget = max(0.0, neutral_spend_budget - spent_total)

        # 5. Aggressive pressure / frontline reinforcement when we have excess surplus
        if enemies:
            # If we have a big ship advantage and didn't spend everything, pile onto the weakest enemy planet!
            weakest_en = min(enemies, key=lambda e: e.ships - 0.15 * e.production)
            if my_tot_ships > en_tot_ships * 1.08:
                for src in mine:
                    if launches_per_src.get(src.id, 0) >= 2:
                        continue
                    if rem_surplus.get(src.id, 0.0) >= 12.0:
                        issue_send(src, weakest_en.id, rem_surplus[src.id] * 0.85)

            # Rear-planet funneling toward frontline
            if len(mine) >= 2:
                frontline = min(mine, key=lambda p: math.hypot(p.x - enemy_cx, p.y - enemy_cy))
                front_dist = math.hypot(frontline.x - enemy_cx, frontline.y - enemy_cy)
                for src in mine:
                    if launches_per_src.get(src.id, 0) > 0 or src.id == frontline.id:
                        continue
                    avail = rem_surplus.get(src.id, 0.0)
                    if avail < 14.0:
                        continue
                    src_dist = math.hypot(src.x - enemy_cx, src.y - enemy_cy)
                    if src_dist > front_dist + 35.0:
                        best_relay = min(
                            (
                                p
                                for p in mine
                                if p.id != src.id
                                and math.hypot(p.x - enemy_cx, p.y - enemy_cy) < src_dist - 25.0
                            ),
                            key=lambda p: math.hypot(p.x - src.x, p.y - src.y),
                            default=None,
                        )
                        if best_relay is not None:
                            issue_send(src, best_relay.id, avail * 0.75)

        return orders

    def _best_redirect_target(
        self,
        sim: GalconSim,
        fl,
        player: int,
        futures: Dict[int, Tuple],
        threatened: Dict[int, Tuple[float, int]],
    ) -> Optional[int]:
        best_id: Optional[int] = None
        best_score = 0.0
        for p in sim.planets.values():
            if p.id == fl.target:
                continue
            eta = sim.eta_ticks(fl.x, fl.y, p)
            if p.id in threatened:
                deficit, urg_eta = threatened[p.id]
                if eta <= urg_eta and fl.ships >= deficit * 0.5:
                    sc = (p.production * 3.0) / max(1.0, eta)
                    if sc > best_score:
                        best_score = sc
                        best_id = p.id
            elif p.owner != player:
                _, _, po, ps = futures[p.id]
                t_idx = min(len(ps) - 1, eta - 1)
                if po[t_idx] <= 0 and fl.ships > ps[t_idx] + 1.0:
                    sc = (p.production * (2.0 if p.owner != 0 else 1.2)) / max(2.0, ps[t_idx] + eta * 1.5)
                    if sc > best_score:
                        best_score = sc
                        best_id = p.id
        return best_id
