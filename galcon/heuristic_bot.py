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
    project_all_planet_futures,
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
        my_arrs, en_arrs, po_all, ps_all = project_all_planet_futures(
            sim, planets, player, horizon=horizon
        )
        futures: Dict[int, Tuple] = {
            p.id: (my_arrs[i], en_arrs[i], po_all[i], ps_all[i]) for i, p in enumerate(planets)
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


class GalconRusherBot:
    """Hyper-aggressive Galcon home-rusher and tempo-pressure sparring bot.

    Supports three realistic competitive rush styles:
    - ``"all_in"``: Punishes early neutral over-expansion immediately with a heavy
      opening home-rush wave and relentless continuous reinforcements at the enemy home.
    - ``"blitz"``: Grabs at most 1 ultra-cheap nearby neutral if available on tick 0,
      then pivots instantly into an overwhelming multi-planet blitz on enemy planets.
    - ``"adaptive"``: Waits 2-6 ticks to read enemy opening spend; the moment the enemy
      drops garrison below our strike force, fires an immediate lethal rush.
    """

    def __init__(
        self,
        *,
        style: str = "all_in",
        aggression: float = 1.0,
        enable_redirect: bool = True,
    ) -> None:
        self.style = style
        self.aggression = float(aggression)
        self.enable_redirect = bool(enable_redirect)

    def act(self, sim: GalconSim, player: int) -> List[Order]:
        orders: List[Order] = []
        planets = list(sim.planets.values())
        mine = [p for p in planets if p.owner == player]
        enemies = [p for p in planets if p.owner != 0 and p.owner != player]
        neutrals = [p for p in planets if p.owner == 0]
        if not mine:
            return orders

        horizon = 24
        my_arrs, en_arrs, po_all, ps_all = project_all_planet_futures(
            sim, planets, player, horizon=horizon
        )
        futures: Dict[int, Tuple] = {
            p.id: (my_arrs[i], en_arrs[i], po_all[i], ps_all[i]) for i, p in enumerate(planets)
        }

        # 1. Aggressive mid-flight redirect: pivot wasted/doomed fleets onto vulnerable enemy planets
        if self.enable_redirect and sim.allow_redirect and sim.fleets:
            for fl in sim.fleets.values():
                if fl.owner != player:
                    continue
                cur_dst = sim.planets.get(fl.target)
                if cur_dst is None:
                    continue
                _, _, po, ps = futures[cur_dst.id]
                eta_cur = sim.eta_ticks(fl.x, fl.y, cur_dst)
                t_cur = min(horizon - 1, max(0, eta_cur - 1))
                wasted_friendly = cur_dst.owner == player and po[-1] > 0 and ps[-1] > fl.ships + 12.0
                doomed_attack = cur_dst.owner != player and po[t_cur] <= 0 and fl.ships < ps[t_cur] * 0.75
                if wasted_friendly or doomed_attack:
                    best_alt = None
                    best_sc = 0.0
                    for cand in enemies + neutrals:
                        if cand.id == cur_dst.id:
                            continue
                        eta = sim.eta_ticks(fl.x, fl.y, cand)
                        _, _, cpo, cps = futures[cand.id]
                        t_idx = min(horizon - 1, max(0, eta - 1))
                        if cpo[t_idx] <= 0 and fl.ships > cps[t_idx] + 1.0:
                            mult = 2.6 if cand.owner != 0 else 0.9
                            sc = (cand.production * mult) / max(2.0, cps[t_idx] + eta * 1.4)
                            if sc > best_sc:
                                best_sc = sc
                                best_alt = cand.id
                    if best_alt is not None:
                        orders.append(RedirOrder(fl.id, best_alt))

        if not enemies:
            # Cleanup any remaining neutrals if opponent only has flying fleets
            if neutrals:
                target_n = min(neutrals, key=lambda n: n.ships / max(10.0, n.production))
                for src in mine:
                    if src.ships >= 8.0:
                        orders.append(SendOrder(src.id, target_n.id, 60))
            return orders

        # Identify primary enemy target (weakest / highest-prod killable enemy planet)
        def enemy_target_score(src: Planet, e: Planet) -> float:
            eta = sim.planet_eta_ticks(src, e)
            _, _, po, ps = futures[e.id]
            t_idx = min(horizon - 1, max(0, eta - 1))
            if po[t_idx] > 0:
                return -1e6
            extra = (e.production / PRODUCTION_DIVISOR) * sim.tick_dt * max(0, eta - horizon)
            def_ships = float(ps[t_idx]) + extra
            return (e.production * 2.2 + 30.0) / max(4.0, def_ships + 1.1 * eta)

        # Opening phase (first ~16 ticks)
        if sim.tick <= 16 and len(mine) == 1:
            home = mine[0]
            en_home = max(enemies, key=lambda e: e.production)
            eta_home = sim.planet_eta_ticks(home, en_home)
            _, _, _, ps_en = futures[en_home.id]
            t_idx = min(horizon - 1, max(0, eta_home - 1))
            en_ships_at_land = float(ps_en[t_idx]) + (en_home.production / PRODUCTION_DIVISOR) * sim.tick_dt * max(
                0, eta_home - horizon
            )

            # Check if our own home is currently being rushed
            _, en_arr_home, po_home, _ = futures[home.id]
            incoming_on_home = float(en_arr_home.sum())

            if self.style == "blitz" and sim.tick == 0 and neutrals:
                # Grab 1 ultra-cheap high-prod neutral nearby before rushing
                cheap = [
                    n
                    for n in neutrals
                    if n.ships <= 18.0 and n.production >= 45.0 and sim.planet_eta_ticks(home, n) <= 10
                ]
                if cheap:
                    best_n = max(cheap, key=lambda n: n.production / max(3.0, n.ships + sim.planet_eta_ticks(home, n)))
                    pct = _ships_to_percent(home.ships, best_n.ships + 3.0)
                    if 5 <= pct <= 30:
                        orders.append(SendOrder(home.id, best_n.id, pct))
                        rem = home.ships - math.floor(home.ships * pct / 100.0)
                        if rem >= 45.0:
                            orders.append(SendOrder(home.id, en_home.id, min(90, int(75 * self.aggression))))
                        return orders

            if self.style == "all_in":
                if sim.tick == 0 and home.ships >= 50.0:
                    pct = min(92, max(65, int(round(82 * self.aggression))))
                    orders.append(SendOrder(home.id, en_home.id, pct))
                    return orders
                if incoming_on_home <= home.ships * 0.4 and home.ships >= 12.0:
                    orders.append(SendOrder(home.id, en_home.id, min(85, max(50, int(round(72 * self.aggression))))))
                    return orders

            elif self.style == "adaptive":
                # Wait briefly; punish the instant enemy spends ships or has less than us
                if en_home.ships < home.ships * 0.85 or sim.tick >= 6:
                    if incoming_on_home <= home.ships * 0.5 and home.ships >= 20.0:
                        desired = min(home.ships * 0.90, max(home.ships * 0.65, en_ships_at_land + 8.0))
                        pct = _ships_to_percent(home.ships, desired)
                        if pct >= 20:
                            orders.append(SendOrder(home.id, en_home.id, pct))
                        return orders

        # Mid/late rush & continuous pressure
        for src in mine:
            _, en_arr, po, ps = futures[src.id]
            lost_idx = next((t for t in range(horizon) if po[t] <= 0), -1)
            if lost_idx >= 0:
                # If doomed in <= 3 ticks, counter-rush enemy! Otherwise hold to defend.
                if lost_idx <= 3 and src.ships >= 5.0:
                    tgt = max(enemies, key=lambda e: enemy_target_score(src, e))
                    orders.append(SendOrder(src.id, tgt.id, 100))
                continue

            min_hold = float(ps.min())
            avail = max(0.0, min(src.ships, min_hold) - 2.0)
            if avail < 7.0:
                continue

            # Snag an ultra-cheap neutral right next door if it's basically free
            free_neutrals = [
                n
                for n in neutrals
                if n.ships <= 10.0
                and n.production >= 50.0
                and sim.planet_eta_ticks(src, n) <= 7
                and futures[n.id][2][-1] <= 0
                and avail > n.ships + 5.0
            ]
            if free_neutrals and len(mine) < 5:
                bn = max(free_neutrals, key=lambda n: n.production / max(2.0, n.ships + 1.0))
                pct = _ships_to_percent(src.ships, bn.ships + 2.5)
                if 5 <= pct <= 45:
                    orders.append(SendOrder(src.id, bn.id, pct))
                    continue

            # Otherwise hammer the best enemy target relentlessly
            best_en = max(enemies, key=lambda e: enemy_target_score(src, e))
            eta = sim.planet_eta_ticks(src, best_en)
            _, _, po_e, ps_e = futures[best_en.id]
            t_idx = min(horizon - 1, max(0, eta - 1))
            if po_e[t_idx] > 0:
                # Already capturing this target; pick next enemy or reinforce frontline
                other_en = [e for e in enemies if e.id != best_en.id]
                if not other_en:
                    continue
                best_en = max(other_en, key=lambda e: enemy_target_score(src, e))
            send_ratio = min(0.92, max(0.55, 0.72 * self.aggression))
            pct = _ships_to_percent(src.ships, avail * send_ratio)
            if pct >= 15:
                orders.append(SendOrder(src.id, best_en.id, pct))

        return orders


class GalconMacroBot:
    """Economic expansion & late-game snowball sparring bot.

    Prioritizes fast high-ROI neutral expansion while keeping a strict defensive
    garrison against enemy rushes, then transitions into heavy multi-planet waves
    once holding a production advantage.
    """

    def __init__(self, *, expand_bias: float = 1.35) -> None:
        self.expand_bias = float(expand_bias)

    def act(self, sim: GalconSim, player: int) -> List[Order]:
        orders: List[Order] = []
        planets = list(sim.planets.values())
        mine = [p for p in planets if p.owner == player]
        enemies = [p for p in planets if p.owner != 0 and p.owner != player]
        if not mine:
            return orders

        horizon = 24
        my_arrs, en_arrs, po_all, ps_all = project_all_planet_futures(
            sim, planets, player, horizon=horizon
        )
        futures = {
            p.id: (my_arrs[i], en_arrs[i], po_all[i], ps_all[i]) for i, p in enumerate(planets)
        }
        opp = 2 if player == 1 else 1
        my_prod = sim.total_production(player)
        en_prod = sim.total_production(opp)
        my_ships = sim.total_ships(player)
        en_ships = sim.total_ships(opp)

        claimed: Dict[int, float] = {}
        threatened: Dict[int, Tuple[float, int]] = {}
        surplus: Dict[int, float] = {}

        for p in mine:
            _, _, po, ps = futures[p.id]
            lost_idx = next((t for t in range(horizon) if po[t] <= 0), -1)
            if lost_idx >= 0:
                threatened[p.id] = (float(ps[lost_idx]) + 2.0, lost_idx + 1)
                surplus[p.id] = 0.0
            else:
                garrison = 2.0
                if len(mine) <= 2 and enemies:
                    nearest_en = min(enemies, key=lambda e: sim.planet_eta_ticks(e, p))
                    eta_e = sim.planet_eta_ticks(nearest_en, p)
                    growth = (p.production / PRODUCTION_DIVISOR) * sim.tick_dt * eta_e
                    garrison = max(2.0, min(p.ships * 0.45, nearest_en.ships * 0.55 - growth))
                surplus[p.id] = max(0.0, min(p.ships, float(ps.min())) - garrison)

        # Defend threatened planets first
        for pid, (deficit, urg_eta) in sorted(threatened.items(), key=lambda kv: -sim.planets[kv[0]].production):
            dst = sim.planets[pid]
            need = deficit
            for h in sorted(mine, key=lambda x: sim.planet_eta_ticks(x, dst)):
                if h.id == pid or h.id in threatened or surplus[h.id] < 3.0:
                    continue
                if sim.planet_eta_ticks(h, dst) <= urg_eta:
                    send_amt = min(surplus[h.id], need + 1.0)
                    pct = _ships_to_percent(h.ships, send_amt)
                    if pct >= 5:
                        orders.append(SendOrder(h.id, pid, pct))
                        surplus[h.id] -= send_amt
                        need -= send_amt
                    if need <= 0:
                        break

        # Expand to high-ROI neutrals or crush enemies once ahead on economy
        boom_phase = (my_prod < en_prod * 1.30) and (sim.tick < 110)
        for src in mine:
            avail = surplus.get(src.id, 0.0)
            if avail < 6.0:
                continue
            best_t: Optional[Planet] = None
            best_need = 0.0
            best_sc = 0.25
            for dst in planets:
                if dst.id == src.id or dst.owner == player:
                    continue
                eta = sim.planet_eta_ticks(src, dst)
                _, _, po, ps = futures[dst.id]
                t_idx = min(horizon - 1, max(0, eta - 1))
                if po[t_idx] > 0:
                    continue
                extra = (
                    (dst.production / PRODUCTION_DIVISOR) * sim.tick_dt * max(0, eta - horizon)
                    if po[t_idx] < 0
                    else 0.0
                )
                need = float(ps[t_idx]) + extra + 2.0 - claimed.get(dst.id, 0.0)
                if need <= 0.0 or avail < need:
                    continue
                prod_rate = dst.production / PRODUCTION_DIVISOR
                if dst.owner == 0 and po[t_idx] == 0:
                    mult = self.expand_bias if boom_phase else 0.85
                elif dst.owner == 0 and po[t_idx] < 0:
                    mult = 2.0
                else:
                    mult = 2.4 if (my_ships > en_ships * 0.95 or not boom_phase) else 1.1
                sc = (prod_rate * 30.0 * mult) / max(3.0, need + 0.65 * eta)
                if sc > best_sc:
                    best_sc = sc
                    best_t = dst
                    best_need = need
            if best_t is not None:
                pct = _ships_to_percent(src.ships, best_need)
                if pct >= 5:
                    orders.append(SendOrder(src.id, best_t.id, pct))
                    claimed[best_t.id] = claimed.get(best_t.id, 0.0) + best_need
                    surplus[src.id] = max(0.0, avail - best_need)
        return orders

