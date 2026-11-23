# Orbit Wars Bot Improvement: `#2 Hober Malloc` Midgame Forensics & Gen 9 (`PT-Gen9`) Roadmap

> **Bot Improvement Track:** Empirical diagnosis of why `1.89M PT-Gen8` (~1,740 Elo) matches Kaggle `#2 Hober Malloc` (`1800.1`) through the opening (`t = 0–25`) and then collapses around the midgame tipping point (`t = 40–60`).
>
> *Interactive charts & policy audit:* [`docs/boss_forensics.html`](boss_forensics.html)  
> *Model progression report:* [`docs/status_report.html`](status_report.html)

---

## 1. Economy & Fleet Trajectory (Mean Across All 6 Boss Seeds `200–205`)

Measured via [`scripts/analyze_boss_gap.py`](../scripts/analyze_boss_gap.py) paired with **860 exact counterfactual queries** against `#2 Hober Malloc`'s full-TTA JAX policy on `PT-Gen8`'s live board states.

| Tick | Planets (`PT-Gen8` vs `#2`) | Prod/tick (`PT-Gen8` vs `#2`) | Total Ships (`PT-Gen8` vs `#2`) | In-Flight % (`PT-Gen8` vs `#2`) | Phase |
|---:|---:|---:|---:|---:|---|
| `t=0` | `1.0` vs `1.0` | `3.0` vs `3.0` (`+0.0`) | `10.0` vs `10.0` (`+0.0`) | `0.0%` vs `0.0%` | Opening |
| `t=10` | `1.5` vs `1.7` | `4.3` vs `5.2` (`-0.8`) | `37.7` vs `37.2` (**`+0.5`**) | `56.9%` vs `57.4%` | Opening (Dead Even) |
| `t=20` | `3.0` vs `3.5` | `10.5` vs `12.7` (`-2.2`) | `87.0` vs `88.3` (`-1.3`) | `58.4%` vs `50.4%` | Opening (`79.7%` exact match) |
| `t=30` | `4.8` vs `6.3` | `17.3` vs `21.3` (`-4.0`) | `190.5` vs `202.2` (`-11.7`) | `44.4%` vs `42.6%` | Early Midgame Drift |
| `t=40` | `7.0` vs `8.7` | `24.5` vs `29.0` (`-4.5`) | `313.8` vs `367.2` (`-53.3`) | `53.4%` vs `59.0%` | Ship Bleed Accelerates |
| `t=50` | `8.7` vs `9.7` | `29.7` vs `32.5` (`-2.8`) | `414.0` vs `545.8` (**`-131.8`**) | `59.3%` vs `62.6%` | **Tipping Point** (Prod nearly tied, `-132` ships!) |
| `t=60` | `8.2` vs `12.3` | `27.2` vs `39.7` (`-12.5`) | `373.7` vs `582.0` (`-208.3`) | `68.2%` vs `68.6%` | Frontline Collapse |
| `t=75` | `5.7` vs `15.2` | `18.2` vs `49.3` (`-31.2`) | `389.7` vs `865.7` (`-476.0`) | `59.8%` vs `78.5%` | Snowball Blowout |
| `t=90` | `2.7` vs `17.5` | `6.8` vs `60.0` (`-53.2`) | `294.8` vs `1398.8` (`-1104.0`) | `43.8%` vs `80.6%` | Endgame Cleanup |

---

## 2. The 5 Root Causes (and What Was Ruled Out)

- **Ruled Out — Small-Planet Expansion Bias:** `PT-Gen8` captures virtually the same number of neutrals (`9.67` vs. `10.00`/game) with slightly *higher* average production (`3.28` vs. `3.10`) and an almost identical tier mix (`prod <= 2 / 3 / >= 4`: `18 / 11 / 29` vs. `21 / 11 / 28`).

1. **Impulsive Midgame All-In Shuffling (`58.3%` of All Midgame Policy Errors):**
   - Exact counterfactual match with `#2 Hober Malloc` drops from **`79.7%`** (`98/123` decisions in `t = 0..25`) down to **`66.3%`** (`416/627` decisions in `t = 26..65`).
   - Breakdown of all `211` midgame disagreements:
     - **`94` (`44.5%`):** `PT-Gen8 REINFORCE` when `#2` would **`HOLD`** — premature all-in shuffles leaving frontline garrisons empty.
     - **`41` (`19.4%`):** `PT-Gen8 HOLD` when `#2` would **`REINFORCE`** — mistimed defensive transfers.
     - **`29` (`13.7%`):** `PT-Gen8 ENEMY` attack when `#2` would **`HOLD`** — impulsive solo pokes that bounce.
     - **`18` (`8.5%`):** `PT-Gen8 HOLD` when `#2` would expand to a **`NEUTRAL`**.
     - **`10` (`4.7%`):** `PT-Gen8 HOLD` when `#2` would launch an **`ENEMY`** strike.
2. **`3.14×` Neutral Overkill (`1,521` vs. `485` Excess Ships):** Blanket `force_all_in=True` dumps 3× more surplus ships onto neutrals, explaining the `-131.8` ship deficit at `t = 50` despite near-equal production.
3. **`+55%` More Bounced / Failed Attacks (`235` vs. `152` Bounces; `3,074` vs. `2,573` Ships Lost):** Solo waves bounce while `#2` flips **`2.03×` more enemy planets (`124` vs. `61`)**.
4. **Half the Midgame Offensive Pressure (`70` vs. `135` Enemy Attacks in `t = 26..65`):** Neutral (`63` vs. `64`) and reinforce (`344` vs. `337`) volume match, but `#2` launches **`1.93×` more direct strikes on our planets**.
5. **Comet Blindness (`0` vs. `5`) & Late-Game Tempo (`3.12` vs. `9.46` Fleets/Active Turn):** `#2` intercepts 5 comets at `t = 50+` vs. our 0, then sustains ~80% airborne fleet utilization.

---

## 3. Gen 9 (`PT-Gen9`) Action Plan

1. **Midgame-Weighted Soft-KD + Hold-Calibration (`t = 25..75` upweight `2.5×`):** Focus training on the `t = 25..75` HOLD-vs-LAUNCH boundary and test separate thresholds (`thr_reinforce = 0.60` vs. `thr_enemy = 0.45`).
2. **Surgical Neutral Send Cap:** Cap neutral expansion sends at `ships_needed_to_capture + margin` while keeping All-In for enemy assaults (~150+ ships saved/game by `t = 50`).
3. **Anti-Bounce Attack Filter:** Hold solo attacks guaranteed to bounce (`send_ships <= projected_defense`) unless another friendly fleet arrives within `±3` ticks.
4. **Comet-Window Oversampling:** Oversample `t >= 48` comet-spawn states in Soft-KD DAgger rollouts (`0 → 3+` comets).
5. **Post-KD Midgame RL:** KL-anchored PPO against `PT-Gen8`, `Gen 7`, and `#2 Hober` to sharpen midgame offensive pressure (`70 → 120+` strikes).
