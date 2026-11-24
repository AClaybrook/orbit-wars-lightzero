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

## 3. Gen 9 (`PT-Gen9`) Five-Fix Ablation Results

Every one of the 5 hypothesized fixes was implemented, unit-tested, and empirically ablated against `PT-Gen8` and `#2 Hober Malloc (1800.1)`:

| # | Intervention | Ablation Result (vs. `PT-Gen8`) | Verdict & Physics Insight |
|---:|---|---|---|
| **1a** | **Midgame-Weighted Soft-KD (`t=25..75`, `2.5×`)** on `3,383` boards (`+961` dense DAgger boards) | `val_kl`: **`0.369 → 0.247` (`-33%`)**, `val_tgt`: **`73.6% → 77.3%`**, `aux_own`: **`94.9%`** | **KEEP — Huge fidelity jump.** Two complementary runs (`Balanced` `launch_b=1.18` + `Hold-Boost` `hold_b=1.25`) each tied `PT-Gen8` `4–4` solo, and blended as **`PT-Gen9 (Ens-2x)`** beat `PT-Gen8` **`5–3` (`62.5%`, ~+90 Elo, 63t blitz KO)**! |
| **1b** | **Peaceful Hold Calibration (`reinforce_threshold=0.60`)** (never blocks threatened rescues) | **`4–2` (`66.7%`, `+120 Elo`)** on `PT-Gen8`; `0.65` lost `1–5` | **KEEP (optional decoder knob for single-net Gen 8).** On Gen 9 `Hold-Boost`, hold discipline is already internalized in neural weights. |
| **2** | **Decoder Surgical Neutral Cap (`surgical_neutral_cap`)** | **`2–4` (`33.3%`)** | **REJECTED (Key Physics Finding).** Orbit Wars fleet speed scales steeply with ship count (`1 + 5·(ln(s)/ln(1000))^1.5`), and `#2` itself sends `100%` All-In. `#2` achieves low neutral overkill by *launching earlier*, not by sending slower partial fleets. |
| **3** | **Decoder Anti-Bounce Guard (`anti_bounce_guard`)** | **`1–5` (`16.7%`)** | **REJECTED.** Hard-blocking solo attacks suppresses attrition/softening waves; bounce reduction must be learned in neural timing (which Soft-KD cut by **`-31%` to `-62%`**!). |
| **4** | **Comet-Window Dense Oversampling (`t >= 48`, `2.0×`)** | Flipped comets **`0 vs 5 → 1 vs 0`** (`Hold-Boost`) & cut `#2` comets `5 → 3` (`Ens-2x`) | **KEEP.** Integrated directly into [`scripts/distill_soft_kd.py`](../scripts/distill_soft_kd.py). |
| **5** | **Post-KD 2P Midgame KL-Anchored PPO (`--force-all-in --two-player-only --midgame-adv-weight 1.8`)** | `75%` (`6–2`) on 8 eval seeds at iter 1 (`kl=0.05`), but **`1–7`** on unseen seeds `990–997` | **REJECTED for promotion on CPU.** Reconfirms Gen 6's finding: 10–30 episode CPU PPO (`~1K` states) overfits eval seeds and degrades `val_kl = 0.247` generalization; scaling PPO past Soft-KD requires GPU-scale rollouts (`10M+` states). |

---

## 4. Before / After Forensics: `PT-Gen8` → `PT-Gen9` vs. `#2 Hober Malloc (1800.1)` (`seeds 200–205`)

| Forensics Metric (6 Boss Seeds) | `PT-Gen8` (Before) | `PT-Gen9 (Hold-Boost)` | `PT-Gen9 (Ens-2x, Winner)` | Delta (`Gen8 → Gen9`) |
|---|---:|---:|---:|---|
| **Official 2P vs. `PT-Gen8`** | — | `4–4` (`50.0%`) | **`5–3` (`62.5%`, ~+90 Elo)** | **+90 Elo (`63t` blitz KO)** |
| **Pure Boss Survival (Avg / Max)** | `111.3t` / `127t` | `110.7t` / `124t` | **`115.2t` / `140t` (seed 201!)** | **`+3.9t` avg, `+13t` new pure record** |
| **Ship Deficit at `t=20` / `t=50`** | `-1.3` / `-131.8` | **`+5.3`** / `-119.5` | **`+1.8` / `-112.5`** | **Early lead flipped positive; `+19.3` sh at `t=50`** |
| **Avg Ship Tipping Point Tick** | `t = 16.3` | `t = 26.5` (max `t=54`) | **`t = 27.8` (seed 200 held to `t=59`!)** | **`+11.5` ticks later** |
| **Premature Reinforces (`mid_26_65`)** | `94` (`44.5%` err) | **`62` (`-34%`)** | **`74` (`-21%`)** | **Slashed premature shuffles** |
| **Impulsive Enemy Bounces (`mid_26_65`)** | `29` (`13.7%` err) | **`11` (`-62%`)** | **`13` (`-55%`)** | **Over half of solo pokes eliminated** |
| **Total Bounced Attacks (Ships)** | `235` (`3,074 sh`) | **`163` (`2,146 sh`)** | **`211` (`2,785 sh` vs `#2` `3,100`)** | **Fewer bounced ships than `#2`!** |
| **Midgame Enemy Strikes (`t=26–65`)** | `70` | `46` (too passive) | **`76` (`+9%`)** | **Restored & raised offensive pressure** |
| **Neutral Overkill Ships (6g)** | `1,521` | **`1,167` (`-23%`)** | `1,890` | Hold-Boost cuts overkill `-23%` |
| **Comet Captures (`Ours vs #2`)** | `0 vs 5` | **`1 vs 0`** | `0 vs 3` | Stopped `#2`'s free comet runaway |
| **Late-Game Exact Match (`t >= 66`)** | `57.3%` | **`67.0%`** | **`61.2%`** | **`+3.9%` to `+9.7%` sharper endgame** |

