# Galcon AI — PlanetTimelineTransformer + Tactical Solver + `gbots` Adapter

A competitive, self-contained Galcon AI stack inside `AClaybrook/orbit-wars-lightzero` that cross-pollinates our proven **Orbit Wars `PlanetTimelineTransformer`** (variable-N planet tokens + 20-step arrival timelines + D4-invariant pairwise ETA/ROI edges + 8× D4 TTA + Self-Play PPO) directly to classic Galcon and Phil Hassey's official **`gbots`** protocol.

## Why Entity-Transformer Over Grid CNN for Galcon?
Instead of rasterizing planets onto a fixed `8×8` grid (`(W·H)² = 4,096` discrete actions with grid-channel reflection remapping):
- **Any map size & any planet count:** Scale-invariant coordinates work identically across `8`-planet `200×125` tactical maps, `18–24`-planet `400×240` classic Galcon maps, and larger `gbots` arenas without retraining.
- **Exact arrival timelines & pairwise ETAs:** Every flying fleet's exact landing tick is simulated into each planet's 20-step arrival tensor; pairwise flight times, ROI payback, snipe-steal flags, and defense urgency sit directly in the Transformer attention edge bias.
- **4-line exact 8× D4 symmetry:** Rotating/reflecting only `(x, y)` gives exact 8× data augmentation and 8× test-time augmentation.
- **Full `gbots` compatibility:** Speaks Phil Hassey's tab-delimited `gbots pipe` wire protocol (`/PLANET`, `/FLEET`, `/SEND`, `/REDIR`, `/TICK`, `/TOCK`) out of the box.

## Architecture
- [`galcon/sim.py`](sim.py) — Fast continuous-coordinate Galcon physics engine (`production / 50` ships/sec, `40.0` fleet speed, `0.25s` ticks, `SendOrder` + mid-flight `RedirOrder`, symmetric map generator, exact future timeline projection).
- [`galcon/heuristic_bot.py`](heuristic_bot.py) — `GalconClassicBot` greedy baseline + **`GalconTacticalBot`** (exact safe-surplus defense, enemy neutral-steal sniping, ROI expansion, frontline funneling, and wasted-fleet redirects).
- [`galcon/transformer_bot.py`](transformer_bot.py) — Reuses [`PlanetTimelineTransformer`](../orbit_wars_bot/planet_transformer.py) with `GalconTransformerAgent` (8× D4 TTA + redirect overlay) and `GalconSearchTransformerAgent` (value-guided lookahead search in `GalconSim`).
- [`galcon/gbots_client.py`](gbots_client.py) — Official `gbots pipe` stdin/stdout client.
- [`galcon/train.py`](train.py) — Multi-map training pipeline: Behavior Cloning → DAgger → **Self-Play PPO + Anchor-KL RL**.

## Quickstart

### 1. Train & Evaluate Locally (No `gbots` license required)
```bash
python -m galcon.train --bc-games 70 --dagger-games 30 --epochs 10 --ppo-iters 4 --eval-games 12
pytest -v tests/test_galcon.py
```

### 2. Plug Straight Into Phil Hassey's Official `gbots`
```bash
# Run our trained PlanetTimelineTransformer bot inside gbots:
./gbots pipe -name OurGalconPT -key YOUR_KEY \
  -exec "python3 -m galcon.gbots_client --bot transformer --checkpoint checkpoints/galcon_transformer_best.pt"

# Or run with value-guided lookahead search (~50-120 ms/tick):
./gbots pipe -name OurGalconSearch -key YOUR_KEY \
  -exec "python3 -m galcon.gbots_client --bot search --checkpoint checkpoints/galcon_transformer_best.pt"

# Or test our zero-weights analytical Tactical Solver immediately:
./gbots pipe -name OurGalconTactical -key YOUR_KEY \
  -exec "python3 -m galcon.gbots_client --bot tactical"
```

## Verified Multi-Map Benchmarks (8–22 Planets, `240×150` to `400×240`)
Trained on 60 multi-map BC games (`10,995` states, `val_pol=0.349`, `val_all=87.9%`, `val_tgt=85.8%`) + 64-episode Self-Play PPO + Anchor-KL RL (`checkpoints/galcon_transformer_best.pt`, `2.3 MB`):

| Agent | vs. `GalconTacticalBot` (12g) | vs. `GalconClassicBot` (12g) | Mean Latency |
|---|---:|---:|---:|
| **`GalconPT-PPO + Search` (8× D4)** | **`11–1–0` (`91.7%`, 52.2t)** | **`8–4–0` (`66.7%`, 79.8t)** | `55.0 ms/tick` |
| **`GalconPT-PPO Pure` (8× D4)** | `6–6–0` (`50.0%`, 78.4t) | `5–7–0` (`41.7%`, 70.4t) | `17.1 ms/tick` |
| `GalconTacticalBot` (Analytical Solver) | — | `11–4–1` (`68.8%` over 16g) | `<2 ms/tick` |

Interactive HTML5 Canvas replay viewer: [`docs/galcon_viewer.html`](../docs/galcon_viewer.html).

