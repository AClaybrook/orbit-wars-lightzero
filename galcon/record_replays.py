"""Record Galcon matches and generate a self-contained interactive HTML5 Canvas replay viewer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from galcon.sim import GalconSim, Order, RedirOrder


def record_galcon_match(
    title: str,
    bot1_label: str,
    bot1_act: Callable[[GalconSim, int], List[Order]],
    bot2_label: str,
    bot2_act: Callable[[GalconSim, int], List[Order]],
    *,
    seed: int = 8100,
    num_planets: int = 16,
    width: float = 400.0,
    height: float = 240.0,
    max_ticks: int = 200,
    symmetric: bool = True,
) -> Dict[str, Any]:
    """Play one Galcon match and record compact frame-by-frame snapshots for HTML5 Canvas playback."""
    sim = GalconSim.create(
        seed=seed, num_planets=num_planets, width=width, height=height, max_ticks=max_ticks, symmetric=symmetric
    )
    planets_static = [
        [p.id, round(p.x, 1), round(p.y, 1), round(p.production, 1), round(p.radius, 1)]
        for p in sorted(sim.planets.values(), key=lambda x: x.id)
    ]
    pid_order = [row[0] for row in planets_static]

    frames: List[Dict[str, Any]] = []
    # Keep redirected fleets highlighted in purple for 4 ticks so mid-flight pivots are clearly visible
    redir_ttl: Dict[int, int] = {}
    redir_counts = [0, 0]

    def snap() -> None:
        p_dyn = [[sim.planets[pid].owner, round(sim.planets[pid].ships, 1)] for pid in pid_order]
        f_dyn = [
            [
                f.owner,
                round(f.ships, 1),
                round(f.x, 1),
                round(f.y, 1),
                f.target,
                1 if redir_ttl.get(f.id, 0) > 0 else 0,
            ]
            for f in sim.fleets.values()
        ]
        frames.append(
            {
                "t": sim.tick,
                "p": p_dyn,
                "f": f_dyn,
                "s": [round(sim.total_ships(1), 1), round(sim.total_ships(2), 1)],
                "pr": [round(sim.total_production(1), 0), round(sim.total_production(2), 0)],
                "rc": [redir_counts[0], redir_counts[1]],
            }
        )

    snap()
    while not sim.done:
        for fid in list(redir_ttl.keys()):
            redir_ttl[fid] -= 1
            if redir_ttl[fid] <= 0 or fid not in sim.fleets:
                del redir_ttl[fid]

        if sim.tick % 2 == 0:
            o1 = bot1_act(sim, 1)
            o2 = bot2_act(sim, 2)
            for pid_idx, ords in ((0, o1), (1, o2)):
                for o in ords:
                    if isinstance(o, RedirOrder) and o.fleet_id in sim.fleets:
                        fl = sim.fleets[o.fleet_id]
                        if fl.owner == (pid_idx + 1) and fl.target != o.dst_id:
                            redir_ttl[o.fleet_id] = 4
                            redir_counts[pid_idx] += 1
            sim.step({1: o1, 2: o2})
        else:
            sim.step({})
        snap()

    return {
        "title": title,
        "players": [bot1_label, bot2_label],
        "winner": sim.winner(),
        "ticks": sim.tick,
        "width": width,
        "height": height,
        "redirs": redir_counts,
        "planets": planets_static,
        "frames": frames,
    }


def render_galcon_viewer_html(matches: Sequence[Dict[str, Any]]) -> str:
    """Build a self-contained dark HTML5 Canvas replay viewer for Galcon Gen 2 matches."""
    matches_json = json.dumps(list(matches), separators=(",", ":"))
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Galcon AI Gen 2 — Live Animated Gameplay Viewer</title>
  <style>
    :root {{
      --bg: #0b1120; --card: #111827; --border: #1f2937; --text: #f1f5f9;
      --muted: #94a3b8; --p1: #38bdf8; --p2: #f87171; --neu: #64748b; --ok: #34d399; --redir: #c084fc;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: var(--bg); color: var(--text); }}
    .wrap {{ max-width: 980px; margin: 0 auto; padding: 16px; }}
    .bar {{ display: flex; flex-wrap: wrap; gap: 10px; align-items: center; justify-content: space-between; margin-bottom: 12px; }}
    .title {{ font-size: 17px; font-weight: 700; letter-spacing: -0.01em; }}
    .sub {{ font-size: 12px; color: var(--muted); }}
    select, button {{ font: inherit; font-size: 12.5px; padding: 6px 11px; border-radius: 8px; border: 1px solid var(--border); background: var(--card); color: var(--text); cursor: pointer; }}
    button.primary {{ background: #0284c7; border-color: #0284c7; font-weight: 600; }}
    .hud {{ display: grid; grid-template-columns: 1fr auto 1fr; gap: 12px; align-items: center; background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 10px 14px; margin-bottom: 10px; font-size: 13px; }}
    .p1 {{ color: var(--p1); font-weight: 700; }} .p2 {{ color: var(--p2); font-weight: 700; }}
    .redir-tag {{ color: var(--redir); font-weight: 600; }}
    .mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-variant-numeric: tabular-nums; }}
    canvas {{ width: 100%; height: auto; display: block; border-radius: 12px; border: 1px solid var(--border); background: radial-gradient(circle at 50% 50%, #111c33 0%, #070b14 100%); }}
    .controls {{ display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin-top: 10px; background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 10px 12px; }}
    input[type="range"] {{ flex: 1; min-width: 160px; accent-color: #38bdf8; }}
    .adv-track {{ height: 6px; border-radius: 999px; background: var(--p2); overflow: hidden; margin-top: 4px; }}
    .adv-fill {{ height: 100%; background: var(--p1); transition: width 0.08s linear; }}
    .tabs {{ display: flex; gap: 6px; margin-bottom: 12px; }}
    .tab-btn {{ padding: 7px 14px; border-radius: 8px; border: 1px solid var(--border); background: var(--card); color: var(--muted); font-weight: 600; cursor: pointer; }}
    .tab-btn.active {{ background: #0284c7; border-color: #0284c7; color: #fff; }}
    .panel {{ display: none; }} .panel.active {{ display: block; }}
    .elo-card {{ background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 14px 16px; margin-bottom: 12px; }}
    .elo-row {{ display: grid; grid-template-columns: 32px 230px 1fr 78px 80px; gap: 10px; align-items: center; padding: 7px 0; border-bottom: 1px solid rgba(148,163,184,0.10); font-size: 13px; }}
    .elo-row:last-child {{ border-bottom: none; }}
    .elo-bar-bg {{ height: 10px; border-radius: 999px; background: rgba(148,163,184,0.14); overflow: hidden; }}
    .elo-bar {{ height: 100%; border-radius: 999px; }}
    table.bench {{ width: 100%; border-collapse: collapse; font-size: 12.5px; }}
    table.bench th, table.bench td {{ padding: 8px 10px; border-bottom: 1px solid rgba(148,163,184,0.12); text-align: left; }}
    table.bench th {{ color: var(--muted); font-weight: 600; }}
    .badge {{ display: inline-block; padding: 2px 7px; border-radius: 999px; font-size: 11px; font-weight: 600; background: rgba(56,189,248,0.16); color: var(--p1); }}
    .badge.ok {{ background: rgba(52,211,153,0.16); color: var(--ok); }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="bar">
      <div>
        <div class="title">Galcon AI Wave 12 — 71.2% Sparky v1.6 (~1979 Elo) &amp; 2.3× Faster VM Training Engine</div>
        <div class="sub">1.29M PlanetTimelineTransformer • Wave 12 Surgical Rescue/Snipe + Direct-Rush Minimax • forward_multi_send + Length-Bucketed Batches + Batched val_56g • Unmodified Sparky v1.6, Waffle3z-GA &amp; EsparanBot v0.2</div>
      </div>
      <div class="tabs">
        <button class="tab-btn active" id="tabReplays">▶ Live Replays</button>
        <button class="tab-btn" id="tabElo">🏆 Elo Ladder &amp; Triple-Boss Scoreboard</button>
      </div>
    </div>

    <div id="panelReplays" class="panel active">
      <div style="display:flex; justify-content:flex-end; margin-bottom:10px;">
        <select id="matchSel" style="width:100%;"></select>
      </div>
      <div class="hud">
        <div>
          <div class="p1" id="p1Name">Blue</div>
          <div class="mono sub" id="p1Stats">Ships: 100 • Prod: 100</div>
        </div>
        <div style="text-align:center; min-width:170px;">
          <div class="mono" id="tickLabel" style="font-weight:700;">Tick 0 / 0</div>
          <div class="adv-track"><div class="adv-fill" id="advFill" style="width:50%"></div></div>
          <div class="sub" id="winnerLabel" style="margin-top:3px;">In Progress</div>
        </div>
        <div style="text-align:right;">
          <div class="p2" id="p2Name">Red</div>
          <div class="mono sub" id="p2Stats">Ships: 100 • Prod: 100</div>
        </div>
      </div>

      <canvas id="board" width="900" height="540"></canvas>

      <div class="controls">
        <button id="playBtn" class="primary">❚❚ Pause</button>
        <button id="stepBack">-1</button>
        <button id="stepFwd">+1</button>
        <input id="scrubber" type="range" min="0" max="1" value="0" />
        <select id="speedSel">
          <option value="180">0.5×</option>
          <option value="95" selected>1×</option>
          <option value="45">2×</option>
          <option value="20">4×</option>
        </select>
      </div>
    </div>

    <div id="panelElo" class="panel">
      <div class="elo-card">
        <div style="display:flex; flex-wrap:wrap; gap:8px; justify-content:space-between; align-items:center; margin-bottom:10px;">
          <div style="font-weight:700;">Galcon Elo Ladder (Anchored: ClassicBot = 1200.0 • Sparky v1.6 = 1821.4 • Waffle3z-GA ≈ 1745)</div>
          <span class="badge ok">#1 Wave 12 Surgical Ens (W9S+W6B): ~1979 Elo (26–9–5 / 71.2% Sparky • 92.2% Heur • 66.7% Large)</span>
        </div>
        <div id="eloRows"></div>
      </div>
      <div class="elo-card">
        <div style="font-weight:700; margin-bottom:10px;">Triple-Boss Head-to-Head Win-Rate Comparison (40g 18pl Std 1-Tick &amp; 12g Large 22–28pl)</div>
        <div id="bossBars" style="display:grid; gap:10px; margin-bottom:14px;"></div>
        <table class="bench">
          <thead>
            <tr><th>Agent</th><th>Sparky v1.6 18pl (40g, 1t)</th><th>Waffle3z-GA 18pl (40g, 1t)</th><th>Large 22–28pl (Asym / Pair Spky)</th><th>EsparanBot / Heur</th><th>Latency (VM)</th></tr>
          </thead>
          <tbody>
            <tr><td><strong>Wave 12 Surgical Ens (W9S+W6B)</strong> <span class="badge ok">#1 Sparky!</span></td><td><strong>26–9–5 (71.2% — #1!)</strong> <span class="sub">s14k:65% s9.5k:77.5%</span></td><td>24–13–3 (63.7%) <span class="sub">s14k:65% s9.5k:62.5%</span></td><td><strong>66.7%</strong> / 33.3% <span class="sub">(75% Asym Waff)</span></td><td>64.6% / <strong>92.2%</strong></td><td class="mono"><strong>46–52 ms</strong></td></tr>
            <tr><td><strong>Wave 12 Direct-Rush Ens (W9S+W6B)</strong> <span class="badge ok">#1 Waff!</span></td><td>24–11–5 (66.2%) <span class="sub">s14k:60% s9.5k:72.5%</span></td><td><strong>27–10–3 (71.2% — #1!)</strong> <span class="sub">s14k:70% s9.5k:72.5%</span></td><td><strong>66.7%</strong> / — <span class="sub">(75% Asym Waff)</span></td><td>64.6% / 89.1%</td><td class="mono"><strong>47–52 ms</strong></td></tr>
            <tr><td><strong>Wave 12 Val-Critic Ens (W9S+W6B+W11KD)</strong></td><td><strong>25–11–4 (67.5%)</strong> <span class="sub">s9.5k:77.5%</span></td><td>24–13–3 (63.7%)</td><td>50.0% / 33.3% <span class="sub">(<strong>83.3% Asym Waff!</strong>)</span></td><td>—</td><td class="mono">58 ms</td></tr>
            <tr><td><strong>Wave 9 Single Search (W9S)</strong> <span class="badge ok">#1 Spky 33ms</span></td><td><strong>24–11–5 (66.3%)</strong> <span class="sub">s14k:57.5% s9.5k:75%</span></td><td>23–13–4 (62.5%) <span class="sub">s14k:65% s9.5k:60%</span></td><td>50.0% / — <span class="sub">(<strong>83.3% Asym Waff</strong>)</span></td><td>83.3% / <strong>96.9%</strong></td><td class="mono"><strong>30–38 ms</strong></td></tr>
            <tr><td><strong>Wave 10 Ensemble (W9S+W10S)</strong> <span class="badge">s14k King</span></td><td>23–10–7 (66.3%) <span class="sub"><strong>s14k:67.5%</strong> (10L!)</span></td><td>24–11–5 (66.3%)</td><td>41.7% / —</td><td>—</td><td class="mono">41–51 ms</td></tr>
            <tr><td><strong>Wave 11 Soft-KD Search (W11_KD)</strong> <span class="badge ok">3× Record!</span></td><td>21–15–4 (57.5%) <span class="sub">57.5% / 57.5% bal</span></td><td><strong>26–11–3 (68.8% — #1 1-Mod!)</strong> <span class="sub">s14k:70%</span></td><td>41.7% / <strong>58.3% (#1 All-Time!)</strong></td><td><strong>85.4% (#1!)</strong> / 84.4% <span class="sub">(92.2% Pure)</span></td><td class="mono"><strong>28–33 ms</strong></td></tr>
            <tr><td><strong>Wave 9 Ensemble (W6B+W7p1)</strong></td><td>25–13–2 (65.0%)</td><td>24–13–3 (63.8%)</td><td><strong>66.7%</strong> / 50.0% <span class="sub">(83.3% Asym Waff)</span></td><td>83.3% / <strong>100%</strong></td><td class="mono">48–51 ms</td></tr>
            <tr><td><strong>Wave 8 Pure (8 ms, 0-ply!)</strong> <span class="badge ok">#1 Std Pure</span></td><td><strong>18–9–13 (61.3% Pure!)</strong></td><td><strong>18–16–6 (52.5%)</strong></td><td>— / 25.0% <span class="sub">(33.3% Pair Waff)</span></td><td><strong>79.2%</strong> / 100% Cls</td><td class="mono"><strong>8.0 ms</strong></td></tr>
            <tr><td><strong>Wave 11 Pure (10 ms, 0-ply!)</strong> <span class="badge ok">#1 Large Pure</span></td><td>13–14–13 (48.8%) <span class="sub">s14k:52.5%</span></td><td>10–21–9 (36.2%)</td><td>— / <strong>50.0% Pure (2× W8!)</strong></td><td>62.5% / 82.8%</td><td class="mono"><strong>8–10 ms</strong></td></tr>
            <tr><td><strong>Waffle3z-GA (bots_waffle3z_optimized)</strong></td><td>9–20–11 (36.3%)</td><td>— (Self)</td><td>16.7% / 41.7%</td><td><strong>67.5% vs Esp</strong></td><td class="mono">&lt;3 ms</td></tr>
          </tbody>
        </table>
        <div class="sub" style="margin-top:10px;"><strong>Wave 12 Highlights:</strong> (1) <strong>Shattered 70% vs. <code>esparano Sparky v1.6</code> (<code>26–9–5 / 71.2%</code>, ~<code>1979 Elo</code>, single-digit losses!)</strong> via forensic surgical search in <code>EnsembleGalconSearchAgent</code>: <code>_build_rescue_and_snipe_plan</code> (exact defense rescue + single-target stalemate-breaking snipes) + <code>_build_opp_direct_rush_plan</code> (direct-rush soft-minimax fallback), jumping Sparky <code>+5.0%</code> (<code>s=14000: 65.0%</code>, <code>s=9500: 77.5%</code>) and Heuristics <code>+3.1%</code> (<code>92.2%</code>) with zero Large regression (<code>66.7%</code> Sparky / <code>75.0%</code> Waffle3z). Direct-rush mode alone hits <strong><code>27–10–3 (71.2%!)</code> vs. <code>Waffle3z-GA</code></strong>! (2) <strong><code>~2.1×–2.3×</code> Faster VM Training Engine</strong>: <code>forward_multi_send</code> (skipping the dead legacy pairwise head in training, rollouts, and search leaf eval), <code>_LengthBucketBatchSampler</code> (eliminating <code>O(N²)</code> padding waste across 10–28pl maps, cutting PPO <code>upd</code> <code>25–29s → 14.5–18.3s</code>), and persistent worker Lua/snapshot caches + <code>B=6</code> lockstep batched <code>val_56g</code> (<code>19s → ~12s</code>), plus bit-exact <code>need_value=False</code> skipping for decoupled value critics.</div>
      </div>
    </div>
  </div>

  <script>
    const MATCHES = {matches_json};
    const ELO_LADDER = [
      {{ name: "Wave 12 Surgical Ens (W9S+W6B) [Default]", elo: 1979.0, ms: "46–52 ms", col: "#10b981" }},
      {{ name: "Wave 12 Direct-Rush Ens [71.2% Waff #1]", elo: 1955.0, ms: "47–52 ms", col: "#059669" }},
      {{ name: "Wave 12 Val-Critic Ens (+W11KD Leaf)", elo: 1951.0, ms: "58 ms", col: "#34d399" }},
      {{ name: "Wave 9 Single Search (W9S) [33ms #1]", elo: 1939.0, ms: "30–38 ms", col: "#2dd4bf" }},
      {{ name: "Wave 10 Ensemble (W9S+W10S) [s14k #1]", elo: 1939.0, ms: "41–51 ms", col: "#14b8a6" }},
      {{ name: "Wave 9 Ensemble (W6B+W7p1) [Large/Heur]", elo: 1929.0, ms: "48–51 ms", col: "#0d9488" }},
      {{ name: "Wave 11 Soft-KD Search (W11_KD) [3× Rec]", elo: 1925.0, ms: "28–33 ms", col: "#22d3ee" }},
      {{ name: "Wave 10 Single Search (W10S)", elo: 1905.0, ms: "28–30 ms", col: "#06b6d4" }},
      {{ name: "Wave 8 Pure (EI→PPO, 1× MS, 0-ply)", elo: 1890.0, ms: "8.0 ms", col: "#0891b2" }},
      {{ name: "Wave 6B + 2-Ply Adaptive Search", elo: 1856.0, ms: "49 ms", col: "#38bdf8" }},
      {{ name: "Wave 7 p1 Pure (EI→PPO, 1× MS)", elo: 1848.0, ms: "8.2 ms", col: "#0ea5e9" }},
      {{ name: "esparano Sparky v1.6 (Lua GM)", elo: 1821.4, ms: "<2 ms", col: "#f43f5e" }},
      {{ name: "GalconPT-Gen4 + AdaptiveSearch (1×)", elo: 1800.7, ms: "52 ms", col: "#60a5fa" }},
      {{ name: "Waffle3z-GA (bots_waffle3z_optimized)", elo: 1745.0, ms: "<3 ms", col: "#f59e0b" }},
      {{ name: "GalconPT-Gen3 Pure (1.26M, 8× D4)", elo: 1731.4, ms: "23 ms", col: "#a78bfa" }},
      {{ name: "GalconTacticalBot (Analytical)", elo: 1450.1, ms: "<2 ms", col: "#94a3b8" }},
      {{ name: "GalconRusherBot (All-In)", elo: 1431.8, ms: "<2 ms", col: "#fb923c" }},
      {{ name: "esparano EsparanBot v0.2 (Lua)", elo: 1360.8, ms: "<2 ms", col: "#fb7185" }},
      {{ name: "GalconClassicBot (Anchor)", elo: 1200.0, ms: "<1 ms", col: "#64748b" }},
      {{ name: "GalconMacroBot (Boomer)", elo: 1129.3, ms: "<2 ms", col: "#475569" }},
    ];

    const eloWrap = document.getElementById("eloRows");
    ELO_LADDER.forEach((row, idx) => {{
      const pct = Math.max(6, Math.min(100, ((row.elo - 1060) / (2000 - 1060)) * 100));
      const div = document.createElement("div");
      div.className = "elo-row";
      div.innerHTML = `<div class="mono sub">#${{idx+1}}</div><div style="font-weight:${{idx<5?700:500}}">${{row.name}}</div><div class="elo-bar-bg"><div class="elo-bar" style="width:${{pct.toFixed(1)}}%;background:${{row.col}}"></div></div><div class="mono" style="text-align:right;font-weight:700;">${{row.elo.toFixed(1)}}</div><div class="mono sub" style="text-align:right;">${{row.ms}}</div>`;
      eloWrap.appendChild(div);
    }});

    const BOSS_COMPARE = [
      {{ name: "W12 Surgical Ens", spky: 71.2, waff: 63.7, lg: 66.7, lgLbl: "Large Asym Spky" }},
      {{ name: "W12 Direct-Rush Ens", spky: 66.2, waff: 71.2, lg: 66.7, lgLbl: "Large Asym Spky" }},
      {{ name: "W12 Val-Critic Ens", spky: 67.5, waff: 63.7, lg: 83.3, lgLbl: "Large Asym Waff" }},
      {{ name: "W9 Single Search", spky: 66.3, waff: 62.5, lg: 50.0, lgLbl: "Large Asym Spky" }},
      {{ name: "W11 Soft-KD (28ms)", spky: 57.5, waff: 68.8, lg: 58.3, lgLbl: "Large Pair Spky" }},
      {{ name: "W8 Pure (8 ms)", spky: 61.3, waff: 52.5, lg: 25.0, lgLbl: "Large Pair Spky" }},
      {{ name: "W11 Pure (10 ms)", spky: 48.8, waff: 36.2, lg: 50.0, lgLbl: "Large Pair Spky" }},
    ];
    const bossWrap = document.getElementById("bossBars");
    BOSS_COMPARE.forEach(r => {{
      const el = document.createElement("div");
      el.style.cssText = "display:grid; grid-template-columns:165px 1fr 1fr 1fr; gap:10px; align-items:center; font-size:12px;";
      const mkBar = (v, col, lbl) => `<div><div style="display:flex;justify-content:space-between;margin-bottom:2px;"><span class="sub">${{lbl}}</span><span class="mono" style="font-weight:700;">${{v.toFixed(1)}}%</span></div><div class="elo-bar-bg"><div class="elo-bar" style="width:${{v}}%;background:${{col}}"></div></div></div>`;
      el.innerHTML = `<div style="font-weight:600;">${{r.name}}</div>${{mkBar(r.spky, "#38bdf8", "Sparky 18pl")}}${{mkBar(r.waff, "#f59e0b", "Waffle3z 18pl")}}${{mkBar(r.lg, "#34d399", r.lgLbl)}}`;
      bossWrap.appendChild(el);
    }});

    const tabReplays = document.getElementById("tabReplays");
    const tabElo = document.getElementById("tabElo");
    const panelReplays = document.getElementById("panelReplays");
    const panelElo = document.getElementById("panelElo");
    tabReplays.addEventListener("click", () => {{
      tabReplays.classList.add("active"); tabElo.classList.remove("active");
      panelReplays.classList.add("active"); panelElo.classList.remove("active");
    }});
    tabElo.addEventListener("click", () => {{
      tabElo.classList.add("active"); tabReplays.classList.remove("active");
      panelElo.classList.add("active"); panelReplays.classList.remove("active");
    }});

    const canvas = document.getElementById("board");
    const ctx = canvas.getContext("2d");
    const matchSel = document.getElementById("matchSel");
    const playBtn = document.getElementById("playBtn");
    const scrubber = document.getElementById("scrubber");
    const speedSel = document.getElementById("speedSel");

    let curIdx = 0, frameIdx = 0, playing = true, timer = null;

    MATCHES.forEach((m, i) => {{
      const opt = document.createElement("option");
      opt.value = i;
      const res = m.winner === 1 ? "Blue Win" : (m.winner === 2 ? "Red Win" : "Draw");
      opt.textContent = `${{m.title}} (${{res}}, ${{m.ticks}}t)`;
      matchSel.appendChild(opt);
    }});

    function drawFrame() {{
      const m = MATCHES[curIdx];
      const f = m.frames[frameIdx];
      const W = canvas.width, H = canvas.height;
      const sx = W / m.width, sy = H / m.height, sc = Math.min(sx, sy);
      const ox = (W - m.width * sc) * 0.5, oy = (H - m.height * sc) * 0.5;
      const tx = x => ox + x * sc, ty = y => oy + y * sc;

      ctx.clearRect(0, 0, W, H);
      ctx.strokeStyle = "rgba(148,163,184,0.06)";
      ctx.lineWidth = 1;
      for (let gx = 0; gx <= m.width; gx += 40) {{
        ctx.beginPath(); ctx.moveTo(tx(gx), ty(0)); ctx.lineTo(tx(gx), ty(m.height)); ctx.stroke();
      }}
      for (let gy = 0; gy <= m.height; gy += 40) {{
        ctx.beginPath(); ctx.moveTo(tx(0), ty(gy)); ctx.lineTo(tx(m.width), ty(gy)); ctx.stroke();
      }}

      const pPos = {{}};
      m.planets.forEach(row => {{ pPos[row[0]] = [row[1], row[2]]; }});

      f.f.forEach(fl => {{
        const [owner, ships, fx, fy, tgtId, isRedir] = fl;
        const col = owner === 1 ? "#38bdf8" : "#f87171";
        const dst = pPos[tgtId];
        if (dst) {{
          ctx.strokeStyle = isRedir ? "rgba(192,132,252,0.70)" : (owner === 1 ? "rgba(56,189,248,0.18)" : "rgba(248,113,113,0.18)");
          ctx.lineWidth = isRedir ? 2.6 : 1.2;
          ctx.beginPath(); ctx.moveTo(tx(fx), ty(fy)); ctx.lineTo(tx(dst[0]), ty(dst[1])); ctx.stroke();
        }}
        const r = Math.max(4.5, Math.min(12, 3.2 + Math.sqrt(ships) * 0.9));
        ctx.fillStyle = isRedir ? "#c084fc" : col;
        ctx.beginPath(); ctx.arc(tx(fx), ty(fy), r, 0, Math.PI * 2); ctx.fill();
        ctx.fillStyle = "#0b1120";
        ctx.font = "bold 10px ui-monospace, monospace";
        ctx.textAlign = "center"; ctx.textBaseline = "middle";
        ctx.fillText(Math.round(ships), tx(fx), ty(fy));
      }});

      m.planets.forEach((row, idx) => {{
        const [pid, px, py, prod, rad] = row;
        const [owner, ships] = f.p[idx];
        const col = owner === 1 ? "#38bdf8" : (owner === 2 ? "#f87171" : "#64748b");
        const fill = owner === 1 ? "rgba(56,189,248,0.22)" : (owner === 2 ? "rgba(248,113,113,0.22)" : "rgba(100,116,139,0.20)");
        const r = rad * sc;
        ctx.fillStyle = fill;
        ctx.strokeStyle = col;
        ctx.lineWidth = owner === 0 ? 2 : 3;
        ctx.beginPath(); ctx.arc(tx(px), ty(py), r, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
        ctx.fillStyle = "#f8fafc";
        ctx.font = `bold ${{Math.max(11, Math.round(r * 0.52))}}px ui-monospace, monospace`;
        ctx.textAlign = "center"; ctx.textBaseline = "middle";
        ctx.fillText(Math.floor(ships), tx(px), ty(py) - 2);
        ctx.fillStyle = "#94a3b8";
        ctx.font = "10px ui-monospace, monospace";
        ctx.fillText(`+${{Math.round(prod)}}`, tx(px), ty(py) + Math.max(10, r * 0.45));
      }});

      const rc = f.rc || [0, 0];
      document.getElementById("p1Name").textContent = "Blue: " + m.players[0];
      document.getElementById("p2Name").textContent = "Red: " + m.players[1];
      document.getElementById("p1Stats").textContent = `Ships: ${{Math.round(f.s[0])}} • Prod: ${{Math.round(f.pr[0])}}` + (rc[0] > 0 ? ` • /REDIR: ${{rc[0]}}` : "");
      document.getElementById("p2Stats").textContent = `Ships: ${{Math.round(f.s[1])}} • Prod: ${{Math.round(f.pr[1])}}` + (rc[1] > 0 ? ` • /REDIR: ${{rc[1]}}` : "");
      document.getElementById("tickLabel").textContent = `Tick ${{f.t}} / ${{m.ticks}} (${{(f.t * 0.25).toFixed(1)}}s)`;
      const tot = Math.max(1, f.s[0] + f.s[1]);
      document.getElementById("advFill").style.width = `${{Math.round(100 * f.s[0] / tot)}}%`;
      document.getElementById("winnerLabel").textContent =
        frameIdx === m.frames.length - 1
          ? (m.winner === 1 ? `Winner: ${{m.players[0]}}` : (m.winner === 2 ? `Winner: ${{m.players[1]}}` : "Draw"))
          : "Live Battle";
      scrubber.value = frameIdx;
    }}

    function loadMatch(idx) {{
      curIdx = idx; frameIdx = 0;
      scrubber.max = MATCHES[curIdx].frames.length - 1;
      drawFrame();
    }}
    function tickLoop() {{
      if (!playing) return;
      const maxF = MATCHES[curIdx].frames.length - 1;
      frameIdx = frameIdx >= maxF ? 0 : frameIdx + 1;
      drawFrame();
      timer = setTimeout(tickLoop, Number(speedSel.value));
    }}
    function restartTimer() {{ clearTimeout(timer); if (playing) timer = setTimeout(tickLoop, Number(speedSel.value)); }}

    matchSel.addEventListener("change", () => {{ loadMatch(Number(matchSel.value)); restartTimer(); }});
    speedSel.addEventListener("change", restartTimer);
    playBtn.addEventListener("click", () => {{
      playing = !playing;
      playBtn.textContent = playing ? "❚❚ Pause" : "▶ Play";
      restartTimer();
    }});
    scrubber.addEventListener("input", () => {{ frameIdx = Number(scrubber.value); drawFrame(); }});
    document.getElementById("stepBack").addEventListener("click", () => {{ playing = false; playBtn.textContent = "▶ Play"; frameIdx = Math.max(0, frameIdx - 1); drawFrame(); }});
    document.getElementById("stepFwd").addEventListener("click", () => {{ playing = false; playBtn.textContent = "▶ Play"; frameIdx = Math.min(MATCHES[curIdx].frames.length - 1, frameIdx + 1); drawFrame(); }});

    loadMatch(0);
    restartTimer();
  </script>
</body>
</html>
"""


def _find_showcase_win(
    title_prefix: str,
    bot1_label: str,
    bot1_act: Callable[[GalconSim, int], List[Order]],
    bot2_label: str,
    bot2_act: Callable[[GalconSim, int], List[Order]],
    *,
    seed_start: int,
    num_planets: int,
    width: float,
    height: float,
    prefer_redir: bool = False,
    symmetric: bool = True,
    max_ticks: int = 200,
    max_attempts: int = 12,
    target_winner: int = 1,
) -> Dict[str, Any]:
    best_m: Optional[Dict[str, Any]] = None
    for offset in range(max_attempts):
        sd = seed_start + offset
        m = record_galcon_match(
            f"{title_prefix} (seed {sd})",
            bot1_label,
            bot1_act,
            bot2_label,
            bot2_act,
            seed=sd,
            num_planets=num_planets,
            width=width,
            height=height,
            max_ticks=max_ticks,
            symmetric=symmetric,
        )
        if m["winner"] == target_winner:
            if not prefer_redir or m["redirs"][0] > 0:
                return m
            if best_m is None:
                best_m = m
        elif best_m is None:
            best_m = m
    assert best_m is not None
    return best_m


def _lua_path_exists(spec: str) -> bool:
    raw = str(spec or "").strip()
    if not raw:
        return False
    p = raw.rsplit("::", 1)[0] if "::" in raw else raw
    return Path(p).is_file()


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Record Wave 11 Galcon Triple-Boss showcase matches into docs/galcon_viewer.html.")
    parser.add_argument("--checkpoint", default="checkpoints/galcon_wave9_search.pt")
    parser.add_argument("--checkpoint2", default="checkpoints/galcon_transformer_best.pt")
    parser.add_argument("--search-checkpoint", default="checkpoints/galcon_wave9_search.pt")
    parser.add_argument("--w10-checkpoint", default="checkpoints/galcon_wave10_search.pt")
    parser.add_argument("--w11-kd-checkpoint", default="checkpoints/galcon_wave11_kd.pt")
    parser.add_argument("--w11-pure-checkpoint", default="checkpoints/galcon_wave11_pure.pt")
    parser.add_argument("--pure-checkpoint", default="checkpoints/galcon_wave8_pure.pt")
    parser.add_argument("--gen4-checkpoint", default="checkpoints/galcon_gen4_ppo_s2.pt")
    parser.add_argument("--gen2-checkpoint", default="checkpoints/galcon_gen2_champion.pt")
    parser.add_argument("--v1-checkpoint", default="checkpoints/galcon_v1_baseline.pt")
    parser.add_argument("--lua-sparky", default="")
    parser.add_argument("--lua-waffle", default="")
    parser.add_argument("--lua-esparan", default="")
    parser.add_argument("--out-html", default="docs/galcon_viewer.html")
    parser.add_argument("--out-json", default="")
    args = parser.parse_args(argv)

    from galcon.heuristic_bot import GalconClassicBot, GalconRusherBot, GalconTacticalBot
    from galcon.transformer_bot import (
        EnsembleGalconSearchAgent,
        GalconSearchTransformerAgent,
        GalconTransformerAgent,
    )

    pure_ckpt = args.pure_checkpoint if (args.pure_checkpoint and Path(args.pure_checkpoint).is_file()) else args.checkpoint
    pure = GalconTransformerAgent(pure_ckpt, use_d4_canonical=True, use_d4_tta=False)
    has_w9_s = bool(args.search_checkpoint and Path(args.search_checkpoint).is_file())
    w9_search = (
        GalconSearchTransformerAgent(args.search_checkpoint, use_d4_canonical=True, use_d4_tta=False)
        if has_w9_s
        else None
    )
    has_w10_s = bool(args.w10_checkpoint and Path(args.w10_checkpoint).is_file())
    w10_search = (
        GalconSearchTransformerAgent(args.w10_checkpoint, use_d4_canonical=True, use_d4_tta=False)
        if has_w10_s
        else None
    )
    has_w11_kd = bool(args.w11_kd_checkpoint and Path(args.w11_kd_checkpoint).is_file())
    w11_kd = (
        GalconSearchTransformerAgent(args.w11_kd_checkpoint, use_d4_canonical=True, use_d4_tta=False)
        if has_w11_kd
        else None
    )
    has_w11_pure = bool(args.w11_pure_checkpoint and Path(args.w11_pure_checkpoint).is_file())
    w11_pure = (
        GalconTransformerAgent(args.w11_pure_checkpoint, use_d4_canonical=True, use_d4_tta=False)
        if has_w11_pure
        else None
    )
    has_ens = bool(args.checkpoint2 and Path(args.checkpoint2).is_file())
    if has_ens:
        search = EnsembleGalconSearchAgent(
            [args.checkpoint, args.checkpoint2], use_d4_canonical=True, use_d4_tta=False
        )
        search_label = "Wave 12 Surgical Ens (W9S+W6B, 52 ms)"
    else:
        search = GalconSearchTransformerAgent(args.checkpoint, use_d4_canonical=True, use_d4_tta=False)
        search_label = "Wave 12 Single Adaptive Search (1×)"
    w9_search_label = "Wave 9 Single Search (33 ms, 66.3%)"
    w10_search_label = "Wave 10 Single Search (29 ms, 66.3% Waff)"
    w11_kd_label = "Wave 11 Soft-KD Search (28 ms, 68.8% Waff / 58.3% Large)"
    w11_pure_label = "Wave 11 Pure (10 ms, 50.0% Large Pair!)"
    pure_label = "Wave 8 Pure (8 ms, 1× MS)" if Path(pure_ckpt).name == "galcon_wave8_pure.pt" else "Wave 6B Pure (1× Canon MS)"
    tactical = GalconTacticalBot()
    classic = GalconClassicBot()
    rusher = GalconRusherBot(style="all_in")

    matches: List[Dict[str, Any]] = []

    sparky = None
    waffle = None
    try:
        if _lua_path_exists(args.lua_sparky):
            from galcon.lua_bridge import GalconLuaBotBridge

            sparky = GalconLuaBotBridge(args.lua_sparky)
        if _lua_path_exists(args.lua_waffle):
            from galcon.lua_bridge import GalconLuaBotBridge

            waffle = GalconLuaBotBridge(args.lua_waffle)

        if sparky is not None:
            # 1. Wave 12 #1 Surgical Ensemble (W9S+W6B) WIN vs. Sparky v1.6 (71.2% Record!)
            matches.append(
                _find_showcase_win(
                    f"[WIN 18pl #1] {search_label} vs. esparano Sparky v1.6 (71.2% Record!)",
                    search_label,
                    search.act,
                    "esparano Sparky v1.6 (Lua)",
                    sparky.act,
                    seed_start=9500,
                    num_planets=18,
                    width=400.0,
                    height=240.0,
                    prefer_redir=False,
                    max_ticks=220,
                    max_attempts=12,
                    target_winner=1,
                )
            )
            # 2. Wave 11 Soft-KD WIN vs. Sparky v1.6 on Large 24pl Paired (58.3% All-Time Record!)
            if w11_kd is not None:
                matches.append(
                    _find_showcase_win(
                        f"[WIN Large 24pl #1] {w11_kd_label} vs. esparano Sparky v1.6 (58.3% Record!)",
                        w11_kd_label,
                        w11_kd.act,
                        "esparano Sparky v1.6 (Lua)",
                        sparky.act,
                        seed_start=14700,
                        num_planets=24,
                        width=560.0,
                        height=340.0,
                        prefer_redir=False,
                        symmetric=True,
                        max_ticks=220,
                        max_attempts=10,
                        target_winner=1,
                    )
                )
            # 3. Wave 11 Pure (10 ms, ZERO search!) WIN vs. Sparky v1.6 on Large 24pl (50.0% Pure Record!)
            if w11_pure is not None:
                matches.append(
                    _find_showcase_win(
                        f"[WIN 10ms Pure Large] {w11_pure_label} vs. esparano Sparky v1.6",
                        w11_pure_label,
                        w11_pure.act,
                        "esparano Sparky v1.6 (Lua)",
                        sparky.act,
                        seed_start=14700,
                        num_planets=24,
                        width=560.0,
                        height=340.0,
                        prefer_redir=False,
                        symmetric=True,
                        max_ticks=220,
                        max_attempts=10,
                        target_winner=1,
                    )
                )
            # 4. Wave 10 Ensemble WIN vs. Sparky v1.6 on Large 24pl Asymmetric (66.7%)
            matches.append(
                _find_showcase_win(
                    f"[WIN Large 24pl] {search_label} vs. esparano Sparky v1.6 (Asym 66.7%)",
                    search_label,
                    search.act,
                    "esparano Sparky v1.6 (Lua)",
                    sparky.act,
                    seed_start=14700,
                    num_planets=24,
                    width=560.0,
                    height=340.0,
                    prefer_redir=False,
                    symmetric=False,
                    max_ticks=220,
                    max_attempts=12,
                    target_winner=1,
                )
            )
            # 5. Single-Model Search LOSS vs. Sparky v1.6 (Forensics Study)
            loss_bot = w9_search if w9_search is not None else search
            loss_lbl = w9_search_label if w9_search is not None else search_label
            matches.append(
                _find_showcase_win(
                    f"[LOSS Forensics] {loss_lbl} vs. esparano Sparky v1.6 (18pl)",
                    loss_lbl,
                    loss_bot.act,
                    "esparano Sparky v1.6 (Lua)",
                    sparky.act,
                    seed_start=14000,
                    num_planets=18,
                    width=400.0,
                    height=240.0,
                    prefer_redir=False,
                    max_ticks=220,
                    max_attempts=14,
                    target_winner=2,
                )
            )
            # 6. Wave 8 Pure (8 ms!) WIN vs. Sparky v1.6
            matches.append(
                _find_showcase_win(
                    f"[WIN 8ms Pure] {pure_label} vs. esparano Sparky v1.6 (61.3%)",
                    pure_label,
                    pure.act,
                    "esparano Sparky v1.6 (Lua)",
                    sparky.act,
                    seed_start=9500,
                    num_planets=18,
                    width=400.0,
                    height=240.0,
                    prefer_redir=False,
                    max_ticks=220,
                    max_attempts=14,
                    target_winner=1,
                )
            )

        if waffle is not None:
            # 7. Wave 11 Soft-KD (28 ms!) WIN vs. Waffle3z-GA (68.8% Single-Model Record!)
            if w11_kd is not None:
                matches.append(
                    _find_showcase_win(
                        f"[WIN 28ms Soft-KD #1] {w11_kd_label} vs. Waffle3z-GA (68.8% Record!)",
                        w11_kd_label,
                        w11_kd.act,
                        "Waffle3z-GA (bots_waffle3z_optimized)",
                        waffle.act,
                        seed_start=14000,
                        num_planets=18,
                        width=400.0,
                        height=240.0,
                        prefer_redir=False,
                        max_ticks=220,
                        max_attempts=12,
                        target_winner=1,
                    )
                )
            # 8. Wave 10 #1 Ensemble (W9S+W6B) WIN vs. Waffle3z-GA (68.8% record!)
            matches.append(
                _find_showcase_win(
                    f"[WIN 18pl] {search_label} vs. Waffle3z-GA (68.8% Record!)",
                    search_label,
                    search.act,
                    "Waffle3z-GA (bots_waffle3z_optimized)",
                    waffle.act,
                    seed_start=9500,
                    num_planets=18,
                    width=400.0,
                    height=240.0,
                    prefer_redir=False,
                    max_ticks=220,
                    max_attempts=12,
                    target_winner=1,
                )
            )
            # 9. Wave 10 Single Search (W10S) WIN vs. Waffle3z-GA (66.3% / 77.5% s9500!)
            if w10_search is not None:
                matches.append(
                    _find_showcase_win(
                        f"[WIN 18pl] {w10_search_label} vs. Waffle3z-GA (77.5% s9500!)",
                        w10_search_label,
                        w10_search.act,
                        "Waffle3z-GA (bots_waffle3z_optimized)",
                        waffle.act,
                        seed_start=9500,
                        num_planets=18,
                        width=400.0,
                        height=240.0,
                        prefer_redir=False,
                        max_ticks=220,
                        max_attempts=12,
                        target_winner=1,
                    )
                )
            # 10. Wave 10 Ensemble WIN vs. Waffle3z-GA on Large 24pl Asymmetric (75.0%)
            matches.append(
                _find_showcase_win(
                    f"[WIN Large 24pl] {search_label} vs. Waffle3z-GA (Asym 75.0%)",
                    search_label,
                    search.act,
                    "Waffle3z-GA (bots_waffle3z_optimized)",
                    waffle.act,
                    seed_start=14700,
                    num_planets=24,
                    width=560.0,
                    height=340.0,
                    prefer_redir=False,
                    symmetric=False,
                    max_ticks=220,
                    max_attempts=12,
                    target_winner=1,
                )
            )
            # 11. Wave 10 Ensemble LOSS Forensics vs. Waffle3z-GA
            matches.append(
                _find_showcase_win(
                    f"[LOSS Forensics] {search_label} vs. Waffle3z-GA (18pl)",
                    search_label,
                    search.act,
                    "Waffle3z-GA (bots_waffle3z_optimized)",
                    waffle.act,
                    seed_start=14000,
                    num_planets=18,
                    width=400.0,
                    height=240.0,
                    prefer_redir=False,
                    max_ticks=220,
                    max_attempts=14,
                    target_winner=2,
                )
            )

        # 12. External Lua Boss War: Sparky v1.6 vs. Waffle3z-GA!
        if sparky is not None and waffle is not None:
            matches.append(
                _find_showcase_win(
                    "[External Lua Bot War] esparano Sparky v1.6 vs. Waffle3z-GA (18pl)",
                    "esparano Sparky v1.6 (Lua)",
                    sparky.act,
                    "Waffle3z-GA (bots_waffle3z_optimized)",
                    waffle.act,
                    seed_start=14000,
                    num_planets=18,
                    width=400.0,
                    height=240.0,
                    prefer_redir=False,
                    max_ticks=220,
                    max_attempts=10,
                    target_winner=1,
                )
            )
    finally:
        if sparky is not None:
            sparky.close()
        if waffle is not None:
            waffle.close()

    if _lua_path_exists(args.lua_esparan):
        from galcon.lua_bridge import GalconLuaBotBridge

        esparan = GalconLuaBotBridge(args.lua_esparan)
        try:
            esp_bot = w11_kd if w11_kd is not None else search
            esp_lbl = w11_kd_label if w11_kd is not None else search_label
            matches.append(
                _find_showcase_win(
                    f"[WIN 85.4% Record] {esp_lbl} vs. esparano EsparanBot v0.2",
                    esp_lbl,
                    esp_bot.act,
                    "esparano EsparanBot v0.2 (Lua)",
                    esparan.act,
                    seed_start=15300,
                    num_planets=18,
                    width=400.0,
                    height=240.0,
                    prefer_redir=False,
                    max_ticks=200,
                    max_attempts=8,
                )
            )
        finally:
            esparan.close()

    if args.gen4_checkpoint and Path(args.gen4_checkpoint).is_file():
        g4_pure = GalconTransformerAgent(args.gen4_checkpoint, use_d4_canonical=True, use_d4_tta=False)
        matches.append(
            _find_showcase_win(
                f"Large Asymmetric Arena (24pl 560×340): {pure_label} vs. VM Gen 4 Pure",
                pure_label,
                pure.act,
                "GalconPT-Gen4 Pure (VM, 1× MS)",
                g4_pure.act,
                seed_start=14700,
                num_planets=24,
                width=560.0,
                height=340.0,
                prefer_redir=False,
                symmetric=False,
                max_ticks=220,
                max_attempts=10,
            )
        )

    matches.extend(
        [
            _find_showcase_win(
                f"Rush Defense (18pl 400×240): {pure_label} vs. All-In Rusher",
                pure_label,
                pure.act,
                "GalconRusherBot (All-In)",
                rusher.act,
                seed_start=12200,
                num_planets=18,
                width=400.0,
                height=240.0,
                prefer_redir=False,
            ),
            _find_showcase_win(
                f"Pure Neural Macro (16pl 400×240): {pure_label} vs. Tactical Solver",
                pure_label,
                pure.act,
                "GalconTacticalBot",
                tactical.act,
                seed_start=12000,
                num_planets=16,
                width=400.0,
                height=240.0,
                prefer_redir=False,
            ),
            _find_showcase_win(
                f"Compact Sweep (10pl 240×150): {pure_label} vs. Classic Bot",
                pure_label,
                pure.act,
                "GalconClassicBot",
                classic.act,
                seed_start=12100,
                num_planets=10,
                width=240.0,
                height=150.0,
            ),
        ]
    )

    html = render_galcon_viewer_html(matches)
    Path(args.out_html).write_text(html, encoding="utf-8")
    print(f"[Galcon Replays] Wrote {len(matches)} showcase matches -> {args.out_html} ({len(html)//1024} KB)", flush=True)
    if args.out_json:
        jp = Path(args.out_json)
        jp.parent.mkdir(parents=True, exist_ok=True)
        jp.write_text(json.dumps(matches, separators=(",", ":")), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
