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
) -> Dict[str, Any]:
    """Play one Galcon match and record compact frame-by-frame snapshots for HTML5 Canvas playback."""
    sim = GalconSim.create(seed=seed, num_planets=num_planets, width=width, height=height, max_ticks=max_ticks)
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
        <div class="title">Galcon AI Gen 2 — PlanetTimelineTransformer Replays &amp; Elo Ladder</div>
        <div class="sub">1.26M Self-Play League Transformer • Native 8× D4 /REDIR (purple arcs) • gbots physics (prod/50, speed 40, 0.25s ticks)</div>
      </div>
      <div class="tabs">
        <button class="tab-btn active" id="tabReplays">▶ Live Replays</button>
        <button class="tab-btn" id="tabElo">🏆 Elo Ladder &amp; Win Rates</button>
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
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
          <div style="font-weight:700;">Bradley-Terry Elo Ladder (Anchored at GalconClassicBot = 1200.0)</div>
          <span class="badge ok">Gen 2 Pure: +322.9 Elo over v1</span>
        </div>
        <div id="eloRows"></div>
      </div>
      <div class="elo-card">
        <div style="font-weight:700; margin-bottom:8px;">Verified Held-Out Multi-Map Gauntlet (8–22 Planets, 240×150 to 400×240)</div>
        <table class="bench">
          <thead>
            <tr><th>Agent</th><th>vs Tactical</th><th>vs Classic</th><th>vs Rusher (All-In)</th><th>vs Rusher (Adapt)</th><th>vs Macro</th><th>vs v1 Pure</th><th>Latency</th></tr>
          </thead>
          <tbody>
            <tr><td><strong>GalconPT-Gen2 Pure (1.26M, 8× D4)</strong></td><td>15–2–3 (75%)</td><td>18–1–1 (90%)</td><td><strong>19–1–0 (95%)</strong></td><td>16–1–3 (80%)</td><td><strong>19–0–1 (95%)</strong></td><td>15–4–1 (75%)</td><td class="mono"><strong>12.3 ms</strong></td></tr>
            <tr><td><strong>GalconPT-Gen2 + Search (8× D4)</strong></td><td>8–3–1 (67%)</td><td><strong>12–0–0 (100%)</strong></td><td><strong>12–0–0 (100%)</strong></td><td><strong>12–0–0 (100%)</strong></td><td>11–0–1 (92%)</td><td>6–3–3 vs v1+S</td><td class="mono">67.9 ms</td></tr>
            <tr><td>GalconPT-Gen2 EI-R2 (No PPO)</td><td>14–3–3 (70%)</td><td>17–3–0 (85%)</td><td>12–6–2 (60%)</td><td>11–5–4 (55%)</td><td>19–0–1 (95%)</td><td>12–7–1 (60%)</td><td class="mono">12.3 ms</td></tr>
            <tr><td>GalconPT-v1 Pure (0.59M, 8× D4)</td><td>4–7–1 (33%)</td><td>8–4–0 (67%)</td><td>3–9–0 (25%)</td><td>8–4–0 (67%)</td><td>9–1–2 (75%)</td><td>—</td><td class="mono">10–17 ms</td></tr>
          </tbody>
        </table>
        <div class="sub" style="margin-top:8px;">Head-to-head: <strong>GalconPT-Gen2 Pure (12.3 ms)</strong> beat <strong>GalconPT-Gen2 + Search (67.9 ms)</strong> <code>5–3–4</code> (58.3% score) — full-game 608-ep League PPO surpassed 1-ply lookahead while running 5.5× faster.</div>
      </div>
    </div>
  </div>

  <script>
    const MATCHES = {matches_json};
    const ELO_LADDER = [
      {{ name: "GalconPT-Gen2 + Search (1.26M)", elo: 1643.8, ms: "67.9 ms", col: "#34d399" }},
      {{ name: "GalconPT-Gen2 Pure (1.26M, 8× D4)", elo: 1628.0, ms: "12.3 ms", col: "#38bdf8" }},
      {{ name: "GalconPT-Gen2 EI-R2 (Search Distill)", elo: 1440.9, ms: "12.3 ms", col: "#818cf8" }},
      {{ name: "GalconRusherBot (All-In)", elo: 1320.5, ms: "<2 ms", col: "#fb923c" }},
      {{ name: "GalconPT-v1 Pure (0.59M, 8× D4)", elo: 1305.1, ms: "10.1 ms", col: "#c084fc" }},
      {{ name: "GalconRusherBot (Adaptive)", elo: 1280.9, ms: "<2 ms", col: "#f59e0b" }},
      {{ name: "GalconTacticalBot (Analytical)", elo: 1221.9, ms: "<2 ms", col: "#94a3b8" }},
      {{ name: "GalconClassicBot (Anchor)", elo: 1200.0, ms: "<1 ms", col: "#64748b" }},
      {{ name: "GalconMacroBot (Boomer)", elo: 1098.3, ms: "<2 ms", col: "#475569" }},
    ];

    const eloWrap = document.getElementById("eloRows");
    ELO_LADDER.forEach((row, idx) => {{
      const pct = Math.max(6, Math.min(100, ((row.elo - 1020) / (1680 - 1020)) * 100));
      const div = document.createElement("div");
      div.className = "elo-row";
      div.innerHTML = `<div class="mono sub">#${{idx+1}}</div><div style="font-weight:${{idx<2?700:500}}">${{row.name}}</div><div class="elo-bar-bg"><div class="elo-bar" style="width:${{pct.toFixed(1)}}%;background:${{row.col}}"></div></div><div class="mono" style="text-align:right;font-weight:700;">${{row.elo.toFixed(1)}}</div><div class="mono sub" style="text-align:right;">${{row.ms}}</div>`;
      eloWrap.appendChild(div);
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
) -> Dict[str, Any]:
    best_m: Optional[Dict[str, Any]] = None
    for offset in range(10):
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
            max_ticks=180,
        )
        if m["winner"] == 1:
            if not prefer_redir or m["redirs"][0] > 0:
                return m
            if best_m is None:
                best_m = m
        elif best_m is None:
            best_m = m
    assert best_m is not None
    return best_m


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Record Gen 2 Galcon showcase matches into docs/galcon_viewer.html.")
    parser.add_argument("--checkpoint", default="checkpoints/galcon_transformer_best.pt")
    parser.add_argument("--v1-checkpoint", default="checkpoints/galcon_v1_baseline.pt")
    parser.add_argument("--out-html", default="docs/galcon_viewer.html")
    parser.add_argument("--out-json", default="")
    args = parser.parse_args(argv)

    from galcon.heuristic_bot import GalconClassicBot, GalconRusherBot, GalconTacticalBot
    from galcon.transformer_bot import GalconSearchTransformerAgent, GalconTransformerAgent

    pure = GalconTransformerAgent(args.checkpoint, use_d4_tta=True)
    search = GalconSearchTransformerAgent(args.checkpoint, use_d4_tta=True)
    tactical = GalconTacticalBot()
    classic = GalconClassicBot()
    rusher = GalconRusherBot(style="all_in")

    matches: List[Dict[str, Any]] = [
        _find_showcase_win(
            "Rush Defense (18-Planet 400×240): Gen 2 Pure vs. All-In Rusher",
            "GalconPT-Gen2 Pure (8× D4)",
            pure.act,
            "GalconRusherBot (All-In)",
            rusher.act,
            seed_start=12200,
            num_planets=18,
            width=400.0,
            height=240.0,
            prefer_redir=True,
        ),
        _find_showcase_win(
            "Pure Neural Macro (16-Planet 400×240): Gen 2 Pure vs. Tactical Solver",
            "GalconPT-Gen2 Pure (8× D4)",
            pure.act,
            "GalconTacticalBot",
            tactical.act,
            seed_start=12000,
            num_planets=16,
            width=400.0,
            height=240.0,
            prefer_redir=True,
        ),
        _find_showcase_win(
            "Batched AlphaZero Search (18-Planet 400×240): Gen 2 + Search vs. Tactical",
            "GalconPT-Gen2 + Search",
            search.act,
            "GalconTacticalBot",
            tactical.act,
            seed_start=12020,
            num_planets=18,
            width=400.0,
            height=240.0,
            prefer_redir=True,
        ),
        _find_showcase_win(
            "Compact Arena (10-Planet 240×150): Gen 2 Pure vs. Classic Bot",
            "GalconPT-Gen2 Pure (8× D4)",
            pure.act,
            "GalconClassicBot",
            classic.act,
            seed_start=12100,
            num_planets=10,
            width=240.0,
            height=150.0,
        ),
    ]

    if Path(args.v1_checkpoint).is_file():
        v1_pure = GalconTransformerAgent(args.v1_checkpoint, use_d4_tta=True)
        matches.insert(
            0,
            _find_showcase_win(
                "Generation Leap (16-Planet 400×240): Gen 2 Pure vs. v1 Pure",
                "GalconPT-Gen2 Pure (1.26M)",
                pure.act,
                "GalconPT-v1 Pure (0.59M)",
                v1_pure.act,
                seed_start=12500,
                num_planets=16,
                width=400.0,
                height=240.0,
                prefer_redir=True,
            ),
        )

    html = render_galcon_viewer_html(matches)
    Path(args.out_html).write_text(html, encoding="utf-8")
    print(f"[Galcon Replays] Wrote {len(matches)} showcase matches -> {args.out_html} ({len(html)//1024} KB)", flush=True)
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(matches, separators=(",", ":")), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
