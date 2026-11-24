"""Drop-in Phil Hassey ``gbots pipe`` wire-protocol client for our Galcon bots.

Usage with Phil Hassey's official ``gbots`` CLI:
    ./gbots pipe -exec "python3 -m galcon.gbots_client --bot transformer --checkpoint checkpoints/galcon_transformer_best.pt"
    ./gbots pipe -exec "python3 -m galcon.gbots_client --bot tactical"
    ./gbots pipe -exec "python3 -m galcon.gbots_client --bot search --checkpoint checkpoints/galcon_transformer_best.pt"

Speaks the exact tab-delimited stdin/stdout protocol documented in ``gbots/protocol.md``:
- Incoming: ``/SET``, ``/RESET``, ``/USER``, ``/PLANET``, ``/FLEET``, ``/DESTROY``,
  field-sync rows (``XYSR...``), ``/TICK``
- Outgoing: ``/SEND PERCENT SOURCE TARGET``, ``/REDIR FLEET_ID TARGET``, ``/TOCK``
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, TextIO

from galcon.heuristic_bot import GalconClassicBot, GalconTacticalBot
from galcon.sim import DEFAULT_FLEET_SPEED, DEFAULT_TICK_DT, Fleet, GalconSim, Order, Planet, RedirOrder, SendOrder


class GbotsProtocolState:
    """Maintains live galaxy state synchronized from Phil Hassey's ``gbots`` wire stream."""

    def __init__(self, *, fleet_speed: float = DEFAULT_FLEET_SPEED, tick_dt: float = DEFAULT_TICK_DT) -> None:
        self.fleet_speed = float(fleet_speed)
        self.tick_dt = float(tick_dt)
        self.reset()

    def reset(self) -> None:
        self.state: str = ""
        self.you: int = 0
        self.speed: float = 1.0
        self.tick: int = 0
        self.users: Dict[int, Dict[str, object]] = {}
        self.planets: Dict[int, Planet] = {}
        self.fleets: Dict[int, Fleet] = {}

    def feed_line(self, raw_line: str) -> bool:
        """Process one wire-protocol line. Returns ``True`` if ``/TICK`` requests bot actions."""
        line = raw_line.rstrip("\r\n")
        if not line:
            return False
        parts = line.split("\t")
        cmd = parts[0]
        if not cmd.startswith("/"):
            self._apply_sync(parts)
            return False

        if cmd == "/TICK":
            if len(parts) >= 2:
                try:
                    dur = float(parts[1])
                    if dur > 0:
                        self.tick_dt = dur
                except ValueError:
                    pass
            self.tick += 1
            return True
        if cmd == "/RESET":
            self.reset()
        elif cmd == "/SET" and len(parts) >= 3:
            key, val = parts[1].upper(), parts[2]
            if key == "YOU":
                self.you = int(val)
            elif key == "STATE":
                self.state = val
            elif key == "SPEED":
                try:
                    self.speed = float(val)
                except ValueError:
                    pass
        elif cmd == "/USER" and len(parts) >= 6:
            uid = int(parts[1])
            self.users[uid] = {
                "id": uid,
                "name": parts[2],
                "color": parts[3],
                "team": int(parts[4]),
                "xid": int(parts[5]),
            }
        elif cmd == "/PLANET" and len(parts) >= 8:
            pid = int(parts[1])
            self.planets[pid] = Planet(
                id=pid,
                owner=int(parts[2]),
                ships=float(parts[3]),
                x=float(parts[4]),
                y=float(parts[5]),
                production=float(parts[6]),
                radius=float(parts[7]),
            )
        elif cmd == "/FLEET" and len(parts) >= 9:
            fid = int(parts[1])
            self.fleets[fid] = Fleet(
                id=fid,
                owner=int(parts[2]),
                ships=float(parts[3]),
                x=float(parts[4]),
                y=float(parts[5]),
                source=int(parts[6]),
                target=int(parts[7]),
                radius=float(parts[8]),
            )
        elif cmd == "/DESTROY" and len(parts) >= 2:
            oid = int(parts[1])
            self.planets.pop(oid, None)
            self.fleets.pop(oid, None)
        return False

    def _apply_sync(self, parts: List[str]) -> None:
        fields = parts[0].upper()
        n_fields = len(fields)
        i = 1
        while i + n_fields < len(parts) + 1:
            oid = int(parts[i])
            i += 1
            obj = self.planets.get(oid) or self.fleets.get(oid)
            if obj is None:
                i += n_fields
                continue
            for k in fields:
                v = parts[i]
                i += 1
                if k == "X":
                    obj.x = float(v)
                elif k == "Y":
                    obj.y = float(v)
                elif k == "S":
                    obj.ships = float(v)
                elif k == "R":
                    obj.radius = float(v)
                elif k == "O":
                    obj.owner = int(float(v))
                elif k == "T" and isinstance(obj, Fleet):
                    obj.target = int(float(v))

    def to_sim(self) -> GalconSim:
        """Convert live ``gbots`` state into a ``GalconSim`` snapshot (mapping ``you -> 1``, enemies -> 2)."""
        # Map gbots user IDs cleanly: 0 stays neutral, self.you -> 1, any other owner -> 2
        def map_owner(raw_owner: int) -> int:
            if raw_owner == 0:
                return 0
            return 1 if raw_owner == self.you else 2

        planets = [
            Planet(p.id, map_owner(p.owner), p.ships, p.x, p.y, p.production, p.radius)
            for p in self.planets.values()
        ]
        fleets = [
            Fleet(f.id, map_owner(f.owner), f.ships, f.x, f.y, f.source, f.target, f.radius)
            for f in self.fleets.values()
        ]
        if planets:
            max_x = max(p.x + p.radius for p in planets)
            max_y = max(p.y + p.radius for p in planets)
            width = max(200.0, float(max_x) + 10.0)
            height = max(125.0, float(max_y) + 10.0)
        else:
            width, height = 400.0, 240.0
        return GalconSim(
            planets,
            fleets,
            width=width,
            height=height,
            fleet_speed=self.fleet_speed,
            tick_dt=self.tick_dt,
            tick=self.tick,
        )


def format_gbots_orders(orders: List[Order]) -> List[str]:
    """Encode ``SendOrder`` / ``RedirOrder`` objects into official ``gbots`` protocol commands."""
    lines: List[str] = []
    for o in orders:
        if isinstance(o, SendOrder):
            pct = max(1, min(100, int(o.percent)))
            lines.append(f"/SEND {pct} {int(o.src_id)} {int(o.dst_id)}")
        elif isinstance(o, RedirOrder):
            lines.append(f"/REDIR {int(o.fleet_id)} {int(o.dst_id)}")
    return lines


def run_gbots_loop(
    act_fn: Callable[[GalconSim, int], List[Order]],
    *,
    in_stream: Optional[TextIO] = None,
    out_stream: Optional[TextIO] = None,
) -> int:
    """Run the blocking stdin/stdout ``gbots pipe`` event loop."""
    fin = in_stream if in_stream is not None else sys.stdin
    fout = out_stream if out_stream is not None else sys.stdout
    proto = GbotsProtocolState()
    ticks_handled = 0

    for line in fin:
        if proto.feed_line(line):
            ticks_handled += 1
            if proto.you > 0 and proto.planets:
                sim = proto.to_sim()
                orders = act_fn(sim, 1)
                for cmd in format_gbots_orders(orders):
                    fout.write(cmd + "\n")
            fout.write("/TOCK\n")
            fout.flush()
    return ticks_handled


def _build_bot(
    bot_kind: str, checkpoint: Optional[str], checkpoint2: Optional[str] = None
) -> Callable[[GalconSim, int], List[Order]]:
    if bot_kind == "classic":
        return GalconClassicBot().act
    if bot_kind == "tactical":
        return GalconTacticalBot().act
    if bot_kind == "rusher":
        from galcon.heuristic_bot import GalconRusherBot

        return GalconRusherBot().act
    if bot_kind == "macro":
        from galcon.heuristic_bot import GalconMacroBot

        return GalconMacroBot().act
    if bot_kind in ("transformer", "search", "ensemble"):
        from galcon.transformer_bot import (
            EnsembleGalconSearchAgent,
            GalconSearchTransformerAgent,
            GalconTransformerAgent,
        )

        ckpt = checkpoint or "checkpoints/galcon_transformer_best.pt"
        if not Path(ckpt).is_file():
            raise FileNotFoundError(f"Galcon checkpoint not found: {ckpt}")
        if bot_kind == "ensemble":
            ckpt2 = checkpoint2 or "checkpoints/galcon_wave7_p1.pt"
            if not Path(ckpt2).is_file():
                raise FileNotFoundError(f"Galcon ensemble secondary checkpoint not found: {ckpt2}")
            return EnsembleGalconSearchAgent([ckpt, ckpt2], use_d4_canonical=True, use_d4_tta=False).act
        if bot_kind == "search":
            return GalconSearchTransformerAgent(ckpt, use_d4_canonical=True, use_d4_tta=False).act
        return GalconTransformerAgent(ckpt, use_d4_canonical=True, use_d4_tta=False).act
    raise ValueError(f"Unknown --bot kind: {bot_kind}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run our Galcon bot over Phil Hassey's gbots pipe protocol.")
    parser.add_argument(
        "--bot",
        choices=["tactical", "classic", "rusher", "macro", "transformer", "search", "ensemble"],
        default="transformer",
    )
    parser.add_argument("--checkpoint", default="checkpoints/galcon_transformer_best.pt")
    parser.add_argument("--checkpoint2", default="checkpoints/galcon_wave7_p1.pt")
    args = parser.parse_args(argv)
    act_fn = _build_bot(args.bot, args.checkpoint, args.checkpoint2)
    run_gbots_loop(act_fn)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
