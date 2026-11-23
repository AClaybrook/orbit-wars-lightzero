#!/usr/bin/env python3
"""Download open-source Orbit Wars benchmark bots into ``external/opponents/``.

All downloaded files live inside ``external/`` (ignored by ``.gitignore``) so
you can benchmark local agents against stronger community bots without adding
third-party code to this repository.
"""

from __future__ import annotations

import subprocess
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OPP_DIR = REPO_ROOT / "external" / "opponents"


def main() -> int:
    OPP_DIR.mkdir(parents=True, exist_ok=True)

    # 1. kvatsa5/orbit-wars-agent (Hugging Face single-file submission)
    kv_path = OPP_DIR / "kvatsa5_agent.py"
    if not kv_path.exists():
        url = "https://huggingface.co/kvatsa5/orbit-wars-agent/resolve/main/submission.py"
        print(f"Downloading {url} -> {kv_path}")
        urllib.request.urlretrieve(url, kv_path)
    else:
        print(f"Already present: {kv_path}")

    # 2. wanderduck/Orbit_Wars (GitHub: includes bestv6, peer, sniper, swarm, turtle, heuristic)
    wd_dir = OPP_DIR / "wanderduck_orbit_wars"
    if not wd_dir.exists():
        print(f"Cloning https://github.com/wanderduck/Orbit_Wars.git -> {wd_dir}")
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/wanderduck/Orbit_Wars.git", str(wd_dir)],
            check=True,
        )
    else:
        print(f"Already present: {wd_dir}")

    print("\nReady! Run benchmarks with e.g.:")
    print("  python scripts/evaluate.py --candidate heuristic --opponents kvatsa5 wd_bestv6 wd_peer wd_sniper wd_swarm wd_turtle --games 6")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
