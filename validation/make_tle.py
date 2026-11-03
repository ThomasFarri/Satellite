"""Write the scenario satellite as a TLE for STK (and any other SGP4 tool).

Usage:  python validation/make_tle.py [--config configs/example.yaml]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from opticalpass.config import load_config  # noqa: E402
from opticalpass.orbit import design_mean_elements  # noqa: E402
from opticalpass.propagators import SGP4Propagator  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "example.yaml")
    ap.add_argument("--out", type=Path, default=ROOT / "validation" / "stk" / "satellite.tle")
    args = ap.parse_args()
    cfg = load_config(args.config)
    l1, l2 = SGP4Propagator(design_mean_elements(cfg.orbit, cfg.epoch)).to_tle()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(f"OPTICALPASS SYNTHETIC SAT\n{l1}\n{l2}\n", encoding="ascii")
    print(f"Wrote {args.out}\n{l1}\n{l2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
