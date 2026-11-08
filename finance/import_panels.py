"""Verify configured log-return panels and copy them into ignored local data."""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

from bcd_lab.finance.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Directory containing log_returns_N=*.csv")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    planned = []
    for n in (50, 100, 200):
        spec = load_config(root / f"configs/log_panel_N{n}.yaml")
        target = Path(spec["panel"]["path"])
        source = args.source / target.name
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if digest != spec["panel"]["sha256"]:
            raise ValueError(f"Panel does not match the configured fingerprint: {source}")
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            raise FileExistsError(f"Refusing to replace a different panel: {target}")
        planned.append((source, target))
    for source, target in planned:
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copyfile(source, target)
        print(target)


if __name__ == "__main__":
    main()
