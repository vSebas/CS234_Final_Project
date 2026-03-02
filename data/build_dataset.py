#!/usr/bin/env python3
"""
End-to-end dataset build (Fix A + Fix B).
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    print(" ".join(cmd))
    subprocess.check_call(cmd)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build full dataset.")
    parser.add_argument("--map-files", type=str, default="maps/Oval_Track_260m.mat")
    parser.add_argument("--base-laps", type=int, default=6)
    parser.add_argument("--obstacle-laps", type=int, default=8)
    parser.add_argument("--shift-episodes", type=int, default=1000)
    parser.add_argument("--repair-segments", type=int, default=200)
    parser.add_argument("--output-root", type=str, default="data/datasets")
    parser.add_argument("--base-laps-dir", type=str, default="data/base_laps")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--N", type=int, default=120)
    parser.add_argument("--H", type=int, default=50)
    args = parser.parse_args()

    map_files = [p.strip() for p in args.map_files.split(",") if p.strip()]
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    run(
        [
            sys.executable,
            "data/build_base_laps.py",
            "--map-files",
            ",".join(map_files),
            "--base-laps",
            str(args.base_laps),
            "--obstacle-laps",
            str(args.obstacle_laps),
            "--seed",
            str(args.seed),
            "--N",
            str(args.N),
        ]
    )

    for map_file in map_files:
        stem = Path(map_file).stem
        run(
            [
                sys.executable,
                "data/make_shift_episodes.py",
                "--map-file",
                map_file,
                "--base-laps-dir",
                args.base_laps_dir,
                "--output-dir",
                str(output_root / f"{stem}_shifts"),
                "--num-episodes",
                str(args.shift_episodes),
                "--seed",
                str(args.seed),
            ]
        )
        run(
            [
                sys.executable,
                "data/build_repair_segments.py",
                "--map-file",
                map_file,
                "--base-laps-dir",
                args.base_laps_dir,
                "--output-dir",
                str(output_root / f"{stem}_repairs"),
                "--num-segments",
                str(args.repair_segments),
                "--seed",
                str(args.seed),
                "--H",
                str(args.H),
            ]
        )


if __name__ == "__main__":
    main()
