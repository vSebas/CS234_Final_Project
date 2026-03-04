#!/usr/bin/env python3
"""
End-to-end dataset build (Fix A + Fix B).
"""

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed


def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str, progress_file: Path | None = None) -> None:
    line = f"[{_timestamp()}] {msg}"
    print(line, flush=True)
    if progress_file is not None:
        progress_file.parent.mkdir(parents=True, exist_ok=True)
        with progress_file.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def run(cmd: list[str], progress_file: Path | None = None) -> None:
    log("RUN: " + " ".join(cmd), progress_file)
    subprocess.check_call(cmd)


def _manifest_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build full dataset.")
    parser.add_argument("--map-files", type=str, default="maps/Oval_Track_260m.mat")
    parser.add_argument("--base-laps", type=int, default=6)
    parser.add_argument("--obstacle-laps", type=int, default=8)
    parser.add_argument("--shift-episodes", type=int, default=1000)
    parser.add_argument(
        "--all-shifts",
        action="store_true",
        help="Generate all unique shifts per base lap (k0=0..N).",
    )
    parser.add_argument("--repair-segments", type=int, default=200)
    parser.add_argument("--hard-repair-segments", type=int, default=0)
    parser.add_argument("--output-root", type=str, default="data/datasets")
    parser.add_argument("--base-laps-dir", type=str, default="data/base_laps")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--N", type=int, default=200)
    parser.add_argument("--H", type=int, default=20)
    parser.add_argument("--hard-repair-hotspot-json", type=str, default=None)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume by reusing completed work and appending missing outputs.",
    )
    parser.add_argument(
        "--parallel",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run per-track shift/repair stages in parallel.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Max parallel workers when --parallel is set.",
    )
    args = parser.parse_args()

    map_files = [p.strip() for p in args.map_files.split(",") if p.strip()]
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    progress_file = output_root / "progress.log"
    log("Starting dataset build.", progress_file)

    expected_base = int(args.base_laps) + int(args.obstacle_laps)
    base_ready = True
    for map_file in map_files:
        stem = Path(map_file).stem
        manifest = Path(args.base_laps_dir) / stem / "manifest.jsonl"
        if _manifest_lines(manifest) < expected_base:
            base_ready = False
            break

    if args.resume and base_ready:
        log("Stage A1: base laps already present; skipping.", progress_file)
    else:
        log("Stage A1: build base laps", progress_file)
        run(
            [
                sys.executable,
                "-u",
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
                "--resume",
            ],
            progress_file,
        )
        log("Stage A1 complete.", progress_file)

    base_rep = args.repair_segments // max(1, len(map_files))
    rem = args.repair_segments - base_rep * max(1, len(map_files))
    base_hard_rep = args.hard_repair_segments // max(1, len(map_files))
    hard_rem = args.hard_repair_segments - base_hard_rep * max(1, len(map_files))

    def _process_track(idx: int, map_file: str) -> None:
        stem = Path(map_file).stem
        shifts_dir = output_root / f"{stem}_shifts"
        shifts_manifest = shifts_dir / "manifest.jsonl"
        base_dir = Path(args.base_laps_dir) / stem
        base_count = len(list(base_dir.glob("*.npz"))) if base_dir.exists() else 0
        expected_shifts = (args.N + 1) * base_count if args.all_shifts else args.shift_episodes
        shifts_done = _manifest_lines(shifts_manifest)
        if args.resume and shifts_done >= expected_shifts and expected_shifts > 0:
            log(f"Stage A2: shifts already present for {stem}; skipping.", progress_file)
        else:
            log(f"Stage A2: shifts for {stem}", progress_file)
            shift_cmd = [
                sys.executable,
                "-u",
                "data/make_shift_episodes.py",
                "--map-file",
                map_file,
                "--base-laps-dir",
                args.base_laps_dir,
                "--output-dir",
                str(shifts_dir),
                "--seed",
                str(args.seed),
            ]
            if args.all_shifts:
                shift_cmd.append("--all-shifts")
            else:
                shift_cmd.extend(["--num-episodes", str(args.shift_episodes)])
            shift_cmd.append("--resume")
            run(shift_cmd, progress_file)
            log(f"Stage A2 complete for {stem}.", progress_file)

        nseg = base_rep + (1 if idx < rem else 0)
        repairs_dir = output_root / f"{stem}_repairs"
        repairs_manifest = repairs_dir / "manifest.jsonl"
        repairs_done = _manifest_lines(repairs_manifest)
        if args.resume and repairs_done >= nseg and nseg > 0:
            log(f"Stage B: target accepted repairs already present for {stem}; skipping.", progress_file)
        else:
            log(f"Stage B: target accepted repairs for {stem} (target={nseg})", progress_file)
            run(
                [
                    sys.executable,
                    "-u",
                    "data/build_repair_segments.py",
                    "--map-file",
                    map_file,
                    "--base-laps-dir",
                    args.base_laps_dir,
                    "--output-dir",
                    str(repairs_dir),
                    "--num-segments",
                    str(nseg),
                    "--seed",
                    str(args.seed),
                    "--H",
                    str(args.H),
                    "--save-every",
                    "10",
                    "--resume",
                ],
                progress_file,
            )
            repairs_done_after = _manifest_lines(repairs_manifest)
            log(
                f"Stage B complete for {stem}. accepted={repairs_done_after}/{nseg}",
                progress_file,
            )

        nhard = base_hard_rep + (1 if idx < hard_rem else 0)
        hard_repairs_dir = output_root / f"{stem}_repairs_hard"
        hard_repairs_manifest = hard_repairs_dir / "manifest.jsonl"
        hard_repairs_done = _manifest_lines(hard_repairs_manifest)
        if nhard <= 0:
            return
        if args.resume and hard_repairs_done >= nhard:
            log(f"Stage B2: target accepted hard repairs already present for {stem}; skipping.", progress_file)
        else:
            log(f"Stage B2: target accepted hard repairs for {stem} (target={nhard})", progress_file)
            hard_cmd = [
                sys.executable,
                "-u",
                "data/build_repair_segments.py",
                "--map-file",
                map_file,
                "--base-laps-dir",
                args.base_laps_dir,
                "--output-dir",
                str(hard_repairs_dir),
                "--num-segments",
                str(nhard),
                "--seed",
                str(args.seed),
                "--H",
                str(args.H),
                "--save-every",
                "10",
                "--resume",
                "--hard-mode",
                "--uy-perturb-mps",
                "0.15",
                "--r-perturb-radps",
                "0.08",
            ]
            if args.hard_repair_hotspot_json:
                hard_cmd.extend(["--hotspot-json", args.hard_repair_hotspot_json])
            run(hard_cmd, progress_file)
            hard_repairs_done_after = _manifest_lines(hard_repairs_manifest)
            log(
                f"Stage B2 complete for {stem}. accepted={hard_repairs_done_after}/{nhard}",
                progress_file,
            )

    if args.parallel:
        max_workers = min(args.max_workers, len(map_files))
        log(f"Running per-track stages in parallel (workers={max_workers}).", progress_file)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_process_track, idx, mf) for idx, mf in enumerate(map_files)]
            for fut in as_completed(futures):
                fut.result()
    else:
        for idx, map_file in enumerate(map_files):
            _process_track(idx, map_file)

    log("Dataset build finished.", progress_file)


if __name__ == "__main__":
    main()
