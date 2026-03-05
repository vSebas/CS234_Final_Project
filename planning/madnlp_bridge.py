"""
Julia bridge for MadNLP + ExaModels trajectory optimization backend.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict


def _default_julia_entrypoint() -> Path:
    return Path(__file__).resolve().parent / "julia" / "madnlp_exa_solver.jl"


def _default_julia_project(julia_bin: str) -> Path | None:
    # Prefer the DT_trajopt conda environment project when available.
    try:
        jb = Path(julia_bin).resolve()
    except Exception:
        return None
    env_root = jb.parent.parent
    candidate = env_root / "share" / "julia" / "environments" / "DT_trajopt"
    if candidate.exists():
        return candidate
    return None


def solve_with_julia_madnlp(payload: Dict[str, Any], timeout_s: int = 1800) -> Dict[str, Any]:
    julia_bin = os.environ.get("JULIA_BIN", "julia")
    if shutil.which(julia_bin) is None:
        raise FileNotFoundError(
            f"Julia binary not found: {julia_bin!r}. "
            "Install Julia and/or set JULIA_BIN to an absolute path."
        )

    entrypoint = Path(os.environ.get("MADNLP_EXA_SCRIPT", str(_default_julia_entrypoint())))
    if not entrypoint.exists():
        raise FileNotFoundError(
            f"MadNLP/ExaModels Julia script not found: {entrypoint}. "
            "Set MADNLP_EXA_SCRIPT to a valid path."
        )
    project_override = os.environ.get("MADNLP_EXA_PROJECT", "").strip()
    project_dir = Path(project_override) if project_override else _default_julia_project(julia_bin)

    with tempfile.TemporaryDirectory(prefix="madnlp_exa_") as tmpdir:
        tmp = Path(tmpdir)
        req = tmp / "request.json"
        res = tmp / "response.json"
        with open(req, "w", encoding="utf-8") as f:
            json.dump(payload, f)

        cmd = [
            julia_bin,
        ]
        if project_dir is not None:
            cmd.append("--project=" + str(project_dir))
        cmd.extend(
            [
            str(entrypoint),
            "--request",
            str(req),
            "--response",
            str(res),
            ]
        )
        stream_logs = os.environ.get("MADNLP_EXA_STREAM", "0").strip() == "1"
        try:
            if stream_logs:
                proc = subprocess.run(
                    cmd,
                    timeout=max(1, int(timeout_s)),
                    check=False,
                )
            else:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=max(1, int(timeout_s)),
                    check=False,
                )
        except subprocess.TimeoutExpired as e:
            out = e.stdout if isinstance(e.stdout, str) else ""
            err = e.stderr if isinstance(e.stderr, str) else ""
            raise RuntimeError(
                "Julia MadNLP backend timed out.\n"
                f"Command: {' '.join(cmd)}\n"
                f"Timeout: {timeout_s}s\n"
                f"Partial STDOUT:\n{out}\n"
                f"Partial STDERR:\n{err}\n"
            ) from e
        if proc.returncode != 0:
            if stream_logs:
                raise RuntimeError(
                    "Julia MadNLP backend failed.\n"
                    f"Command: {' '.join(cmd)}\n"
                    f"Return code: {proc.returncode}\n"
                )
            raise RuntimeError(
                "Julia MadNLP backend failed.\n"
                f"Command: {' '.join(cmd)}\n"
                f"Return code: {proc.returncode}\n"
                f"STDOUT:\n{proc.stdout}\n"
                f"STDERR:\n{proc.stderr}\n"
            )

        if not res.exists():
            raise RuntimeError(
                "Julia MadNLP backend did not produce a response file.\n"
                f"STDOUT:\n{proc.stdout}\n"
                f"STDERR:\n{proc.stderr}\n"
            )

        with open(res, "r", encoding="utf-8") as f:
            return json.load(f)
