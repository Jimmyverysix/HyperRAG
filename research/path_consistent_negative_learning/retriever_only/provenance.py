"""Minimal run provenance for formal Retriever-only artifacts."""

from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Sequence

import torch


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments],
        cwd=repository,
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()


def collect_provenance(
    repository: Path,
    command: Sequence[str],
) -> dict[str, Any]:
    try:
        commit = _git(repository, "rev-parse", "HEAD")
        dirty = bool(_git(repository, "status", "--porcelain"))
    except (OSError, subprocess.CalledProcessError):
        commit = "unavailable"
        dirty = None
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "research_commit": commit,
        "research_worktree_dirty": dirty,
        "official_upstream_commit": "6d5a9033353c516a9220d78591f2c666f19ee0b1",
        "command": list(command),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
