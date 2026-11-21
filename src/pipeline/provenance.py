"""What produced a run: code version, input data hashes, environment."""
from __future__ import annotations

import datetime as dt
import hashlib
import platform
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGES = ("torch", "transformers", "peft", "mlx", "mlx-lm", "mlx-metal", "tokenizers")


def file_sha256(path) -> str | None:
    path = Path(path)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rel(path) -> str:
    """Repo-relative path when inside the repository (portable across clones), else the path as given."""
    path = Path(path)
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def files_sha256(directory, names) -> dict:
    """Hashes of the named files in `directory`; a missing file is recorded as null (e.g. on another machine)."""
    return {name: file_sha256(Path(directory) / name) for name in names}


def _git(*args, strip=True) -> str | None:
    try:
        out = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout
        return out.strip() if strip else out
    except (OSError, subprocess.CalledProcessError):
        return None


def code_version() -> dict:
    status = _git("status", "--porcelain", strip=False)  # each line is "XY path"; keep the leading space
    files = [line[3:] for line in (status or "").splitlines() if line.strip()]
    return {"commit": _git("rev-parse", "HEAD"), "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": None if status is None else bool(files), "dirty_files": files or None}


def environment() -> dict:
    packages = {}
    for name in PACKAGES:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            pass
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": packages}


def run_info(argv=None) -> dict:
    return {"started": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "command": " ".join(argv if argv is not None else sys.argv),
            "code": code_version(), "environment": environment()}
