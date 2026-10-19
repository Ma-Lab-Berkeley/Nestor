"""Repository snapshot utilities."""

from __future__ import annotations

import os
import zipfile
from pathlib import Path

# Directories and patterns to exclude from snapshots
DEFAULT_EXCLUDES = [
    ".git",
    ".cursor",
    "__pycache__",
    ".venv",
    "venv",
    ".mypy_cache",
    ".ruff_cache",
    ".pytest_cache",
    "*.pyc",
    "*.pyo",
    "*.egg-info",
    "data",
    "logs",
    "tmp",
    "outputs",
    "checkpoints",
    "wandb",
    "*.pt",
    "*.pth",
    "*.ckpt",
    "*.zip",
    "*.tar",
    "*.tar.gz",
    "*.h5",
    "*.hdf5",
    ".env",
    ".DS_Store",
]


def _should_exclude(path: Path, excludes: list[str]) -> bool:
    """
    Check if a path should be excluded from the snapshot.

    Args:
        path (Path): Path to check.
        excludes (list[str]): List of patterns to exclude.

    Returns:
        bool: True if the path should be excluded.
    """
    for exclude in excludes:
        if exclude.startswith("*"):
            # Glob pattern
            if path.match(exclude):
                return True
        else:
            # Exact name match (for directories and files)
            if path.name == exclude:
                return True
            # Check if any parent directory matches
            for parent in path.parents:
                if parent.name == exclude:
                    return True
    return False


def create_source_snapshot(
    source_dir: Path,
    output_path: Path,
    excludes: list[str] | None = None,
) -> Path:
    """
    Create a zip snapshot of the source code.

    Args:
        source_dir (Path): Root directory to snapshot.
        output_path (Path): Path for the output zip file.
        excludes (list[str] | None): Patterns to exclude. Defaults to DEFAULT_EXCLUDES.

    Returns:
        Path: Path to the created zip file.
    """
    if excludes is None:
        excludes = DEFAULT_EXCLUDES

    dir_excludes = {e for e in excludes if not e.startswith("*")}

    output_path = output_path.with_suffix(".zip")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for dirpath, dirnames, filenames in os.walk(source_dir):
            dirnames[:] = [d for d in dirnames if d not in dir_excludes]
            for fname in filenames:
                file_path = Path(dirpath) / fname
                try:
                    rel_path = file_path.relative_to(source_dir)
                    if not _should_exclude(rel_path, excludes):
                        zf.write(file_path, rel_path)
                except OSError:
                    pass

    return output_path
