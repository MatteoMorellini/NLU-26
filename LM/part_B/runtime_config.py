"""Runtime environment settings that must be applied before importing torch."""

import os
from pathlib import Path

PART_DIR = Path(__file__).resolve().parent
DEFAULT_CACHE_DIR = PART_DIR / "hf_cache"


def configure_runtime_environment(cache_dir: Path = DEFAULT_CACHE_DIR) -> None:
    """Set process environment defaults used by training scripts."""

    os.environ.setdefault("HF_HOME", str(cache_dir))
    os.environ.setdefault("HF_HUB_CACHE", str(cache_dir / "hub"))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


configure_runtime_environment()
