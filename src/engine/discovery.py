"""
Cross-Platform Stockfish Binary Discovery and On-Demand Downloader.

Implements multi-tier binary resolution:
1. Explicit custom path or constructor parameter.
2. `STOCKFISH_PATH` environment variable.
3. Project local `./bin/` directory and user cache.
4. System PATH (`shutil.which`).
5. On-demand download from official Stockfish releases.
"""

from __future__ import annotations

import logging
import os
import platform
import shutil
import stat
import tarfile
import urllib.request
import zipfile
from pathlib import Path

from src.engine.exceptions import EngineNotFoundError

logger = logging.getLogger(__name__)

# Official Stockfish latest release URL mappings
STOCKFISH_RELEASE_BASE = "https://github.com/official-stockfish/Stockfish/releases/latest/download"

# Common binary filenames by platform
COMMON_BIN_NAMES = {
    "Windows": ["stockfish.exe", "stockfish-windows-x86-64-avx2.exe", "stockfish-windows-x86-64-modern.exe", "stockfish-windows-x86-64.exe"],
    "Linux": ["stockfish", "stockfish-ubuntu-x86-64-avx2", "stockfish-ubuntu-x86-64-modern", "stockfish-ubuntu-x86-64"],
    "Darwin": ["stockfish", "stockfish-macos-x86-64-avx2", "stockfish-macos-m1-apple-silicon"],
}


def _get_platform_asset_name() -> tuple[str, str]:
    """
    Determines the asset archive filename and expected format based on OS and architecture.

    Returns:
        tuple[str, str]: (asset_filename, archive_type: 'zip' | 'tar')
    """
    system = platform.system()
    machine = platform.machine().lower()
    is_arm = "arm" in machine or "aarch" in machine

    if system == "Windows":
        return "stockfish-windows-x86-64-avx2.zip", "zip"
    elif system == "Linux":
        return "stockfish-ubuntu-x86-64-avx2.tar", "tar"
    elif system == "Darwin":
        if is_arm:
            return "stockfish-macos-m1-apple-silicon.tar", "tar"
        return "stockfish-macos-x86-64-avx2.tar", "tar"
    else:
        raise EngineNotFoundError(f"Unsupported operating system for automated Stockfish download: {system}")


def discover_stockfish_binary(custom_path: Path | str | None = None) -> Path | None:
    """
    Discovers the location of a local Stockfish executable across multi-tier search paths.

    Search Priority:
    1. `custom_path` argument if provided.
    2. `STOCKFISH_PATH` or `STOCKFISH_BINARY` environment variables.
    3. Project `bin/` directory and search within `./bin/` recursively.
    4. User cache directory `~/.cache/chess_ml/stockfish/`.
    5. System `PATH` via `shutil.which`.

    Args:
        custom_path: Optional explicit path to Stockfish binary.

    Returns:
        Path to the discovered binary, or None if not found.
    """
    # 1. Explicit path - if specified, strictly check this path
    if custom_path is not None:
        p = Path(custom_path).resolve()
        if p.is_file():
            return p
        return None

    # 2. Environment variables
    for env_var in ("STOCKFISH_PATH", "STOCKFISH_BINARY"):
        env_val = os.environ.get(env_var)
        if env_val:
            p = Path(env_val).resolve()
            if p.is_file():
                return p

    # 3. Project ./bin directory
    search_dirs = [
        Path.cwd() / "bin",
        Path(__file__).resolve().parent.parent.parent / "bin",
        Path.home() / ".cache" / "chess_ml" / "stockfish",
        Path.home() / ".local" / "bin",
    ]

    system = platform.system()
    candidate_names = COMMON_BIN_NAMES.get(system, ["stockfish", "stockfish.exe"])

    for s_dir in search_dirs:
        if not s_dir.exists():
            continue

        # Direct name checks
        for name in candidate_names:
            candidate = s_dir / name
            if candidate.is_file():
                return candidate.resolve()

        # Recursive search for any stockfish executable in s_dir
        for item in s_dir.rglob("stockfish*"):
            if item.is_file():
                if system == "Windows" and item.suffix.lower() == ".exe":
                    return item.resolve()
                elif system != "Windows" and item.suffix == "":
                    return item.resolve()

    # 4. System PATH lookup
    for name in candidate_names:
        which_path = shutil.which(name)
        if which_path:
            return Path(which_path).resolve()

    which_generic = shutil.which("stockfish")
    if which_generic:
        return Path(which_generic).resolve()

    return None


def download_stockfish_binary(
    target_dir: Path | str = "bin",
    force_download: bool = False,
) -> Path:
    """
    Downloads and extracts the official Stockfish binary for the host platform.

    Args:
        target_dir: Directory where the binary should be extracted and saved.
        force_download: If True, re-downloads even if binary already exists.

    Returns:
        Path to the extracted Stockfish executable.

    Raises:
        EngineNotFoundError: If download, extraction, or permission setting fails.
    """
    dest_path = Path(target_dir).resolve()
    dest_path.mkdir(parents=True, exist_ok=True)

    if not force_download:
        existing = discover_stockfish_binary(dest_path)
        if existing and existing.is_file():
            logger.info("Using existing Stockfish binary at %s", existing)
            return existing

    asset_name, archive_type = _get_platform_asset_name()
    download_url = f"{STOCKFISH_RELEASE_BASE}/{asset_name}"
    archive_file = dest_path / asset_name

    logger.info("Downloading official Stockfish release from %s -> %s", download_url, archive_file)

    try:
        req = urllib.request.Request(
            download_url,
            headers={"User-Agent": "ChessML-StockfishManager/1.0"},
        )
        with urllib.request.urlopen(req, timeout=30.0) as response, open(archive_file, "wb") as out_file:
            shutil.copyfileobj(response, out_file)
    except Exception as e:
        raise EngineNotFoundError(
            f"Failed to download Stockfish binary from {download_url}: {e}"
        ) from e

    # Extraction
    try:
        if archive_type == "zip":
            with zipfile.ZipFile(archive_file, "r") as zf:
                zf.extractall(dest_path)
        elif archive_type == "tar":
            with tarfile.open(archive_file, "r:*") as tf:
                tf.extractall(dest_path)
    except Exception as e:
        raise EngineNotFoundError(f"Failed to extract Stockfish archive {archive_file}: {e}") from e
    finally:
        if archive_file.exists():
            archive_file.unlink(missing_ok=True)

    # Locate extracted binary
    discovered = discover_stockfish_binary(dest_path)
    if not discovered or not discovered.is_file():
        # Look for any newly extracted executable in dest_path
        candidates = list(dest_path.rglob("stockfish*"))
        for c in candidates:
            if c.is_file() and (platform.system() != "Windows" or c.suffix.lower() == ".exe"):
                discovered = c
                break

    if not discovered:
        raise EngineNotFoundError(f"Could not locate extracted Stockfish binary inside {dest_path}")

    # Set executable permissions on POSIX
    if platform.system() != "Windows":
        current_stat = os.stat(discovered)
        os.chmod(discovered, current_stat.st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    logger.info("Successfully installed Stockfish at %s", discovered)
    return discovered
