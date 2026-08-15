"""
Automated Physical Dataset Ingestion & Downloader Engine.

Provides fully configurable, data-driven downloaders for physical chess datasets
driven by declarative YAML manifests (configs/dataset/physical_sources.yaml):
- ChessReD (10,800 real-world photos via 4TU.ResearchData / Zenodo)
- Roboflow Universe Staunton (Direct HTTP zip endpoint)
- Kaggle Physical Datasets (Direct / Kaggle API)

Includes resumable chunked downloads, checksum verification, atomic file writes,
and Zip Slip extraction security protection.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import ssl
import sys
import urllib.request
import zipfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable

import yaml
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger("chess_ml.dataset")

# Standard HTTP headers to avoid 403 Forbidden on academic/cloud CDNs
DEFAULT_USER_AGENT = "ChessML-Pipeline/0.1.0 (+https://github.com/EwaldoNieuwenhuis/chess_ml)"
DEFAULT_PHYSICAL_CONFIG_PATH = Path("configs/dataset/physical_sources.yaml")
DEFAULT_DIGITAL_CONFIG_PATH = Path("configs/dataset/digital_sources.yaml")
DEFAULT_SOURCES_CONFIG_PATH = DEFAULT_PHYSICAL_CONFIG_PATH


class DatasetDownloadError(Exception):
    """Raised when a dataset download or verification operation fails."""


class DatasetFileConfig(BaseModel):
    """Declarative specification for a single downloadable file or archive."""

    model_config = ConfigDict(frozen=True)

    filename: str
    url: str
    is_archive: bool = False
    extract_subdir: str = ""
    expected_md5: str | None = None
    description: str = ""


class DatasetSourceConfig(BaseModel):
    """Declarative specification for a complete dataset source."""

    model_config = ConfigDict(extra="ignore")

    name: str = Field(..., description="Human-readable dataset name")
    publisher: str = Field(default="", description="Publisher or repository source")
    doi: str | None = Field(default=None, description="DOI reference if academic")
    license: str = Field(default="Open", description="Dataset license identifier")
    target_subdir: str = Field(..., description="Subdirectory under data/raw/physical/ or data/raw/digital/")
    auth_required: bool = Field(default=False, description="Whether API key authentication is needed")
    env_var: str | None = Field(default=None, description="Primary environment variable for API key")
    env_vars: list[str] = Field(default_factory=list, description="Multiple environment variables if needed")
    direct_url_template: str | None = Field(default=None, description="URL template with {api_key} placeholder")
    archive_filename: str = Field(default="dataset.zip", description="Filename if downloading direct archive")
    dataset_slug: str | None = Field(default=None, description="Kaggle or HuggingFace dataset slug")
    fallback_url: str | None = Field(default=None, description="Documentation or download link for manual access")
    is_archive: bool = Field(default=False, description="Whether top-level direct URL is an archive")
    description: str = Field(default="", description="Overview of dataset contents and modality")
    files: dict[str, DatasetFileConfig] = Field(default_factory=dict, description="Individual file specifications")


def get_credential_from_env(key_name: str) -> str:
    """
    Retrieve an API key from system environment variables or local .env file.
    
    Args:
        key_name: Name of the environment variable (e.g. 'ROBOFLOW_API_KEY').
        
    Returns:
        The stripped string value, or empty string if not found or placeholder.
    """
    val = os.environ.get(key_name, "").strip()
    if val and not val.startswith("your_"):
        return val

    # Search local .env file
    env_path = Path(".env")
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith(f"{key_name}="):
                val = line.split("=", 1)[1].strip().strip("\"'")
                if val and not val.startswith("your_"):
                    return val

    return ""


class BaseDatasetDownloader(ABC):
    """Abstract base class for dataset ingestion and extraction."""

    def __init__(self, name: str, config: DatasetSourceConfig) -> None:
        self.name = name
        self.config = config

    def download_file(
        self,
        url: str,
        dest_path: Path,
        expected_md5: str | None = None,
        description: str = "",
        force: bool = False,
        chunk_size: int = 65536,
        user_agent: str = DEFAULT_USER_AGENT,
        extra_headers: dict[str, str] | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> Path:
        """
        Download a remote file with streaming progress, integrity check, and atomic rename.
        """
        dest_path = Path(dest_path)
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        if dest_path.exists() and not force:
            if expected_md5 is None or self.compute_md5(dest_path) == expected_md5:
                logger.info(f"File already exists and verified: {dest_path.name}")
                return dest_path

        temp_path = dest_path.with_suffix(dest_path.suffix + ".tmp")
        headers = {"User-Agent": user_agent}
        if extra_headers:
            headers.update(extra_headers)

        req = urllib.request.Request(url, headers=headers)
        ctx = ssl.create_default_context()

        try:
            with urllib.request.urlopen(req, context=ctx, timeout=60) as response:
                total_size = int(response.headers.get("Content-Length", 0))
                bytes_downloaded = 0
                hasher = hashlib.md5()

                logger.info(
                    f"Downloading {description or dest_path.name} "
                    f"({total_size / (1024 * 1024):.2f} MB)..."
                )

                with open(temp_path, "wb") as f_out:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        f_out.write(chunk)
                        hasher.update(chunk)
                        bytes_downloaded += len(chunk)
                        if progress_callback:
                            progress_callback(bytes_downloaded, total_size)

            actual_md5 = hasher.hexdigest()
            if expected_md5 and actual_md5.lower() != expected_md5.lower():
                if temp_path.exists():
                    temp_path.unlink()
                raise DatasetDownloadError(
                    f"MD5 checksum mismatch for {dest_path.name}. "
                    f"Expected: {expected_md5}, Got: {actual_md5}"
                )

            # Atomic rename from temporary file to destination
            if dest_path.exists():
                dest_path.unlink()
            shutil.move(str(temp_path), str(dest_path))
            logger.info(f"Successfully saved {dest_path.name} (MD5: {actual_md5})")
            return dest_path

        except Exception as e:
            if temp_path.exists():
                temp_path.unlink()
            if isinstance(e, DatasetDownloadError):
                raise
            raise DatasetDownloadError(f"Failed to download {url}: {e}") from e

    @staticmethod
    def compute_md5(file_path: Path, chunk_size: int = 65536) -> str:
        """Compute the MD5 checksum of a local file."""
        hasher = hashlib.md5()
        with open(file_path, "rb") as f:
            while chunk := f.read(chunk_size):
                hasher.update(chunk)
        return hasher.hexdigest()

    @staticmethod
    def safe_extract_zip(zip_path: Path, extract_dir: Path) -> list[Path]:
        """
        Safely extract a zip file with path traversal (Zip Slip) vulnerability protection.
        """
        extract_dir = Path(extract_dir).resolve()
        extract_dir.mkdir(parents=True, exist_ok=True)
        extracted_files: list[Path] = []

        with zipfile.ZipFile(zip_path, "r") as archive:
            for member in archive.infolist():
                target_path = (extract_dir / member.filename).resolve()
                if not str(target_path).startswith(str(extract_dir)):
                    raise DatasetDownloadError(
                        f"Security Warning: Attempted path traversal in zip archive ({member.filename})"
                    )

                archive.extract(member, extract_dir)
                if not member.is_dir():
                    extracted_files.append(target_path)

        logger.info(f"Extracted {len(extracted_files)} files to {extract_dir}")
        return extracted_files

    @abstractmethod
    def download(self, base_output_dir: Path, force: bool = False) -> Path:
        """Execute download and extraction of the dataset."""
        pass


class ChessReDDownloader(BaseDatasetDownloader):
    """
    Downloader for the Chess Recognition Dataset (ChessReD - VISAPP 2024).
    Driven completely by physical_sources.yaml configuration.
    """

    def download(self, base_output_dir: Path, force: bool = False) -> Path:
        target_dir = Path(base_output_dir) / self.config.target_subdir
        target_dir.mkdir(parents=True, exist_ok=True)

        for file_key, file_spec in self.config.files.items():
            logger.info(f"=== Fetching {self.config.name} [{file_key}] ===")
            local_dest = target_dir / file_spec.filename

            self.download_file(
                url=file_spec.url,
                dest_path=local_dest,
                expected_md5=file_spec.expected_md5,
                description=file_spec.description or local_dest.name,
                force=force,
            )

            if file_spec.is_archive:
                extract_target = target_dir / (file_spec.extract_subdir or file_key)
                self.safe_extract_zip(local_dest, extract_target)

        # Validate annotations format if present
        annotations_file = self.config.files.get("annotations")
        if annotations_file:
            annotations_path = target_dir / annotations_file.filename
            if annotations_path.exists():
                try:
                    with open(annotations_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        categories = len(data.get("categories", []))
                        images_count = len(data.get("images", []))
                        pieces_count = len(data.get("annotations", {}).get("pieces", []))
                        logger.info(
                            f"ChessReD manifest verified: {images_count} images, "
                            f"{pieces_count} piece bounding boxes, {categories} categories."
                        )
                except Exception as e:
                    logger.warning(f"Could not parse annotations.json: {e}")

        return target_dir


class HuggingFaceDatasetDownloader(BaseDatasetDownloader):
    """
    Downloader for Hugging Face digital/hybrid chess datasets.
    Driven completely by declarative YAML configuration (e.g. digital_sources.yaml).
    """

    def download(self, base_output_dir: Path, force: bool = False) -> Path:
        target_dir = Path(base_output_dir) / self.config.target_subdir
        target_dir.mkdir(parents=True, exist_ok=True)

        hf_token = get_credential_from_env("HF_TOKEN") or get_credential_from_env("HUGGINGFACE_TOKEN")
        headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else None

        for file_key, file_spec in self.config.files.items():
            logger.info(f"=== Fetching {self.config.name} [{file_key}] ===")
            local_dest = target_dir / file_spec.filename

            self.download_file(
                url=file_spec.url,
                dest_path=local_dest,
                expected_md5=file_spec.expected_md5,
                description=file_spec.description or local_dest.name,
                force=force,
                extra_headers=headers,
            )

            if file_spec.is_archive:
                extract_target = (target_dir / file_spec.extract_subdir) if file_spec.extract_subdir else target_dir
                self.safe_extract_zip(local_dest, extract_target)

        return target_dir


class RoboflowDatasetDownloader(BaseDatasetDownloader):
    """
    Downloader for Roboflow Universe physical and digital chess datasets.
    Driven completely by declarative YAML configuration.
    """

    def download(self, base_output_dir: Path, force: bool = False) -> Path:
        target_dir = Path(base_output_dir) / self.config.target_subdir
        target_dir.mkdir(parents=True, exist_ok=True)

        env_var_name = self.config.env_var or "ROBOFLOW_API_KEY"
        api_key = get_credential_from_env(env_var_name)

        if not api_key:
            logger.warning(
                f"{env_var_name} not configured in .env or environment.\n"
                f"To download '{self.config.name}' automatically:\n"
                f"1. Get a free API key at {self.config.fallback_url or 'https://app.roboflow.com'}\n"
                f"2. Add '{env_var_name}=your_key' to your .env file.\n"
                f"Skipping {self.config.name} download."
            )
            return target_dir

        if not self.config.direct_url_template:
            raise DatasetDownloadError(f"No direct_url_template defined for {self.name}")

        url = self.config.direct_url_template.format(api_key=api_key)
        archive_name = self.config.archive_filename or f"{self.name}.zip"
        zip_path = target_dir / archive_name

        logger.info(f"=== Downloading {self.config.name} (Direct HTTP Zip) ===")
        self.download_file(
            url=url,
            dest_path=zip_path,
            description=f"{self.config.name} archive",
            force=force,
        )

        self.safe_extract_zip(zip_path, target_dir)
        return target_dir


class KaggleDatasetDownloader(BaseDatasetDownloader):
    """
    Downloader for Kaggle physical and digital chess datasets.
    Driven completely by declarative YAML configuration.
    """

    def download(self, base_output_dir: Path, force: bool = False) -> Path:
        target_dir = Path(base_output_dir) / self.config.target_subdir
        target_dir.mkdir(parents=True, exist_ok=True)

        env_vars = self.config.env_vars or ["KAGGLE_USERNAME", "KAGGLE_KEY"]
        credentials = {var: get_credential_from_env(var) for var in env_vars}
        has_all_credentials = all(bool(v) for v in credentials.values())

        if not has_all_credentials:
            logger.info(
                f"Kaggle API credentials ({', '.join(env_vars)}) not fully configured. "
                f"Skipping {self.config.name} download. "
                f"Available at: {self.config.fallback_url or 'https://www.kaggle.com'}"
            )
            return target_dir

        dataset_slug = self.config.dataset_slug
        if not dataset_slug:
            raise DatasetDownloadError(f"No dataset_slug configured for {self.name}")

        try:
            import kaggle  # type: ignore

            logger.info(f"Downloading Kaggle dataset '{dataset_slug}'...")
            kaggle.api.dataset_download_files(
                dataset_slug,
                path=str(target_dir),
                unzip=True,
                force=force,
            )
            logger.info(f"Kaggle dataset downloaded to {target_dir}")
        except Exception as e:
            logger.warning(f"Kaggle download failed: {e}")

        return target_dir


class GenericDatasetDownloader(BaseDatasetDownloader):
    """Generic downloader for arbitrary HTTP/HTTPS datasets specified in config."""

    def download(self, base_output_dir: Path, force: bool = False) -> Path:
        target_dir = Path(base_output_dir) / self.config.target_subdir
        target_dir.mkdir(parents=True, exist_ok=True)

        for file_key, file_spec in self.config.files.items():
            local_dest = target_dir / file_spec.filename
            self.download_file(
                url=file_spec.url,
                dest_path=local_dest,
                expected_md5=file_spec.expected_md5,
                description=file_spec.description or local_dest.name,
                force=force,
            )
            if file_spec.is_archive:
                extract_target = target_dir / (file_spec.extract_subdir or file_key)
                self.safe_extract_zip(local_dest, extract_target)

        return target_dir


class DatasetRegistry:
    """Registry mapping dataset keys to their respective downloader implementations."""

    _SPECIALIZED_CLASSES: dict[str, type[BaseDatasetDownloader]] = {
        "chessred": ChessReDDownloader,
        "roboflow_staunton": RoboflowDatasetDownloader,
        "kaggle_tripod": KaggleDatasetDownloader,
        "huggingface_digital": HuggingFaceDatasetDownloader,
        "roboflow_chess_com": RoboflowDatasetDownloader,
        "roboflow_lichess": RoboflowDatasetDownloader,
    }

    @classmethod
    def load_sources_config(
        cls,
        config_path: Path | None = None,
        category: str = "all",
    ) -> dict[str, DatasetSourceConfig]:
        """
        Load sources specification from YAML configuration (SSOT).

        Args:
            config_path: Specific YAML file path. If None, resolves by category.
            category: 'physical', 'digital', or 'all' (merges both).
        """
        if config_path is not None:
            paths = [config_path]
        elif category == "physical":
            paths = [DEFAULT_PHYSICAL_CONFIG_PATH]
        elif category == "digital":
            paths = [DEFAULT_DIGITAL_CONFIG_PATH]
        else:
            paths = [DEFAULT_PHYSICAL_CONFIG_PATH, DEFAULT_DIGITAL_CONFIG_PATH]

        configs: dict[str, DatasetSourceConfig] = {}
        for path in paths:
            if not path.exists():
                continue
            with open(path, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
            for key, val in raw.items():
                if isinstance(val, dict):
                    configs[key] = DatasetSourceConfig(**val)
        return configs

    @classmethod
    def get_downloader(
        cls,
        name: str,
        config: DatasetSourceConfig | None = None,
        config_path: Path | None = None,
        category: str = "all",
    ) -> BaseDatasetDownloader:
        """
        Get a fully configured downloader instance by dataset name.
        """
        all_configs = cls.load_sources_config(config_path=config_path, category=category)

        if config is None:
            if name not in all_configs:
                raise KeyError(
                    f"Unknown dataset '{name}'. Available configured datasets: {list(all_configs.keys())}"
                )
            config = all_configs[name]

        downloader_cls = cls._SPECIALIZED_CLASSES.get(name)
        if downloader_cls is None:
            if "huggingface.co" in (config.fallback_url or "") or (
                config.files and any("huggingface.co" in f.url for f in config.files.values())
            ):
                downloader_cls = HuggingFaceDatasetDownloader
            else:
                downloader_cls = GenericDatasetDownloader

        return downloader_cls(name=name, config=config)

    @classmethod
    def list_available(cls, config_path: Path | None = None, category: str = "all") -> list[str]:
        configs = cls.load_sources_config(config_path=config_path, category=category)
        return list(configs.keys()) if configs else list(cls._SPECIALIZED_CLASSES.keys())

    @classmethod
    def list_physical(cls) -> list[str]:
        return cls.list_available(category="physical")

    @classmethod
    def list_digital(cls) -> list[str]:
        return cls.list_available(category="digital")

