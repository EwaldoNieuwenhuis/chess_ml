"""
Dataset ingestion, normalization, and builder package for Chess ML.
"""

from src.dataset.downloaders import (
    DEFAULT_DIGITAL_CONFIG_PATH,
    DEFAULT_PHYSICAL_CONFIG_PATH,
    DEFAULT_SOURCES_CONFIG_PATH,
    BaseDatasetDownloader,
    ChessReDDownloader,
    DatasetDownloadError,
    DatasetFileConfig,
    DatasetRegistry,
    DatasetSourceConfig,
    GenericDatasetDownloader,
    HuggingFaceDatasetDownloader,
    KaggleDatasetDownloader,
    RoboflowDatasetDownloader,
    get_credential_from_env,
)

__all__ = [
    "BaseDatasetDownloader",
    "ChessReDDownloader",
    "RoboflowDatasetDownloader",
    "KaggleDatasetDownloader",
    "HuggingFaceDatasetDownloader",
    "GenericDatasetDownloader",
    "DatasetRegistry",
    "DatasetDownloadError",
    "DatasetSourceConfig",
    "DatasetFileConfig",
    "DEFAULT_PHYSICAL_CONFIG_PATH",
    "DEFAULT_DIGITAL_CONFIG_PATH",
    "DEFAULT_SOURCES_CONFIG_PATH",
    "get_credential_from_env",
]

