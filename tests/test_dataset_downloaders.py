"""
Unit and Integration Tests for Dataset Downloaders and Ingestion Engine.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.dataset import (
    BaseDatasetDownloader,
    ChessReDDownloader,
    DatasetDownloadError,
    DatasetRegistry,
    KaggleDatasetDownloader,
    RoboflowDatasetDownloader,
)


class DummyDownloader(BaseDatasetDownloader):
    """Concrete dummy subclass for testing BaseDatasetDownloader methods."""

    def download(self, base_output_dir: Path, force: bool = False) -> Path:
        return Path(base_output_dir) / self.name


class TestBaseDatasetDownloader:
    def test_compute_md5(self, tmp_path: Path) -> None:
        test_file = tmp_path / "sample.txt"
        test_file.write_bytes(b"chess-ml-test-content-123")
        expected_md5 = hashlib.md5(b"chess-ml-test-content-123").hexdigest()

        assert BaseDatasetDownloader.compute_md5(test_file) == expected_md5

    def test_safe_extract_zip(self, tmp_path: Path) -> None:
        zip_path = tmp_path / "test.zip"
        extract_dir = tmp_path / "extracted"

        # Create a valid zip archive
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("subfolder/file1.txt", "content1")
            zf.writestr("file2.txt", "content2")

        extracted = BaseDatasetDownloader.safe_extract_zip(zip_path, extract_dir)
        assert len(extracted) == 2
        assert (extract_dir / "subfolder" / "file1.txt").read_text() == "content1"
        assert (extract_dir / "file2.txt").read_text() == "content2"

    def test_zip_slip_security_prevention(self, tmp_path: Path) -> None:
        zip_path = tmp_path / "malicious.zip"
        extract_dir = tmp_path / "safe_extract"

        # Create a zip containing a path traversal attempt
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("../outside_target.txt", "malicious content")

        with pytest.raises(DatasetDownloadError, match="Security Warning"):
            BaseDatasetDownloader.safe_extract_zip(zip_path, extract_dir)

    def test_download_file_mock_stream(self, tmp_path: Path) -> None:
        downloader = DummyDownloader("dummy")
        dest_path = tmp_path / "data.bin"
        fake_content = b"downloaded-binary-data-stream"
        expected_md5 = hashlib.md5(fake_content).hexdigest()

        mock_response = MagicMock()
        mock_response.headers.get.return_value = str(len(fake_content))
        mock_response.read.side_effect = [fake_content, b""]
        mock_response.__enter__.return_value = mock_response

        with patch("urllib.request.urlopen", return_value=mock_response):
            res = downloader.download_file(
                url="https://example.com/data.bin",
                dest_path=dest_path,
                expected_md5=expected_md5,
            )

        assert res.exists()
        assert res.read_bytes() == fake_content

    def test_download_file_checksum_mismatch(self, tmp_path: Path) -> None:
        downloader = DummyDownloader("dummy")
        dest_path = tmp_path / "data.bin"
        fake_content = b"test content"

        mock_response = MagicMock()
        mock_response.headers.get.return_value = str(len(fake_content))
        mock_response.read.side_effect = [fake_content, b""]
        mock_response.__enter__.return_value = mock_response

        with patch("urllib.request.urlopen", return_value=mock_response):
            with pytest.raises(DatasetDownloadError, match="MD5 checksum mismatch"):
                downloader.download_file(
                    url="https://example.com/data.bin",
                    dest_path=dest_path,
                    expected_md5="00000000000000000000000000000000",
                )


class TestDatasetRegistry:
    def test_registry_list_and_lookup(self) -> None:
        available = DatasetRegistry.list_available()
        assert "chessred" in available
        assert "roboflow_staunton" in available
        assert "kaggle_tripod" in available

        downloader = DatasetRegistry.get_downloader("chessred")
        assert isinstance(downloader, ChessReDDownloader)

    def test_registry_unknown_dataset(self) -> None:
        with pytest.raises(KeyError, match="Unknown dataset"):
            DatasetRegistry.get_downloader("invalid_dataset_name")


class TestChessReDDownloader:
    def test_chessred_download_flow(self, tmp_path: Path) -> None:
        downloader = ChessReDDownloader()
        fake_json = json.dumps({
            "categories": [{"id": 0, "name": "white_pawn"}],
            "images": [{"id": 1, "path": "images/001.jpg"}],
            "annotations": {"pieces": [{"category_id": 0, "bbox": [10, 10, 50, 50]}]},
        }).encode("utf-8")

        # Fake zip for sample assets
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w") as zf:
            zf.writestr("test_sample.jpg", b"fake-jpg")
        fake_zip = zip_buf.getvalue()

        def fake_download_file(url: str, dest_path: Path, **kwargs: object) -> Path:
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            if "annotations" in url or "annotations.json" in str(dest_path):
                dest_path.write_bytes(fake_json)
            else:
                dest_path.write_bytes(fake_zip)
            return dest_path

        with patch.object(downloader, "download_file", side_effect=fake_download_file):
            result_dir = downloader.download(base_output_dir=tmp_path)

        assert (result_dir / "annotations.json").exists()
        assert (result_dir / "repo_assets" / "test_sample.jpg").exists()


class TestRoboflowDatasetDownloader:
    def test_roboflow_missing_api_key_graceful(self, tmp_path: Path) -> None:
        with patch.dict(os.environ, {}, clear=True):
            downloader = RoboflowDatasetDownloader()
            result_dir = downloader.download(base_output_dir=tmp_path)
            assert result_dir.exists()

    def test_roboflow_download_with_key(self, tmp_path: Path) -> None:
        with patch.dict(os.environ, {"ROBOFLOW_API_KEY": "valid_test_key"}):
            downloader = RoboflowDatasetDownloader()
            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, "w") as zf:
                zf.writestr("data.yaml", "names: [pawn]")
            fake_zip = zip_buf.getvalue()

            def fake_download_file(url: str, dest_path: Path, **kwargs: object) -> Path:
                assert "valid_test_key" in url
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                dest_path.write_bytes(fake_zip)
                return dest_path

            with patch.object(downloader, "download_file", side_effect=fake_download_file):
                result_dir = downloader.download(base_output_dir=tmp_path)

            assert (result_dir / "data.yaml").exists()


class TestKaggleDatasetDownloader:
    def test_kaggle_missing_credentials_graceful(self, tmp_path: Path) -> None:
        with patch.dict(os.environ, {}, clear=True):
            downloader = KaggleDatasetDownloader()
            result_dir = downloader.download(base_output_dir=tmp_path)
            assert result_dir.exists()
