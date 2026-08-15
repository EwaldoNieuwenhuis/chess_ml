"""
Unit and Integration Tests for Dataset Downloaders and Ingestion Engine.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import sys
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.dataset import (
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



class DummyDownloader(BaseDatasetDownloader):
    """Concrete dummy subclass for testing BaseDatasetDownloader methods."""

    def download(self, base_output_dir: Path, force: bool = False) -> Path:
        return Path(base_output_dir) / self.config.target_subdir


@pytest.fixture
def dummy_config() -> DatasetSourceConfig:
    return DatasetSourceConfig(
        name="Dummy Test Dataset",
        target_subdir="dummy_test",
        license="MIT",
        files={
            "main": DatasetFileConfig(
                filename="data.bin",
                url="https://example.com/data.bin",
                expected_md5="098f6bcd4621d373cade4e832627b4f6",  # md5("test")
            )
        },
    )


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

    def test_download_file_mock_stream(self, tmp_path: Path, dummy_config: DatasetSourceConfig) -> None:
        downloader = DummyDownloader("dummy", dummy_config)
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

    def test_download_file_checksum_mismatch(self, tmp_path: Path, dummy_config: DatasetSourceConfig) -> None:
        downloader = DummyDownloader("dummy", dummy_config)
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


class TestDatasetRegistryAndConfig:
    def test_load_sources_config(self) -> None:
        configs = DatasetRegistry.load_sources_config()
        assert "chessred" in configs
        assert "roboflow_staunton" in configs
        assert "kaggle_tripod" in configs

        chessred = configs["chessred"]
        assert chessred.target_subdir == "chessred"
        assert "annotations" in chessred.files
        assert chessred.files["annotations"].expected_md5 == "d34bca5ad46ec7a8df96a1d3c36784f3"

    def test_registry_get_downloader(self) -> None:
        downloader = DatasetRegistry.get_downloader("chessred")
        assert isinstance(downloader, ChessReDDownloader)
        assert downloader.config.target_subdir == "chessred"

    def test_registry_unknown_dataset(self) -> None:
        with pytest.raises(KeyError, match="Unknown dataset"):
            DatasetRegistry.get_downloader("invalid_dataset_name")


class TestCredentialHelper:
    def test_get_credential_from_env(self, tmp_path: Path) -> None:
        with patch.dict(os.environ, {"MY_TEST_KEY": "secret_value_123"}):
            assert get_credential_from_env("MY_TEST_KEY") == "secret_value_123"

        with patch.dict(os.environ, {"MY_TEST_KEY": "your_placeholder"}):
            assert get_credential_from_env("MY_TEST_KEY") == ""


class TestChessReDDownloader:
    def test_chessred_download_flow(self, tmp_path: Path) -> None:
        config = DatasetSourceConfig(
            name="ChessReD Test",
            target_subdir="chessred",
            files={
                "annotations": DatasetFileConfig(
                    filename="annotations.json",
                    url="https://example.com/annotations.json",
                ),
                "sample": DatasetFileConfig(
                    filename="sample.zip",
                    url="https://example.com/sample.zip",
                    is_archive=True,
                    extract_subdir="repo_assets",
                ),
            },
        )
        downloader = ChessReDDownloader(name="chessred", config=config)

        fake_json = json.dumps({
            "categories": [{"id": 0, "name": "white_pawn"}],
            "images": [{"id": 1, "path": "images/001.jpg"}],
            "annotations": {"pieces": [{"category_id": 0, "bbox": [10, 10, 50, 50]}]},
        }).encode("utf-8")

        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w") as zf:
            zf.writestr("test_sample.jpg", b"fake-jpg")
        fake_zip = zip_buf.getvalue()

        def fake_download_file(url: str, dest_path: Path, **kwargs: object) -> Path:
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            if "annotations" in str(dest_path):
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
        config = DatasetSourceConfig(
            name="Roboflow Staunton",
            target_subdir="roboflow_staunton",
            env_var="ROBOFLOW_API_KEY",
            direct_url_template="https://universe.roboflow.com/ds/test?key={api_key}",
        )
        downloader = RoboflowDatasetDownloader(name="roboflow_staunton", config=config)

        with patch.dict(os.environ, {}, clear=True):
            result_dir = downloader.download(base_output_dir=tmp_path)
            assert result_dir.exists()

    def test_roboflow_download_with_key(self, tmp_path: Path) -> None:
        config = DatasetSourceConfig(
            name="Roboflow Staunton",
            target_subdir="roboflow_staunton",
            env_var="ROBOFLOW_API_KEY",
            direct_url_template="https://universe.roboflow.com/ds/test?key={api_key}",
            archive_filename="roboflow.zip",
        )
        downloader = RoboflowDatasetDownloader(name="roboflow_staunton", config=config)

        with patch.dict(os.environ, {"ROBOFLOW_API_KEY": "valid_test_key"}):
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
        config = DatasetSourceConfig(
            name="Kaggle Tripod",
            target_subdir="kaggle_tripod",
            dataset_slug="kneroma/test",
            env_vars=["KAGGLE_USERNAME", "KAGGLE_KEY"],
        )
        downloader = KaggleDatasetDownloader(name="kaggle_tripod", config=config)

        with patch.dict(os.environ, {}, clear=True):
            result_dir = downloader.download(base_output_dir=tmp_path)
            assert result_dir.exists()


class TestHuggingFaceDatasetDownloader:
    def test_huggingface_download_flow(self, tmp_path: Path) -> None:
        config = DatasetSourceConfig(
            name="HuggingFace Digital Test",
            target_subdir="hf_digital",
            files={
                "data_yaml": DatasetFileConfig(
                    filename="data.yaml",
                    url="https://huggingface.co/datasets/example/resolve/main/data.yaml",
                ),
                "part1": DatasetFileConfig(
                    filename="part1.zip",
                    url="https://huggingface.co/datasets/example/resolve/main/part1.zip",
                    is_archive=True,
                    extract_subdir="",
                ),
            },
        )
        downloader = HuggingFaceDatasetDownloader(name="huggingface_digital", config=config)

        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w") as zf:
            zf.writestr("images/train/img001.png", b"fake-png-data")
            zf.writestr("labels/train/img001.txt", "0 0.5 0.5 0.1 0.1")
        fake_zip = zip_buf.getvalue()

        def fake_download_file(url: str, dest_path: Path, **kwargs: object) -> Path:
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            if "data.yaml" in str(dest_path):
                dest_path.write_text("names: [white_pawn, white_knight]\nnc: 2", encoding="utf-8")
            else:
                dest_path.write_bytes(fake_zip)
            return dest_path

        with patch.object(downloader, "download_file", side_effect=fake_download_file):
            result_dir = downloader.download(base_output_dir=tmp_path)

        assert (result_dir / "data.yaml").exists()
        assert (result_dir / "images" / "train" / "img001.png").exists()
        assert (result_dir / "labels" / "train" / "img001.txt").exists()

    def test_huggingface_with_hf_token(self, tmp_path: Path) -> None:
        config = DatasetSourceConfig(
            name="HuggingFace Auth Test",
            target_subdir="hf_auth",
            files={
                "info": DatasetFileConfig(
                    filename="info.txt",
                    url="https://huggingface.co/datasets/example/info.txt",
                )
            },
        )
        downloader = HuggingFaceDatasetDownloader(name="huggingface_digital", config=config)

        captured_headers = {}

        def fake_download_file(url: str, dest_path: Path, **kwargs: object) -> Path:
            captured_headers.update(kwargs.get("extra_headers") or {})
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            dest_path.write_text("ok", encoding="utf-8")
            return dest_path

        with patch.dict(os.environ, {"HF_TOKEN": "hf_test_token_xyz"}):
            with patch.object(downloader, "download_file", side_effect=fake_download_file):
                downloader.download(base_output_dir=tmp_path)

        assert captured_headers.get("Authorization") == "Bearer hf_test_token_xyz"


class TestDigitalSourcesConfig:
    def test_load_digital_sources_config(self) -> None:
        configs = DatasetRegistry.load_sources_config(category="digital")
        assert "huggingface_digital" in configs
        assert "roboflow_chess_com" in configs
        assert "roboflow_lichess" in configs

        hf = configs["huggingface_digital"]
        assert hf.target_subdir == "huggingface_digital"
        assert "part1" in hf.files
        assert "data_yaml" in hf.files
        assert not hf.auth_required

    def test_registry_category_filtering(self) -> None:
        physical = DatasetRegistry.list_physical()
        digital = DatasetRegistry.list_digital()
        all_available = DatasetRegistry.list_available()

        assert "chessred" in physical
        assert "huggingface_digital" not in physical

        assert "huggingface_digital" in digital
        assert "chessred" not in digital

        assert "chessred" in all_available
        assert "huggingface_digital" in all_available

    def test_registry_get_digital_downloader(self) -> None:
        downloader = DatasetRegistry.get_downloader("huggingface_digital")
        assert isinstance(downloader, HuggingFaceDatasetDownloader)

        roboflow_downloader = DatasetRegistry.get_downloader("roboflow_chess_com")
        assert isinstance(roboflow_downloader, RoboflowDatasetDownloader)


class TestCLIScripts:
    def test_digital_cli_verify_only(self, tmp_path: Path) -> None:
        import subprocess

        cmd = [
            sys.executable,
            "scripts/download_digital_datasets.py",
            "--verify-only",
            "--output-dir",
            str(tmp_path),
        ]
        res = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        assert res.returncode == 0
        assert "Digital Dataset Local Verification Status" in res.stdout

    def test_digital_cli_help(self) -> None:
        import subprocess

        cmd = [sys.executable, "scripts/download_digital_datasets.py", "--help"]
        res = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        assert res.returncode == 0
        assert "--verify-only" in res.stdout
        assert "huggingface_digital" in res.stdout

    def test_physical_cli_verify_only(self, tmp_path: Path) -> None:
        import subprocess

        cmd = [
            sys.executable,
            "scripts/download_physical_datasets.py",
            "--verify-only",
            "--output-dir",
            str(tmp_path),
        ]
        res = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        assert res.returncode == 0
        assert "Physical Dataset Local Verification Status" in res.stdout



