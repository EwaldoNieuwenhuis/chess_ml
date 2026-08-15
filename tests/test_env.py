"""
Unit tests for environment verification functions in scripts/verify_env.py.
"""

from scripts.verify_env import (
    CheckResult,
    check_python_version,
    check_pytorch_cuda,
    check_required_packages,
)


def test_check_python_version() -> None:
    res = check_python_version()
    assert isinstance(res, CheckResult)
    assert res.name == "Python Version"
    assert "Python" in res.details


def test_check_pytorch_cuda() -> None:
    results = check_pytorch_cuda()
    assert len(results) >= 1
    assert any("PyTorch" in r.name for r in results)


def test_check_required_packages() -> None:
    results = check_required_packages()
    assert len(results) > 0
    package_names = [r.name for r in results]
    assert any("Pydantic" in name for name in package_names)
