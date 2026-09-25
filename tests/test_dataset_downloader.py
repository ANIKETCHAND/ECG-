"""
Tests for the PhysioNet dataset downloader
==========================================

Regression coverage for a real bug: the downloader passed ``records=None`` to
``wfdb.dl_database`` to mean "the whole database". wfdb does not enumerate the
database on ``None`` — it raises ``'NoneType' object is not iterable`` — so every
full-database download reported ``DOWNLOAD_FAILED`` and the expand-the-dataset
path was silently dead. Since the corpus size is the single biggest legitimate
lever on model accuracy, a broken downloader quietly capped model quality.

These tests stub the network so they run offline and never touch PhysioNet.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest

PROJ_DIR = Path(__file__).resolve().parent.parent
for candidate in (PROJ_DIR / "src", PROJ_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from datasets import downloader as downloader_module
from datasets.downloader import DatasetDownloader


RECORDS_MANIFEST = "100\n101\n\n106\n119\n200\n208\n213\n"


class _FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


@pytest.fixture
def no_network(monkeypatch):
    def boom(*args, **kwargs):
        raise OSError("network disabled in tests")

    monkeypatch.setattr(downloader_module.urllib.request, "urlopen", boom)


def test_resolve_record_list_parses_the_manifest(monkeypatch):
    def fake_urlopen(url, timeout=None):
        assert url == "https://physionet.org/files/mitdb/1.0.0/RECORDS"
        return _FakeResponse(RECORDS_MANIFEST.encode())

    monkeypatch.setattr(downloader_module.urllib.request, "urlopen", fake_urlopen)
    records = DatasetDownloader.resolve_record_list("mitdb", "1.0.0")
    assert records == ["100", "101", "106", "119", "200", "208", "213"]


def test_resolve_record_list_falls_back_across_versioned_urls(monkeypatch):
    seen = []

    def fake_urlopen(url, timeout=None):
        seen.append(url)
        if url.endswith("/1.0.0/RECORDS"):
            return _FakeResponse(RECORDS_MANIFEST.encode())
        raise OSError("404")

    monkeypatch.setattr(downloader_module.urllib.request, "urlopen", fake_urlopen)
    records = DatasetDownloader.resolve_record_list("mitdb", version=None)
    assert len(records) == 7
    # The unversioned path is tried first, then the versioned fallback.
    assert seen[0] == "https://physionet.org/files/mitdb/RECORDS"
    assert seen[-1] == "https://physionet.org/files/mitdb/1.0.0/RECORDS"


def test_resolve_record_list_raises_rather_than_returning_nothing(no_network):
    with pytest.raises(RuntimeError) as exc:
        DatasetDownloader.resolve_record_list("mitdb", "1.0.0")
    assert "record manifest" in str(exc.value).lower()


def test_resolve_record_list_rejects_an_empty_manifest(monkeypatch):
    monkeypatch.setattr(
        downloader_module.urllib.request,
        "urlopen",
        lambda url, timeout=None: _FakeResponse(b"\n\n"),
    )
    with pytest.raises(RuntimeError):
        DatasetDownloader.resolve_record_list("mitdb", "1.0.0")


def test_download_dataset_passes_a_real_record_list_to_wfdb(monkeypatch, tmp_path):
    """The full-database path must never hand wfdb ``records=None`` again."""
    captured = {}

    def fake_dl_database(db_dir, dl_dir, records=None, keep_subdirs=False, overwrite=False):
        captured["db_dir"] = db_dir
        captured["dl_dir"] = dl_dir
        captured["records"] = records
        Path(dl_dir).mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(downloader_module.wfdb, "dl_database", fake_dl_database)
    monkeypatch.setattr(
        downloader_module.urllib.request,
        "urlopen",
        lambda url, timeout=None: _FakeResponse(RECORDS_MANIFEST.encode()),
    )
    monkeypatch.setattr(
        DatasetDownloader, "is_dataset_downloaded", lambda self, dataset_id: False
    )
    # staticmethod so the instance call does not bind `self` into `slug`.
    monkeypatch.setattr(
        DatasetDownloader,
        "resolve_record_list",
        staticmethod(lambda slug, version=None: ["100", "101"]),
    )

    downloader = DatasetDownloader()
    monkeypatch.setattr(
        downloader.registry,
        "get_local_path",
        lambda dataset_id: tmp_path / dataset_id,
    )

    result = downloader.download_dataset("mit_bih_arrhythmia")

    assert result["status"] == "DOWNLOADED", result
    assert captured["records"] == ["100", "101"]
    assert captured["records"] is not None


def test_download_dataset_reports_failure_when_manifest_is_unreachable(
    monkeypatch, tmp_path, no_network
):
    monkeypatch.setattr(
        DatasetDownloader, "is_dataset_downloaded", lambda self, dataset_id: False
    )
    downloader = DatasetDownloader()
    monkeypatch.setattr(
        downloader.registry,
        "get_local_path",
        lambda dataset_id: tmp_path / dataset_id,
    )

    result = downloader.download_dataset("mit_bih_arrhythmia")
    assert result["status"] == "DOWNLOAD_FAILED"
    assert "manifest" in result["message"].lower()


def test_unknown_dataset_is_reported_not_downloaded(tmp_path):
    downloader = DatasetDownloader()
    result = downloader.download_dataset("not_a_real_dataset")
    assert result["status"] == "ERROR"
