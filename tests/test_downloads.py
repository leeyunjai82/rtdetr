# Apache-2.0
"""Weight resolution and the cache — no network touched in these tests."""

from __future__ import annotations

import pytest

from easydetect import downloads
from easydetect.errors import DownloadError, ModelNotFoundError


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("dfine_s", "dfine-s"),
        ("D-FINE-S", "dfine-s"),
        ("DFINE_X", "dfine-x"),
        ("dfine-s.pt", "dfine-s"),
        ("dfine-l.xml", "dfine-l"),
    ],
)
def test_names_normalise_to_one_spelling(given, expected):
    assert downloads.normalize_name(given) == expected
    assert downloads.is_model_name(given)


def test_things_that_are_not_model_names_are_not_mistaken_for_one():
    assert not downloads.is_model_name("best.pt")
    assert not downloads.is_model_name("dfine-xl")
    assert not downloads.is_model_name("rtdetr-r18")


def test_the_cache_lives_under_easydetect_home(tmp_path, monkeypatch):
    monkeypatch.setenv("EASYDETECT_HOME", str(tmp_path / "weights"))
    assert downloads.cache_dir() == tmp_path / "weights"
    assert downloads.cache_dir().is_dir()


def test_the_mirror_url_can_be_pointed_at_an_internal_copy(monkeypatch):
    assert downloads.assets_url() == downloads.DEFAULT_ASSETS_URL
    monkeypatch.setenv("EASYDETECT_ASSETS_URL", "https://mirror.internal/models/")
    assert downloads.assets_url() == "https://mirror.internal/models"


def test_an_already_cached_file_is_not_downloaded_again(tmp_path, monkeypatch):
    cached = tmp_path / "dfine-s" / "dfine-s.xml"
    cached.parent.mkdir(parents=True)
    cached.write_text("<net/>")
    monkeypatch.setattr(
        downloads.urllib.request, "urlopen", lambda *a, **k: pytest.fail("network touched")
    )
    assert downloads.download("https://example.com/x.xml", cached) == cached


def test_an_unknown_variant_never_reaches_the_network():
    with pytest.raises(ModelNotFoundError, match="Known names"):
        downloads.download_ir("easydetect-r101")
    with pytest.raises(ModelNotFoundError, match="Known names"):
        downloads.download_checkpoint("easydetect-r101")


def test_a_missing_mirror_entry_suggests_training_your_own(tmp_path, monkeypatch):
    monkeypatch.setenv("EASYDETECT_HOME", str(tmp_path))
    monkeypatch.setattr(
        downloads, "download", lambda *a, **k: (_ for _ in ()).throw(DownloadError("HTTP 404"))
    )
    with pytest.raises(ModelNotFoundError, match="train it yourself"):
        downloads.download_ir("dfine-s")
    with pytest.raises(ModelNotFoundError, match="ImageNet backbone"):
        downloads.download_checkpoint("dfine-s")


def test_labels_txt_is_optional_when_fetching_an_ir(tmp_path, monkeypatch):
    """A mirror entry without labels.txt still yields a usable model."""
    monkeypatch.setenv("EASYDETECT_HOME", str(tmp_path))
    wanted = []

    def fake_download(url, dest, progress=True):
        wanted.append(dest.name)
        if dest.name == "labels.txt":
            raise DownloadError("HTTP 404")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text("stub")
        return dest

    monkeypatch.setattr(downloads, "download", fake_download)
    xml = downloads.download_ir("dfine_s")
    assert xml.name == "dfine-s.xml"
    assert wanted == ["dfine-s.xml", "dfine-s.bin", "labels.txt"]
