from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import pytest

from tests.download_data import _extract_fma_audio, _extract_fma_tracks, _sha1, _verify_fma_zips


def test_sha1_matches_hashlib(tmp_path: Path) -> None:
    payload = b"contenido de prueba"
    target = tmp_path / "sample.bin"
    target.write_bytes(payload)

    assert _sha1(target) == hashlib.sha1(payload).hexdigest()


def test_extract_fma_tracks_ignores_mac_junk(tmp_path: Path) -> None:
    zip_path = tmp_path / "fma_metadata.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("fma_metadata/._tracks.csv", "basura")
        zf.writestr("fma_metadata/tracks.csv", "track_id,title\n1,uno\n")
        zf.writestr("fma_metadata/genres.csv", "genre_id\n1\n")

    _extract_fma_tracks(zip_path, tmp_path)

    assert (tmp_path / "tracks.csv").read_text() == "track_id,title\n1,uno\n"
    assert not (tmp_path / "genres.csv").exists()


def test_extract_fma_audio_renames_to_audio_folder(tmp_path: Path) -> None:
    zip_path = tmp_path / "fma_small.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("fma_small/000/000123.mp3", "mp3")
        zf.writestr("fma_small/155/155999.mp3", "mp3")
    stale = tmp_path / "audio"
    stale.mkdir()
    (stale / "viejo.mp3").write_text("x")

    _extract_fma_audio(zip_path, tmp_path)

    assert (tmp_path / "audio" / "000" / "000123.mp3").is_file()
    assert (tmp_path / "audio" / "155" / "155999.mp3").is_file()
    assert not (tmp_path / "audio" / "viejo.mp3").exists()
    assert not (tmp_path / "fma_small").exists()


def test_verify_fma_zips_fails_on_missing_or_corrupt(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        _verify_fma_zips(tmp_path)

    (tmp_path / "fma_metadata.zip").write_bytes(b"no es el zip real")
    (tmp_path / "fma_small.zip").write_bytes(b"tampoco")
    with pytest.raises(ValueError):
        _verify_fma_zips(tmp_path)
