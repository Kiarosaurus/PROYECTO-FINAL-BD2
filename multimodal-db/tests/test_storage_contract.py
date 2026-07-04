from __future__ import annotations

from pathlib import Path
from typing import Callable

import pytest

from core.metrics import IOStats
from core.ports.storage import StorageEngine
from core.storage.file_engine import FileStorageEngine
from tests.mocks import MockStorageEngine

STORAGE_FACTORIES: dict[str, Callable[[Path], StorageEngine]] = {
    "mock": lambda tmp_path: MockStorageEngine(),
    "file": lambda tmp_path: FileStorageEngine(tmp_path),
}


@pytest.fixture(params=list(STORAGE_FACTORIES), ids=list(STORAGE_FACTORIES))
def storage(request, tmp_path: Path) -> StorageEngine:
    return STORAGE_FACTORIES[request.param](tmp_path)


def test_read_unwritten_page_returns_empty_bytes(storage: StorageEngine) -> None:
    assert storage.read_page("t", 0) == b""


def test_write_then_read_roundtrip(storage: StorageEngine) -> None:
    storage.write_page("t", 0, b"hola")
    assert storage.read_page("t", 0) == b"hola"


def test_overwrite_replaces_previous_content(storage: StorageEngine) -> None:
    storage.write_page("t", 0, b"primero")
    storage.write_page("t", 0, b"segundo mucho mas largo que el primero")
    assert storage.read_page("t", 0) == b"segundo mucho mas largo que el primero"


def test_write_without_prior_allocate_still_works(storage: StorageEngine) -> None:
    storage.write_page("t", 5, b"directo")
    assert storage.read_page("t", 5) == b"directo"


def test_allocate_page_returns_increasing_numbers(storage: StorageEngine) -> None:
    first = storage.allocate_page("t")
    second = storage.allocate_page("t")
    assert second > first


def test_pages_are_isolated_per_file_id(storage: StorageEngine) -> None:
    storage.write_page("a", 0, b"de a")
    storage.write_page("b", 0, b"de b")
    assert storage.read_page("a", 0) == b"de a"
    assert storage.read_page("b", 0) == b"de b"


def test_stats_reflects_reads_writes_and_allocations(storage: StorageEngine) -> None:
    before = storage.stats()
    reads_before, writes_before, allocs_before = before.disk_reads, before.disk_writes, before.pages_allocated

    storage.allocate_page("t")
    storage.write_page("t", 0, b"x")
    storage.read_page("t", 0)

    after = storage.stats()
    assert after.pages_allocated == allocs_before + 1
    assert after.disk_writes == writes_before + 1
    assert after.disk_reads == reads_before + 1


def test_stats_returns_an_iostats_instance(storage: StorageEngine) -> None:
    assert isinstance(storage.stats(), IOStats)
