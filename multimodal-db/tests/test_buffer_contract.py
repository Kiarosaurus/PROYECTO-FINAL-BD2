from __future__ import annotations

from typing import Callable

import pytest

from core.buffer.lru_buffer import LRUBufferManager
from core.ports.buffer import BufferManager
from core.ports.storage import StorageEngine
from tests.mocks import MockBufferManager, MockStorageEngine

BUFFER_FACTORIES: dict[str, Callable[[StorageEngine], BufferManager]] = {
    "mock": lambda storage: MockBufferManager(storage),
    "lru": lambda storage: LRUBufferManager(storage),
}


@pytest.fixture(params=list(BUFFER_FACTORIES), ids=list(BUFFER_FACTORIES))
def case(request) -> tuple[BufferManager, StorageEngine]:
    storage = MockStorageEngine()
    return BUFFER_FACTORIES[request.param](storage), storage


def test_get_returns_page_with_requested_identity(case) -> None:
    buffer, _storage = case
    page = buffer.get("t", 0)
    assert page.file_id == "t"
    assert page.page_no == 0


def test_get_serves_cached_page_without_rereading_disk(case) -> None:
    buffer, storage = case
    first = buffer.get("t", 0)
    reads_after_first = storage.stats().disk_reads
    second = buffer.get("t", 0)
    assert second is first
    assert storage.stats().disk_reads == reads_after_first


def test_flush_writes_dirty_page_to_storage(case) -> None:
    buffer, storage = case
    page = buffer.get("t", 0)
    page.data[:] = b"contenido"
    page.dirty = True
    buffer.flush()
    assert storage.read_page("t", 0) == b"contenido"
    assert page.dirty is False


def test_flush_ignores_clean_pages(case) -> None:
    buffer, storage = case
    buffer.get("t", 0)
    writes_before = storage.stats().disk_writes
    buffer.flush()
    assert storage.stats().disk_writes == writes_before


def test_flush_with_file_id_only_touches_that_file(case) -> None:
    buffer, storage = case
    page_a = buffer.get("a", 0)
    page_a.data[:] = b"de a"
    page_a.dirty = True
    page_b = buffer.get("b", 0)
    page_b.data[:] = b"de b"
    page_b.dirty = True
    buffer.flush("a")
    assert storage.read_page("a", 0) == b"de a"
    assert page_b.dirty is True


def test_pin_increments_pin_count(case) -> None:
    buffer, _storage = case
    page = buffer.get("t", 0)
    buffer.pin(page)
    assert page.pin_count == 1


def test_lru_evicts_least_recently_used_and_writes_it_back() -> None:
    storage = MockStorageEngine()
    buffer = LRUBufferManager(storage, capacity=2)
    page = buffer.get("t", 0)
    page.data[:] = b"viejo"
    page.dirty = True
    buffer.get("t", 1)
    buffer.get("t", 2)
    assert storage.read_page("t", 0) == b"viejo"


def test_lru_never_evicts_pinned_pages() -> None:
    storage = MockStorageEngine()
    buffer = LRUBufferManager(storage, capacity=2)
    pinned = buffer.get("t", 0)
    buffer.pin(pinned)
    buffer.get("t", 1)
    reloaded = buffer.get("t", 2)
    assert buffer.get("t", 0) is pinned
    assert reloaded.page_no == 2


def test_lru_raises_when_all_pages_are_pinned() -> None:
    storage = MockStorageEngine()
    buffer = LRUBufferManager(storage, capacity=1)
    buffer.pin(buffer.get("t", 0))
    with pytest.raises(RuntimeError):
        buffer.get("t", 1)
