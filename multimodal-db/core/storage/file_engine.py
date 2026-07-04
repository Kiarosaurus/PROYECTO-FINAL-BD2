from __future__ import annotations

from pathlib import Path

from core.metrics import IOStats
from core.ports.storage import StorageEngine
from core.storage.heap_file import HeapFile


class FileStorageEngine(StorageEngine):

    def __init__(self, base_dir: str | Path) -> None:
        self._base_dir = Path(base_dir)
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._files: dict[str, HeapFile] = {}
        self._stats = IOStats()

    def read_page(self, file_id: str, page_no: int) -> bytes:
        data = self._heap_file(file_id).read_page(page_no)
        self._stats.add_read()
        return data

    def write_page(self, file_id: str, page_no: int, data: bytes) -> None:
        self._heap_file(file_id).write_page(page_no, bytes(data))
        self._stats.add_write()

    def allocate_page(self, file_id: str) -> int:
        page_no = self._heap_file(file_id).allocate_page()
        self._stats.add_allocation()
        return page_no

    def stats(self) -> IOStats:
        return self._stats

    # Un heap file por file_id, se crea la primera vez que se pide
    def _heap_file(self, file_id: str) -> HeapFile:
        heap_file = self._files.get(file_id)
        if heap_file is None:
            heap_file = HeapFile(
                self._base_dir / f"{file_id}.heap",
                self._base_dir / f"{file_id}.dir",
                self._base_dir / f"{file_id}.free",
            )
            self._files[file_id] = heap_file
        return heap_file
