from __future__ import annotations

from collections import OrderedDict

from core.ports.buffer import BufferManager, Page
from core.ports.storage import StorageEngine


class LRUBufferManager(BufferManager):

    def __init__(self, storage: StorageEngine, capacity: int = 64) -> None:
        if capacity < 1:
            raise ValueError("la capacidad del buffer debe ser al menos 1")
        self._storage = storage
        self._capacity = capacity
        self._cache: OrderedDict[tuple[str, int], Page] = OrderedDict()

    def get(self, file_id: str, page_no: int) -> Page:
        key = (file_id, page_no)
        if key in self._cache:
            # Se marca como la mas usada recientemente
            self._cache.move_to_end(key)
            return self._cache[key]
        data = bytearray(self._storage.read_page(file_id, page_no))
        page = Page(file_id, page_no, data)
        self._cache[key] = page
        # La pagina recien traida no se puede desalojar a si misma
        self._make_room(protect=key)
        return page

    def pin(self, page: Page) -> None:
        page.pin_count += 1

    def unpin(self, page: Page) -> None:
        if page.pin_count > 0:
            page.pin_count -= 1

    def flush(self, file_id: str | None = None) -> None:
        for page in self._cache.values():
            if file_id is not None and page.file_id != file_id:
                continue
            self._flush_page(page)

    # Desaloja las paginas menos usadas hasta volver a estar dentro de la capacidad
    def _make_room(self, protect: tuple[str, int] | None = None) -> None:
        for key, page in list(self._cache.items()):
            if len(self._cache) <= self._capacity:
                return
            if key == protect or page.pin_count > 0:
                continue
            self._flush_page(page)
            del self._cache[key]
        if len(self._cache) > self._capacity:
            raise RuntimeError("el buffer esta lleno y todas sus paginas estan fijadas")

    def _flush_page(self, page: Page) -> None:
        if page.dirty:
            self._storage.write_page(page.file_id, page.page_no, bytes(page.data))
            page.dirty = False
