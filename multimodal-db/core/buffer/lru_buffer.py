from __future__ import annotations

from collections import OrderedDict

from core.metrics import IOStats
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
        pages = [page for page in self._cache.values() if file_id is None or page.file_id == file_id]
        self._write_back(pages)

    # Un solo IOStats por StorageEngine, nadie mas lleva su propio contador
    def stats(self) -> IOStats:
        return self._storage.stats()

    # Pasa la asignacion de paginas al StorageEngine, el buffer no reserva espacio
    def allocate_page(self, file_id: str) -> int:
        return self._storage.allocate_page(file_id)

    # Primero se eligen todas las victimas, recien despues se escriben juntas
    def _make_room(self, protect: tuple[str, int] | None = None) -> None:
        overflow = len(self._cache) - self._capacity
        if overflow <= 0:
            return
        victims: list[tuple[tuple[str, int], Page]] = []
        for key, page in self._cache.items():
            if len(victims) >= overflow:
                break
            if key == protect or page.pin_count > 0:
                continue
            victims.append((key, page))
        if len(victims) < overflow:
            raise RuntimeError("el buffer esta lleno y todas sus paginas estan fijadas")
        self._write_back([page for _, page in victims])
        for key, _ in victims:
            del self._cache[key]

    # Escribe las paginas sucias ordenadas por archivo, para no saltar entre archivos
    def _write_back(self, pages: list[Page]) -> None:
        for page in sorted(pages, key=lambda p: (p.file_id, p.page_no)):
            if page.dirty:
                self._storage.write_page(page.file_id, page.page_no, bytes(page.data))
                page.dirty = False
