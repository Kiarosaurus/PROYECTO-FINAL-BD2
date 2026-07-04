from __future__ import annotations

from pathlib import Path

from core.storage.page_layout import DIR_ENTRY, FREE_ENTRY


class HeapFile:

    def __init__(self, data_path: Path, dir_path: Path, free_path: Path) -> None:
        self._data_path = data_path
        self._dir_path = dir_path
        self._free_path = free_path
        self._data_path.touch(exist_ok=True)
        self._dir_path.touch(exist_ok=True)
        self._free_path.touch(exist_ok=True)
        self._free = self._load_free()

    def page_count(self) -> int:
        return self._dir_path.stat().st_size // DIR_ENTRY.size

    def read_page(self, page_no: int) -> bytes:
        entry = self._read_entry(page_no)
        if entry is None:
            return b""
        offset, _capacity, length = entry
        with open(self._data_path, "rb") as data_file:
            data_file.seek(offset)
            return data_file.read(length)

    def write_page(self, page_no: int, data: bytes) -> None:
        self._ensure_slot(page_no)
        offset, capacity, _length = self._read_entry(page_no)
        # Si el dato entra en el espacio ya reservado, se sobreescribe ahí mismo
        if len(data) <= capacity:
            self._write_at(offset, data)
            self._write_entry(page_no, offset, capacity, len(data))
            return
        # El hueco que deja esta página queda libre para una futura reasignación
        if capacity > 0:
            self._free.append((offset, capacity))
        new_offset, new_capacity = self._place(len(data))
        self._write_at(new_offset, data)
        self._write_entry(page_no, new_offset, new_capacity, len(data))
        self._save_free()

    def allocate_page(self) -> int:
        page_no = self.page_count()
        self._ensure_slot(page_no)
        return page_no

    # Crea entradas vacías hasta cubrir page_no, incluso si nunca se pidió allocate_page
    def _ensure_slot(self, page_no: int) -> None:
        while self.page_count() <= page_no:
            offset = self._data_path.stat().st_size
            self._write_entry(self.page_count(), offset, 0, 0)

    # Busca el hueco libre más ajustado que alcance, si no hay ninguno agranda el archivo
    def _place(self, size: int) -> tuple[int, int]:
        candidates = [i for i, (_offset, capacity) in enumerate(self._free) if capacity >= size]
        if not candidates:
            return self._end_of_file(), size
        best = min(candidates, key=lambda i: self._free[i][1])
        return self._free.pop(best)

    def _end_of_file(self) -> int:
        return self._data_path.stat().st_size

    def _load_free(self) -> list[tuple[int, int]]:
        raw = self._free_path.read_bytes()
        count = len(raw) // FREE_ENTRY.size
        return [FREE_ENTRY.unpack_from(raw, i * FREE_ENTRY.size) for i in range(count)]

    def _save_free(self) -> None:
        with open(self._free_path, "wb") as free_file:
            for offset, capacity in self._free:
                free_file.write(FREE_ENTRY.pack(offset, capacity))

    def _read_entry(self, page_no: int) -> tuple[int, int, int] | None:
        if page_no < 0 or page_no >= self.page_count():
            return None
        with open(self._dir_path, "rb") as dir_file:
            dir_file.seek(page_no * DIR_ENTRY.size)
            raw = dir_file.read(DIR_ENTRY.size)
        return DIR_ENTRY.unpack(raw)

    def _write_entry(self, page_no: int, offset: int, capacity: int, length: int) -> None:
        with open(self._dir_path, "r+b") as dir_file:
            dir_file.seek(page_no * DIR_ENTRY.size)
            dir_file.write(DIR_ENTRY.pack(offset, capacity, length))

    def _write_at(self, offset: int, data: bytes) -> None:
        with open(self._data_path, "r+b") as data_file:
            data_file.seek(offset)
            data_file.write(data)
