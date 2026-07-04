from __future__ import annotations

import struct
from pathlib import Path

# Cada entrada del directorio guarda offset, capacidad y largo real de la pagina
_DIR_ENTRY = struct.Struct("<QII")


class HeapFile:

    def __init__(self, data_path: Path, dir_path: Path) -> None:
        self._data_path = data_path
        self._dir_path = dir_path
        self._data_path.touch(exist_ok=True)
        self._dir_path.touch(exist_ok=True)

    def page_count(self) -> int:
        return self._dir_path.stat().st_size // _DIR_ENTRY.size

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
        # Si el dato entra en el espacio ya reservado, se sobreescribe ahi mismo
        if len(data) <= capacity:
            self._write_at(offset, data)
            self._write_entry(page_no, offset, capacity, len(data))
            return
        # Si no entra, se agrega al final del archivo de datos
        new_offset = self._append(data)
        self._write_entry(page_no, new_offset, len(data), len(data))

    def allocate_page(self) -> int:
        page_no = self.page_count()
        self._ensure_slot(page_no)
        return page_no

    # Crea entradas vacias hasta cubrir page_no, incluso si nunca se pidio allocate_page
    def _ensure_slot(self, page_no: int) -> None:
        while self.page_count() <= page_no:
            offset = self._data_path.stat().st_size
            self._write_entry(self.page_count(), offset, 0, 0)

    def _read_entry(self, page_no: int) -> tuple[int, int, int] | None:
        if page_no < 0 or page_no >= self.page_count():
            return None
        with open(self._dir_path, "rb") as dir_file:
            dir_file.seek(page_no * _DIR_ENTRY.size)
            raw = dir_file.read(_DIR_ENTRY.size)
        return _DIR_ENTRY.unpack(raw)

    def _write_entry(self, page_no: int, offset: int, capacity: int, length: int) -> None:
        with open(self._dir_path, "r+b") as dir_file:
            dir_file.seek(page_no * _DIR_ENTRY.size)
            dir_file.write(_DIR_ENTRY.pack(offset, capacity, length))

    def _write_at(self, offset: int, data: bytes) -> None:
        with open(self._data_path, "r+b") as data_file:
            data_file.seek(offset)
            data_file.write(data)

    def _append(self, data: bytes) -> int:
        with open(self._data_path, "r+b") as data_file:
            data_file.seek(0, 2)
            offset = data_file.tell()
            data_file.write(data)
        return offset
