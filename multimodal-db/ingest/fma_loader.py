from __future__ import annotations

import ast
import csv
from pathlib import Path
from typing import Iterator

from ingest.ports import DatasetLoader
from service.dto import ColumnSpec, IndexInfo

AUDIO_EXTENSIONS = (".mp3", ".wav", ".ogg")


# Carga el dataset FMA: metadata en csv y archivos de audio en disco
class FMALoader(DatasetLoader):

    def __init__(self, tracks_csv: str | Path, audio_dir: str | Path) -> None:
        self._tracks_csv = Path(tracks_csv)
        self._audio_dir = Path(audio_dir)

    def table_name(self) -> str:
        return "fma_tracks"

    def columns(self) -> list[ColumnSpec]:
        return [
            ColumnSpec(name="id", type="INT"),
            ColumnSpec(name="track_id", type="TEXT"),
            ColumnSpec(name="title", type="TEXT"),
            ColumnSpec(name="artist", type="TEXT"),
            ColumnSpec(name="genre", type="TEXT"),
            ColumnSpec(name="audio", type="TEXT"),
        ]

    def indexes(self) -> list[IndexInfo]:
        return [
            IndexInfo(column="id", index_type="hash"),
            IndexInfo(column="audio", index_type="knn"),
        ]

    # Solo salen los tracks cuyo archivo de audio existe en disco
    def rows(self) -> Iterator[tuple]:
        row_id = 0
        for track_id, fields in self._metadata():
            audio = self._audio_for(track_id)
            if audio is None:
                continue
            row_id += 1
            yield (
                row_id,
                str(track_id),
                fields.get("track_title", ""),
                fields.get("artist_name", ""),
                self._top_genre(fields.get("track_genres", "")),
                audio,
            )

    # El csv de FMA trae una sola fila de encabezado con nombres de columna planos
    def _metadata(self) -> Iterator[tuple[int, dict[str, str]]]:
        with open(self._tracks_csv, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                track_id = row.get("track_id", "").strip()
                if not track_id:
                    continue
                yield int(track_id), row

    # La columna track_genres trae una lista serializada de géneros, se usa el primero
    @staticmethod
    def _top_genre(raw: str) -> str:
        if not raw:
            return ""
        try:
            genres = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return ""
        if not genres:
            return ""
        return genres[0].get("genre_title", "")

    # Busca el archivo del track en la estructura de carpetas de FMA
    def _audio_for(self, track_id: int) -> str | None:
        stem = f"{track_id:06d}"
        folders = [self._audio_dir / stem[:3], self._audio_dir]
        for folder in folders:
            for extension in AUDIO_EXTENSIONS:
                candidate = folder / f"{stem}{extension}"
                if candidate.is_file():
                    return candidate.name
        return None
