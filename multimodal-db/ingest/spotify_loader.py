from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterator

from ingest.ports import DatasetLoader
from service.dto import ColumnSpec, IndexInfo

# Columnas numéricas del csv de features en su orden original
FEATURE_COLUMNS = [
    "danceability",
    "energy",
    "key",
    "loudness",
    "mode",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "duration_ms",
]


# Carga el dataset de Spotify: letras, features y carátulas unidas por track
class SpotifyLoader(DatasetLoader):

    def __init__(
        self,
        features_csv: str | Path,
        lyrics_csv: str | Path,
        images_dir: str | Path | None = None,
    ) -> None:
        self._features_csv = Path(features_csv)
        self._lyrics_csv = Path(lyrics_csv)
        self._images_dir = Path(images_dir) if images_dir is not None else None

    def table_name(self) -> str:
        return "tracks"

    def columns(self) -> list[ColumnSpec]:
        return [
            ColumnSpec(name="id", type="INT"),
            ColumnSpec(name="track_id", type="TEXT"),
            ColumnSpec(name="title", type="TEXT"),
            ColumnSpec(name="artist", type="TEXT"),
            ColumnSpec(name="lyrics", type="TEXT"),
            ColumnSpec(name="feat", type="VECTOR"),
            ColumnSpec(name="image", type="TEXT"),
        ]

    def indexes(self) -> list[IndexInfo]:
        return [
            IndexInfo(column="id", index_type="hash"),
            IndexInfo(column="lyrics", index_type="inverted"),
            IndexInfo(column="feat", index_type="knn"),
        ]

    # Une cada letra con sus features y su carátula
    def rows(self) -> Iterator[tuple]:
        features = self._load_features()
        row_id = 0
        with open(self._lyrics_csv, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                track_id = row["track_id"]
                feat = features.get(track_id)
                if feat is None:
                    continue
                row_id += 1
                yield (
                    row_id,
                    track_id,
                    row["track_name"],
                    row["track_artist"],
                    row["lyrics"],
                    feat,
                    self._image_for(track_id),
                )

    def _load_features(self) -> dict[str, list[float]]:
        table: dict[str, list[float]] = {}
        with open(self._features_csv, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                table[row["track_id"]] = [float(row[col]) for col in FEATURE_COLUMNS]
        return table

    def _image_for(self, track_id: str) -> str:
        if self._images_dir is None:
            return ""
        name = f"{track_id}.jpg"
        if (self._images_dir / name).is_file():
            return name
        return ""
