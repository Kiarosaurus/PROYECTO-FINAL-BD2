#!/usr/bin/env python3
import base64
import io
import math
import mimetypes
import os
import struct
import tempfile
import wave

import httpx

API_URL = os.environ.get("API_URL", "http://localhost:8000")

# Carpeta local con imágenes reales para el seed
IMAGES_DIR = os.environ.get("SEED_IMAGES_DIR", "")

# Carpeta pública de Drive con las imágenes del proyecto
DRIVE_FOLDER_ID = os.environ.get(
    "DRIVE_FOLDER_ID", "1wMM1RE6PQpR1ietWuycNmI4Dj7WHBD0w"
)

# Bajar de Drive solo cuando se pide de forma explícita
USE_DRIVE = os.environ.get("USE_DRIVE", "") not in ("", "0", "false", "False")

# Donde quedan las imágenes bajadas de Drive
DRIVE_CACHE = os.environ.get(
    "DRIVE_CACHE_DIR", os.path.join(tempfile.gettempdir(), "mmdb_drive")
)

IMAGE_EXT = (".png", ".jpg", ".jpeg", ".webp", ".gif")

# Carpeta local con audios reales para el seed
AUDIO_DIR = os.environ.get("SEED_AUDIO_DIR", "")

AUDIO_EXT = (".wav", ".mp3", ".ogg")

# Frecuencias de los tonos sintéticos del seed
_SINE_FREQS = (220, 330, 440, 880, 1760, 3520)

# Top de palabras que conservan los índices de texto del seed
_SEED_VOCABULARY = 50

# Letras cortas para probar la búsqueda de texto
_SONGS = [
    (1, "Luna de abril", "La luna llena ilumina la noche y el corazón espera en silencio"),
    (2, "Camino al mar", "El camino baja hasta el mar y la sal se queda en la piel"),
    (3, "Fuego lento", "Un fuego lento quema el corazón cuando la noche se hace larga"),
    (4, "Viento norte", "El viento del norte trae lluvia fría sobre la ciudad dormida"),
    (5, "Guitarra rota", "Mi guitarra rota todavía canta canciones de amor en la noche"),
    (6, "Sol de enero", "El sol de enero calienta la plaza mientras baila la gente"),
]

# PNG de 1x1 usado cuando no hay imágenes locales
_PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
    "+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


# Arma un WAV corto en memoria para probar el audio player
def _tiny_wav() -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(8000)
        wav.writeframes(b"\x00\x00" * 800)
    return buffer.getvalue()


# Arma un WAV senoidal reproducible con la frecuencia pedida
def _sine_wav(freq: int, seconds: float = 4.0, rate: int = 8000) -> bytes:
    total = int(seconds * rate)
    samples = b"".join(
        struct.pack("<h", int(12000 * math.sin(2 * math.pi * freq * n / rate)))
        for n in range(total)
    )
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(samples)
    return buffer.getvalue()


# Lista los audios de una carpeta que existe
def _audios_in(folder: str) -> list[str]:
    if not folder or not os.path.isdir(folder):
        return []
    paths = []
    for name in sorted(os.listdir(folder)):
        if name.lower().endswith(AUDIO_EXT):
            paths.append(os.path.join(folder, name))
    return paths


# Decide de dónde salen los audios del seed
def _audio_source() -> tuple[list[str], str]:
    local = _audios_in(AUDIO_DIR)
    if local:
        return local, AUDIO_DIR
    cache = os.path.join(tempfile.gettempdir(), "mmdb_audio")
    os.makedirs(cache, exist_ok=True)
    paths = []
    for freq in _SINE_FREQS:
        path = os.path.join(cache, f"tone_{freq}.wav")
        if not os.path.isfile(path):
            with open(path, "wb") as handle:
                handle.write(_sine_wav(freq))
        paths.append(path)
    return paths, "synthetic"


# Lista las imágenes de una carpeta que existe
def _images_in(folder: str) -> list[str]:
    if not folder or not os.path.isdir(folder):
        return []
    paths = []
    for name in sorted(os.listdir(folder)):
        if name.lower().endswith(IMAGE_EXT):
            paths.append(os.path.join(folder, name))
    return paths


# Baja la carpeta pública de Drive y lista sus imágenes
def _drive_images() -> list[str]:
    if not USE_DRIVE:
        return []
    import gdown

    os.makedirs(DRIVE_CACHE, exist_ok=True)
    gdown.download_folder(
        id=DRIVE_FOLDER_ID,
        output=DRIVE_CACHE,
        quiet=True,
        use_cookies=False,
    )
    return _images_in(DRIVE_CACHE)


# Genera imágenes de ruido con suficientes keypoints para SIFT
def _noise_images() -> list[str]:
    try:
        import cv2
        import numpy as np
    except ImportError:
        return []
    cache = os.path.join(tempfile.gettempdir(), "mmdb_images")
    os.makedirs(cache, exist_ok=True)
    paths = []
    for seed in range(1, 7):
        path = os.path.join(cache, f"noise_{seed}.png")
        if not os.path.isfile(path):
            rng = np.random.default_rng(seed)
            image = rng.integers(0, 255, size=(128, 128), dtype=np.uint8)
            cv2.imwrite(path, image)
        paths.append(path)
    return paths


# Decide de dónde salen las imágenes del seed
def _image_source() -> tuple[list[str], str]:
    local = _images_in(IMAGES_DIR)
    if local:
        return local, IMAGES_DIR
    drive = _drive_images()
    if drive:
        return drive, f"drive:{DRIVE_FOLDER_ID}"
    return _noise_images(), "synthetic"


def run_query(client: httpx.Client, sql: str) -> dict:
    res = client.post(f"{API_URL}/query", json={"sql": sql})
    res.raise_for_status()
    return res.json()


def upload(client: httpx.Client, name: str, data: bytes, content_type: str) -> dict:
    files = {"file": (name, data, content_type)}
    res = client.post(f"{API_URL}/upload", files=files)
    res.raise_for_status()
    return res.json()


# Sube las imágenes locales y devuelve las filas a insertar
def _seed_from_folder(client: httpx.Client, paths: list[str]) -> list[tuple]:
    rows = []
    for i, path in enumerate(paths, start=1):
        name = os.path.basename(path)
        content_type = mimetypes.guess_type(name)[0] or "application/octet-stream"
        with open(path, "rb") as handle:
            upload(client, name, handle.read(), content_type)
        rows.append((i, name))
    return rows


# Construye el índice KNN de imágenes y muestra una búsqueda de ejemplo
def _knn_demo(paths: list[str]) -> None:
    if len(paths) < 2:
        return
    from multimedia.codebook.kmeans_codebook import KMeansCodebook
    from multimedia.extractors.sift_extractor import SIFTExtractor
    from multimedia.knn_index import MultimediaKNNIndex
    from multimedia.pipeline import MultimediaPipeline

    pipeline = MultimediaPipeline(SIFTExtractor(), KMeansCodebook(k=32), MultimediaKNNIndex())
    try:
        result = pipeline.build_from_files(paths)
    except ValueError as error:
        print("knn: no se pudo entrenar el codebook:", error)
        return
    if not result.success:
        print("knn:", result.message)
        return
    query = paths[0]
    top = pipeline.search_file(query, k=3)
    print("knn query:", os.path.basename(query))
    for key, score in top.records:
        print(f"knn hit: {key} score={score:.3f}")


# Construye el índice KNN de audio y muestra una búsqueda de ejemplo
def _audio_knn_demo(paths: list[str]) -> None:
    if len(paths) < 2:
        return
    from multimedia.codebook.kmeans_codebook import KMeansCodebook
    from multimedia.extractors.mfcc_extractor import MFCCExtractor
    from multimedia.knn_index import MultimediaKNNIndex
    from multimedia.pipeline import MultimediaPipeline

    pipeline = MultimediaPipeline(MFCCExtractor(), KMeansCodebook(k=8), MultimediaKNNIndex())
    try:
        result = pipeline.build_from_files(paths)
    except ValueError as error:
        print("audio knn: no se pudo entrenar el codebook:", error)
        return
    if not result.success:
        print("audio knn:", result.message)
        return
    query = paths[0]
    top = pipeline.search_file(query, k=3)
    print("audio knn query:", os.path.basename(query))
    for key, score in top.records:
        print(f"audio knn hit: {key} score={score:.3f}")


# Sube datos sintéticos y devuelve las filas a insertar
def _seed_synthetic(client: httpx.Client) -> list[tuple]:
    upload(client, "demo.png", _PNG_1X1, "image/png")
    upload(client, "demo.wav", _tiny_wav(), "audio/wav")
    return [(1, "demo.png"), (2, "demo.wav")]


# Sube una copia del primer archivo como consulta de ejemplo con nombre fijo
def _upload_query_copy(client: httpx.Client, paths: list[str], query_name: str) -> None:
    if not paths:
        return
    content_type = mimetypes.guess_type(query_name)[0] or "application/octet-stream"
    with open(paths[0], "rb") as handle:
        upload(client, query_name, handle.read(), content_type)


# Crea la tabla de letras con su índice de texto
def _seed_songs(client: httpx.Client) -> None:
    run_query(client, "CREATE TABLE songs (id INT, title TEXT, lyrics TEXT)")
    run_query(
        client,
        f"CREATE INDEX ON songs (lyrics) USING inverted WITH (vocabulary = {_SEED_VOCABULARY})",
    )
    values = ", ".join(f'({i}, "{title}", "{lyrics}")' for i, title, lyrics in _SONGS)
    run_query(client, f"INSERT INTO songs (id, title, lyrics) VALUES {values}")


# Crea la tabla que une portada y letra para la búsqueda combinada
def _seed_albums(client: httpx.Client, names: list[str]) -> None:
    if len(names) < 2:
        print("albums: se omite la tabla híbrida por falta de imágenes")
        return
    run_query(client, "CREATE TABLE albums (id INT, cover VECTOR, lyrics TEXT)")
    run_query(client, "CREATE INDEX ON albums (cover) USING knn")
    run_query(
        client,
        f"CREATE INDEX ON albums (lyrics) USING inverted WITH (vocabulary = {_SEED_VOCABULARY})",
    )
    values = ", ".join(
        f'({i}, "{name}", "{lyrics}")'
        for i, (name, (_id, _title, lyrics)) in enumerate(zip(names, _SONGS), start=1)
    )
    run_query(client, f"INSERT INTO albums (id, cover, lyrics) VALUES {values}")


# Crea una tabla con índice knn sobre archivos ya subidos
def _seed_knn_table(client: httpx.Client, table: str, column: str, names: list[str]) -> None:
    if len(names) < 2:
        print(f"{table}: se omite el índice knn por falta de archivos")
        return
    run_query(client, f"CREATE TABLE {table} (id INT, {column} VECTOR)")
    run_query(client, f"CREATE INDEX ON {table} ({column}) USING knn")
    values = ", ".join(f'({i}, "{name}")' for i, name in enumerate(names, start=1))
    run_query(client, f"INSERT INTO {table} (id, {column}) VALUES {values}")


# Corre una consulta de ejemplo y muestra sus primeras filas
def _print_sample(client: httpx.Client, sql: str) -> None:
    result = run_query(client, sql)
    print("query:", sql)
    for row in result["rows"][:5]:
        print("  row:", row)


def main() -> None:
    with httpx.Client(timeout=120) as client:
        client.get(f"{API_URL}/health").raise_for_status()

        run_query(client, "CREATE TABLE media (id INT, path TEXT)")
        run_query(client, "CREATE INDEX ON media (id) USING hash")

        images, source = _image_source()
        rows = _seed_from_folder(client, images) if images else _seed_synthetic(client)

        values = ", ".join(f'({i}, "{name}")' for i, name in rows)
        run_query(client, f"INSERT INTO media (id, path) VALUES {values}")
        result = run_query(client, "SELECT * FROM media")

        print("source:", source)
        print("columns:", result["columns"])
        for row in result["rows"]:
            print("row:", row)

        audios, audio_source = _audio_source()
        print("audio source:", audio_source)
        audio_rows = _seed_from_folder(client, audios)

        # Los archivos de consulta deben existir antes de armar los índices knn
        _upload_query_copy(client, images, "demo_query.png")
        _upload_query_copy(client, audios, "demo_query.wav")

        _seed_songs(client)
        photo_names = [name for _i, name in rows] if images else []
        _seed_knn_table(client, "photos", "img", photo_names)
        _seed_albums(client, photo_names)
        _seed_knn_table(client, "tracks", "audio", [name for _i, name in audio_rows])

        # Las mismas consultas que ofrecen los snippets del frontend
        _print_sample(client, 'SELECT * FROM songs WHERE MATCH(lyrics, "corazón noche", 3)')
        _print_sample(client, 'SELECT * FROM photos WHERE KNN(img, "demo_query.png", 5)')
        _print_sample(client, 'SELECT * FROM tracks WHERE KNN(audio, "demo_query.wav", 3)')
        _print_sample(
            client,
            'SELECT * FROM albums WHERE HYBRID(cover, "demo_query.png", lyrics, "fuego corazón", 3)',
        )

    _knn_demo(images)
    _audio_knn_demo(audios)


if __name__ == "__main__":
    main()
