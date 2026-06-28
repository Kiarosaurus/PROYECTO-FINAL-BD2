#!/usr/bin/env python3
import base64
import io
import mimetypes
import os
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


# Decide de dónde salen las imágenes del seed
def _image_source() -> tuple[list[str], str]:
    local = _images_in(IMAGES_DIR)
    if local:
        return local, IMAGES_DIR
    drive = _drive_images()
    if drive:
        return drive, f"drive:{DRIVE_FOLDER_ID}"
    return [], "synthetic"


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


# Sube datos sintéticos y devuelve las filas a insertar
def _seed_synthetic(client: httpx.Client) -> list[tuple]:
    upload(client, "demo.png", _PNG_1X1, "image/png")
    upload(client, "demo.wav", _tiny_wav(), "audio/wav")
    return [(1, "demo.png"), (2, "demo.wav")]


def main() -> None:
    with httpx.Client(timeout=30) as client:
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


if __name__ == "__main__":
    main()
