#!/usr/bin/env python3
import argparse
import os
import shutil
import zipfile
from pathlib import Path

import gdown

# Carpetas públicas de Drive con la data del proyecto
DRIVE_FOLDERS = {
    "images": "1aouL8-2iWVodDhuWDUmbEQDKPdC6Eltx",
    "images_json": "1_tjGy6Qnvc6eR5uNTSQnOoy9SlBTbsaZ",
    "lyrics": "12l9jUkSr8lgqkntGT4RNp5_9jg753gSm",
    "songs": "1ju98Jk-_tIuPFJROTgyG_YT7u9Ggdwft",
}

# gdown solo lista hasta 50 archivos por carpeta de Drive
GDOWN_FOLDER_LIMIT = 50

# Raíz local donde queda la data descargada
DATA_ROOT = Path(
    os.environ.get("DATA_DIR", str(Path(__file__).resolve().parents[2] / "data" / "raw"))
)


# Cuenta los archivos ya bajados sin contar el marcador de git
def _file_count(target: Path) -> int:
    if not target.is_dir():
        return 0
    return sum(1 for entry in target.rglob("*") if entry.is_file() and entry.name != ".gitkeep")


# Extrae cada zip descargado y luego lo borra
def _extract_zips(target: Path) -> None:
    for zip_path in sorted(target.rglob("*.zip")):
        print(f"extrayendo {zip_path.name}...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(target)
        zip_path.unlink()
    _remove_mac_junk(target)
    _flatten(target)


# Borra los archivos ocultos que dejan los zip hechos en Mac
def _remove_mac_junk(target: Path) -> None:
    for junk in list(target.rglob("__MACOSX")):
        shutil.rmtree(junk)
    for hidden in list(target.rglob("._*")):
        if hidden.is_file():
            hidden.unlink()


# Sube los archivos un nivel cuando el zip trae una carpeta adentro
def _flatten(target: Path) -> None:
    while True:
        entries = [e for e in target.iterdir() if e.name != ".gitkeep"]
        if len(entries) != 1 or not entries[0].is_dir():
            return
        inner = entries[0]
        for item in inner.iterdir():
            shutil.move(str(item), str(target / item.name))
        inner.rmdir()


# Baja una carpeta de Drive a su destino local
def download_folder(name: str, folder_id: str, force: bool = False) -> Path:
    target = DATA_ROOT / name
    existing = _file_count(target)
    if existing > 0 and not force:
        print(f"{name}: {existing} archivos ya en {target}, se omite (usa --force)")
        return target
    target.mkdir(parents=True, exist_ok=True)
    gdown.download_folder(
        id=folder_id,
        output=str(target),
        quiet=False,
        use_cookies=False,
    )
    _extract_zips(target)
    total = _file_count(target)
    print(f"{name}: {total} archivos en {target}")
    if total >= GDOWN_FOLDER_LIMIT:
        print(f"{name}: aviso, gdown corta en {GDOWN_FOLDER_LIMIT} archivos, conviene un zip por carpeta")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description="Descarga la data del proyecto desde Drive")
    parser.add_argument(
        "--only",
        nargs="+",
        choices=sorted(DRIVE_FOLDERS),
        help="descarga solo estas carpetas",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="baja de nuevo aunque ya existan archivos",
    )
    args = parser.parse_args()
    names = args.only if args.only else list(DRIVE_FOLDERS)
    for name in names:
        download_folder(name, DRIVE_FOLDERS[name], force=args.force)


if __name__ == "__main__":
    main()
