#!/usr/bin/env python3
import argparse
import hashlib
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

# Carpetas de Drive con los dos zip oficiales de FMA small
FMA_DRIVE_FOLDERS = {
    "metadata": "1xi4mK84JYWfB9kiTZSRpFNcnJdB2DeO4",
    "audio": "1KhBBLI4_gHDzStxu7_Wn0cRzeRM9fel4",
}

# Huellas oficiales para confirmar que los zip llegaron completos
FMA_SHA1 = {
    "fma_metadata.zip": "f0df49ffe5f2a6008d7dc83c6915b31835dfe733",
    "fma_small.zip": "ade154f733639d52e35e32f5593efe5be76c6d70",
}

# gdown solo lista hasta 50 archivos por carpeta de Drive
GDOWN_FOLDER_LIMIT = 50

# Raíz local donde queda la data descargada
DATA_ROOT = Path(
    os.environ.get("DATA_DIR", str(Path(__file__).resolve().parents[2] / "data"))
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


# Calcula la huella sha1 de un archivo leyendo por partes
def _sha1(path: Path) -> str:
    digest = hashlib.sha1()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# Revisa cada zip de FMA contra su huella conocida
def _verify_fma_zips(workdir: Path) -> None:
    for name, expected in FMA_SHA1.items():
        zip_path = workdir / name
        if not zip_path.is_file():
            raise FileNotFoundError(f"no se descargó {name} desde Drive, revisa la carpeta compartida")
        actual = _sha1(zip_path)
        if actual != expected:
            raise ValueError(f"{name}: sha1 {actual} no coincide con el oficial {expected}")
        print(f"{name}: sha1 verificado")


# Saca solo tracks.csv del zip de metadata
def _extract_fma_tracks(zip_path: Path, target: Path) -> None:
    with zipfile.ZipFile(zip_path) as zf:
        member = next(
            name
            for name in zf.namelist()
            if name.endswith("tracks.csv") and "._" not in name
        )
        with zf.open(member) as source, open(target / "tracks.csv", "wb") as sink:
            shutil.copyfileobj(source, sink)


# Extrae los clips y los deja bajo la carpeta audio
def _extract_fma_audio(zip_path: Path, target: Path) -> None:
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(target)
    _remove_mac_junk(target)
    audio = target / "audio"
    if audio.exists():
        shutil.rmtree(audio)
    (target / "fma_small").rename(audio)


# Baja los dos zip de FMA desde Drive y arma data/fma
def download_fma(force: bool = False) -> Path:
    target = DATA_ROOT / "fma"
    audio = target / "audio"
    if (target / "tracks.csv").is_file() and _file_count(audio) > 0 and not force:
        print(f"fma: data ya en {target}, se omite (usa --force)")
        return target
    target.mkdir(parents=True, exist_ok=True)
    workdir = target / "_zips"
    workdir.mkdir(exist_ok=True)
    for folder_id in FMA_DRIVE_FOLDERS.values():
        gdown.download_folder(
            id=folder_id,
            output=str(workdir),
            quiet=False,
            use_cookies=False,
        )
    _verify_fma_zips(workdir)
    _extract_fma_tracks(workdir / "fma_metadata.zip", target)
    (workdir / "fma_metadata.zip").unlink()
    _extract_fma_audio(workdir / "fma_small.zip", target)
    (workdir / "fma_small.zip").unlink()
    workdir.rmdir()
    print(f"fma: tracks.csv y {_file_count(audio)} archivos de audio en {target}")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description="Descarga la data del proyecto desde Drive")
    parser.add_argument(
        "--only",
        nargs="+",
        choices=sorted(DRIVE_FOLDERS) + ["fma"],
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
        if name == "fma":
            download_fma(force=args.force)
        else:
            download_folder(name, DRIVE_FOLDERS[name], force=args.force)
    if not args.only:
        print("fma: no se baja por defecto (7.2 GiB), corre con --only fma para el audio real")


if __name__ == "__main__":
    main()
