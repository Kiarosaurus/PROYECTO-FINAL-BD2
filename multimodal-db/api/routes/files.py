from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

router = APIRouter()


# Sirve un archivo subido sin dejar salir de la carpeta
@router.get("/files/{name}")
def serve_file(name: str, request: Request) -> FileResponse:
    base = Path(request.app.state.upload_dir).resolve()
    target = (base / name).resolve()
    try:
        target.relative_to(base)
    except ValueError:
        raise HTTPException(status_code=400, detail="ruta no permitida")
    if not target.is_file():
        raise HTTPException(status_code=404, detail="archivo no encontrado")
    return FileResponse(target)
