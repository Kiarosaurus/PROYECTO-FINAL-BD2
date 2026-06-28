from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Request, UploadFile

router = APIRouter()


# Recibe un archivo y lo guarda en la carpeta de subidas
@router.post("/upload")
async def upload_file(request: Request, file: UploadFile = File(...)) -> dict[str, object]:
    name = Path(file.filename or "").name
    if not name:
        raise HTTPException(status_code=400, detail="nombre de archivo vacío")
    base = Path(request.app.state.upload_dir)
    data = await file.read()
    (base / name).write_bytes(data)
    return {"filename": name, "size": len(data)}
