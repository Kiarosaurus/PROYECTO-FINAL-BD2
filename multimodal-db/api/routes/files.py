from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, Response

router = APIRouter()


# Dibuja una imagen plantilla cuando el archivo no fue subido
def _template_svg(name: str) -> str:
    label = escape(name)
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="320" height="240">'
        '<rect width="320" height="240" fill="#161a22" stroke="#262c38"/>'
        '<text x="160" y="112" fill="#c7d0dd" font-family="sans-serif"'
        ' font-size="18" text-anchor="middle">' + label + "</text>"
        '<text x="160" y="140" fill="#6b7280" font-family="sans-serif"'
        ' font-size="13" text-anchor="middle">plantilla demo (sin archivo)</text>'
        "</svg>"
    )


# Sirve un archivo subido sin dejar salir de la carpeta
@router.get("/files/{name}")
def serve_file(name: str, request: Request) -> Response:
    base = Path(request.app.state.upload_dir).resolve()
    target = (base / name).resolve()
    try:
        target.relative_to(base)
    except ValueError:
        raise HTTPException(status_code=400, detail="ruta no permitida")
    if not target.is_file():
        return Response(content=_template_svg(name), media_type="image/svg+xml")
    return FileResponse(target)
