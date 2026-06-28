from __future__ import annotations

import importlib
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from api.models import ErrorResponse
from query.parser.sql_parser import SqlParser
from query.planner import QueryPlanner
from query.executor import QueryExecutor
from tests.mocks import MockIndexFactory, MockStorageEngine

# Rutas que se cargan solo si su módulo está presente
_ROUTE_MODULES = [
    "api.routes.query",
    "api.routes.upload",
    "api.routes.files",
]

# Orígenes del frontend permitidos para llamar a la API
_DEFAULT_ORIGINS = "http://localhost:3000,http://127.0.0.1:3000"


# Lee los orígenes permitidos del entorno o usa los del frontend local
def _allowed_origins() -> list[str]:
    raw = os.environ.get("CORS_ORIGINS", _DEFAULT_ORIGINS)
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


# Nombre corto de error para cada código de estado
_ERROR_LABELS = {
    400: "bad_request",
    404: "not_found",
    422: "validation_error",
    500: "internal_error",
}


# Devuelve cualquier error con el mismo formato
def _error_response(status_code: int, detail: str) -> JSONResponse:
    label = _ERROR_LABELS.get(status_code, "error")
    body = ErrorResponse(error=label, detail=detail)
    return JSONResponse(status_code=status_code, content=body.model_dump())


# Arma la app y conecta el motor de consultas
def create_app() -> FastAPI:
    app = FastAPI(title="Multimodal DB")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_allowed_origins(),
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.parser = SqlParser()
    app.state.planner = QueryPlanner()
    app.state.executor = QueryExecutor(MockIndexFactory(), MockStorageEngine())
    app.state.upload_dir = Path("uploads")
    app.state.upload_dir.mkdir(exist_ok=True)

    @app.exception_handler(StarletteHTTPException)
    async def on_http_error(request, exc: StarletteHTTPException) -> JSONResponse:
        return _error_response(exc.status_code, str(exc.detail))

    @app.exception_handler(RequestValidationError)
    async def on_validation_error(request, exc: RequestValidationError) -> JSONResponse:
        return _error_response(422, str(exc.errors()))

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    for name in _ROUTE_MODULES:
        try:
            module = importlib.import_module(name)
        except ModuleNotFoundError:
            continue
        app.include_router(module.router)

    return app


app = create_app()
