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
from core.buffer.lru_buffer import LRUBufferManager
from core.ports.storage import StorageEngine
from core.storage.file_engine import FileStorageEngine
from query.parser.sql_parser import SqlParser
from query.planner import QueryPlanner
from query.executor import QueryExecutor
from query.index_factory import EngineIndexFactory
from service.catalog import Catalog
from service.session import rehydrate_executor

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


# Arma el resolver multimedia con las modalidades disponibles
# El storage recibido guarda el codebook de cada modalidad
def _build_media_resolver(media_dir: Path, storage: StorageEngine):
    try:
        from multimedia.codebook.kmeans_codebook import KMeansCodebook
        from multimedia.resolver import CompositeMediaResolver, PipelineMediaResolver
    except ImportError:
        return None
    resolvers = []
    # Cada modalidad entra solo si sus librerías están disponibles
    try:
        from multimedia.extractors.sift_extractor import SIFTExtractor

        resolvers.append(
            PipelineMediaResolver(
                SIFTExtractor(),
                KMeansCodebook(k=32, file_id="codebook_image"),
                media_dir,
                storage=storage,
            )
        )
    except ImportError:
        pass
    try:
        from multimedia.extractors.mfcc_extractor import MFCCExtractor

        resolvers.append(
            PipelineMediaResolver(
                MFCCExtractor(sample_rate=8000),
                KMeansCodebook(k=32, file_id="codebook_audio"),
                media_dir,
                storage=storage,
            )
        )
    except ImportError:
        pass
    if not resolvers:
        return None
    return CompositeMediaResolver(resolvers)


# Elige el medio de almacenamiento según el entorno
def _build_storage() -> StorageEngine:
    backend = os.environ.get("STORAGE_BACKEND", "file").lower()
    if backend == "postgres":
        from core.storage.postgres_engine import PostgresStorageEngine

        return PostgresStorageEngine(os.environ["POSTGRES_DSN"])
    return FileStorageEngine(os.environ.get("ENGINE_DATA_DIR", "engine_data"))


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
    upload_dir = Path("uploads")
    upload_dir.mkdir(exist_ok=True)
    app.state.upload_dir = upload_dir
    app.state.parser = SqlParser()
    app.state.planner = QueryPlanner()
    # Un solo storage compartido por índices y codebooks
    storage = _build_storage()
    factory = EngineIndexFactory(media_resolver=_build_media_resolver(upload_dir, storage))
    app.state.executor = QueryExecutor(factory, storage)
    # El catálogo guarda sus tablas en el mismo storage del engine
    app.state.catalog = Catalog(buffer=LRUBufferManager(storage))
    # Al arrancar se reponen las tablas y los índices ya creados
    rehydrate_executor(app.state.executor, app.state.catalog)

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
