from __future__ import annotations

import importlib
from pathlib import Path

from fastapi import FastAPI

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


# Arma la app y conecta el motor de consultas
def create_app() -> FastAPI:
    app = FastAPI(title="Multimodal DB")
    app.state.parser = SqlParser()
    app.state.planner = QueryPlanner()
    app.state.executor = QueryExecutor(MockIndexFactory(), MockStorageEngine())
    app.state.upload_dir = Path("uploads")
    app.state.upload_dir.mkdir(exist_ok=True)

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
