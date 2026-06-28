from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from api.deps import get_session
from api.models import ExplainLine, IOStatsModel, QueryRequest, QueryResponse
from service.session import Session

router = APIRouter()


# Corre una consulta SQL y devuelve sus resultados
@router.post("/query", response_model=QueryResponse)
def run_query(req: QueryRequest, session: Session = Depends(get_session)) -> QueryResponse:
    try:
        result = session.execute(req.sql)
    except Exception as error:
        raise HTTPException(status_code=400, detail=str(error))
    return QueryResponse(
        columns=result.columns,
        rows=[list(row) for row in result.rows],
        io=IOStatsModel(
            disk_reads=result.io.disk_reads,
            disk_writes=result.io.disk_writes,
            pages_allocated=result.io.pages_allocated,
        ),
        index_type=result.index_type,
        predicate_kind=result.predicate_kind,
        elapsed_ms=result.elapsed_ms,
        explain=[ExplainLine(depth=depth, text=text) for depth, text in result.explain],
    )
