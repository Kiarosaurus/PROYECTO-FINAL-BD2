from __future__ import annotations

from fastapi import Request

from service.session import Session


# Arma una Session nueva para cada pedido
def get_session(request: Request) -> Session:
    state = request.app.state
    return Session(state.parser, state.planner, state.executor, state.catalog)
