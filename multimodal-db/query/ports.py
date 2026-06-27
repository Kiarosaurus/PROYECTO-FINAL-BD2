from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from query.plan_types import QueryPlan, ResultSet

# SQL ya leído y armado como árbol
AST = Any
# Lista de tablas e índices que existen
Catalog = Any


# Lee el texto SQL y lo arma como árbol
class Parser(ABC):

    @abstractmethod
    def parse(self, sql: str) -> AST:
        ...


# Arma el plan a partir del árbol
class Planner(ABC):

    @abstractmethod
    def plan(self, ast: AST, catalog: Catalog) -> QueryPlan:
        ...


# Ejecuta el plan y devuelve los resultados
class Executor(ABC):

    @abstractmethod
    def execute(self, plan: QueryPlan) -> ResultSet:
        ...
