from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Sequence, Union


# Tipos de búsqueda disponibles
class PredicateKind(Enum):
    EQUALITY = auto()
    RANGE = auto()
    KNN = auto()
    SPATIAL_RANGE = auto()
    TEXT_MATCH = auto()


# Datos básicos de cualquier búsqueda
@dataclass(frozen=True)
class Predicate:
    column: str


# Busca un valor exacto
@dataclass(frozen=True)
class EqualityPredicate(Predicate):
    value: Any
    kind: PredicateKind = PredicateKind.EQUALITY


# Busca valores entre un mínimo y un máximo
@dataclass(frozen=True)
class RangePredicate(Predicate):
    low: Any
    high: Any
    include_low: bool = True
    include_high: bool = True
    kind: PredicateKind = PredicateKind.RANGE


# Busca los más parecidos a un vector
@dataclass(frozen=True)
class KnnPredicate(Predicate):
    query: Any
    k: int
    kind: PredicateKind = PredicateKind.KNN


# Busca dentro de un área en el mapa
@dataclass(frozen=True)
class SpatialRangePredicate(Predicate):
    min_corner: Sequence[float]
    max_corner: Sequence[float]
    kind: PredicateKind = PredicateKind.SPATIAL_RANGE


# Busca texto y ordena por parecido
@dataclass(frozen=True)
class TextMatchPredicate(Predicate):
    terms: str
    k: int | None = None
    kind: PredicateKind = PredicateKind.TEXT_MATCH


# Todas las búsquedas que acepta Index.search
SearchPredicate = Union[
    EqualityPredicate,
    RangePredicate,
    KnnPredicate,
    SpatialRangePredicate,
    TextMatchPredicate,
]
