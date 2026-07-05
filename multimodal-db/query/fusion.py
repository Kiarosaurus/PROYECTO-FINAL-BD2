from __future__ import annotations

from abc import ABC, abstractmethod

# Un ranking es una lista de pares clave y score ordenada de mayor a menor
Ranking = list[tuple[str, float]]


# Combina varios rankings en un solo ranking final
class RankFusion(ABC):

    @abstractmethod
    def fuse(self, rankings: list[Ranking], k: int | None = None) -> Ranking:
        ...


# Ordena por score descendente y desempata por clave
def _sorted_top(scores: dict[str, float], k: int | None) -> Ranking:
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    if k is not None:
        return ranked[:k]
    return ranked


# Fusión por posición, cada aparición aporta 1 / (k0 + posición)
class ReciprocalRankFusion(RankFusion):

    def __init__(self, k0: int = 60) -> None:
        if k0 < 1:
            raise ValueError("k0 debe ser positivo")
        self._k0 = k0

    def fuse(self, rankings: list[Ranking], k: int | None = None) -> Ranking:
        scores: dict[str, float] = {}
        for ranking in rankings:
            for position, (key, _score) in enumerate(ranking, start=1):
                scores[key] = scores.get(key, 0.0) + 1.0 / (self._k0 + position)
        return _sorted_top(scores, k)


# Lleva los scores de un ranking al rango entre 0 y 1
def _min_max(ranking: Ranking) -> Ranking:
    if not ranking:
        return []
    values = [score for _key, score in ranking]
    low, high = min(values), max(values)
    # Con todos los scores iguales la presencia vale el máximo
    if high == low:
        return [(key, 1.0) for key, _score in ranking]
    return [(key, (score - low) / (high - low)) for key, score in ranking]


# Suma ponderada de dos rankings normalizados a la misma escala
class WeightedSumFusion(RankFusion):

    # alpha es el peso del primer ranking y el resto pesa al segundo
    def __init__(self, alpha: float = 0.5) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha debe estar entre 0 y 1")
        self._alpha = alpha

    def fuse(self, rankings: list[Ranking], k: int | None = None) -> Ranking:
        if len(rankings) != 2:
            raise ValueError("la suma ponderada espera exactamente dos rankings")
        weights = (self._alpha, 1.0 - self._alpha)
        scores: dict[str, float] = {}
        for ranking, weight in zip(rankings, weights):
            for key, score in _min_max(ranking):
                scores[key] = scores.get(key, 0.0) + weight * score
        return _sorted_top(scores, k)
