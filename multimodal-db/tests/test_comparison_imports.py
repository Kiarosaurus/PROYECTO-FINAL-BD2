from __future__ import annotations

from comparison.pgvector_ivf import PgVectorIVFEngine
from comparison.ports import BenchmarkResult
from comparison.postgres_gin import PostgresGINEngine

DSN = "postgresql://mmdb:mmdb@localhost:5432/multimodal"


def test_engines_construct_without_postgres_server() -> None:
    # El constructor no abre conexión, así que no requiere servidor
    gin = PostgresGINEngine(DSN)
    ivf = PgVectorIVFEngine(DSN, modality="IMAGE")
    assert isinstance(gin, PostgresGINEngine)
    assert isinstance(ivf, PgVectorIVFEngine)


def test_benchmark_result_defaults() -> None:
    result = BenchmarkResult()
    assert result.records == []
    assert result.latency_ms == 0.0
    assert not hasattr(result, "memory_bytes")
