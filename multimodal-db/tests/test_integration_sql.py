from __future__ import annotations

import wave
from pathlib import Path

import cv2
import numpy as np
import pytest

from core.storage.file_engine import FileStorageEngine
from query.executor import QueryExecutor
from query.index_factory import EngineIndexFactory
from query.parser.sql_parser import SqlParser
from query.planner import QueryPlanner
from service.catalog import Catalog
from service.session import Session


@pytest.fixture
def run(tmp_path: Path):
    parser = SqlParser()
    planner = QueryPlanner()
    executor = QueryExecutor(EngineIndexFactory(), FileStorageEngine(tmp_path))

    def _run(sql: str):
        return executor.execute(planner.plan(parser.parse(sql)))

    return _run


def test_sql_pipeline_with_real_hash_index_and_file_storage(run) -> None:
    run("CREATE TABLE img (id INT, path TEXT)")
    run("CREATE INDEX ON img (id) USING hash")
    insert = run('INSERT INTO img (id, path) VALUES (1, "a.jpg"), (2, "b.jpg")')
    found = run("SELECT path FROM img WHERE id = 2")

    assert insert.rows == [(2,)]
    assert found.rows == [("b.jpg",)]
    assert insert.io.disk_writes > 0


def test_sql_range_query_uses_real_bplus_index(run) -> None:
    run("CREATE TABLE img (id INT, path TEXT)")
    run("CREATE INDEX ON img (id) USING bplus")
    run('INSERT INTO img (id, path) VALUES (1, "a.jpg"), (2, "b.jpg"), (3, "c.jpg")')

    result = run("SELECT id FROM img WHERE id BETWEEN 2 AND 3")

    assert result.rows == [(2,), (3,)]
    assert result.index_type == "bplus"
    assert result.predicate_kind == "range"


def test_sql_select_without_where_scans_real_index(run) -> None:
    run("CREATE TABLE img (id INT, path TEXT)")
    run("CREATE INDEX ON img (id) USING bplus")
    run('INSERT INTO img (id, path) VALUES (2, "b.jpg"), (1, "a.jpg")')

    result = run("SELECT id FROM img")

    assert result.rows == [(1,), (2,)]


def test_sql_spatial_query_uses_real_rtree(run) -> None:
    run("CREATE TABLE places (id INT, point VECTOR)")
    run("CREATE INDEX ON places (point) USING rtree")
    executor_result = run("SELECT id FROM places WHERE WITHIN(point, [0, 0], [10, 10])")

    assert executor_result.rows == []
    assert executor_result.index_type == "rtree"


def test_session_with_catalog_makes_planner_respect_created_index(tmp_path: Path) -> None:
    executor = QueryExecutor(EngineIndexFactory(), FileStorageEngine(tmp_path))
    session = Session(SqlParser(), QueryPlanner(), executor, Catalog())

    session.execute("CREATE TABLE img (id INT, path TEXT)")
    session.execute("CREATE INDEX ON img (id) USING bplus")
    session.execute('INSERT INTO img (id, path) VALUES (1, "a.jpg")')
    result = session.execute("SELECT path FROM img WHERE id = 1")

    assert result.rows == [("a.jpg",)]
    # Sin catálogo el planner sugiere hash, con catálogo respeta el bplus creado
    assert result.index_type == "bplus"


def test_sql_text_match_ranks_documents_by_relevance(tmp_path: Path) -> None:
    executor = QueryExecutor(EngineIndexFactory(), FileStorageEngine(tmp_path))
    session = Session(SqlParser(), QueryPlanner(), executor, Catalog())

    session.execute("CREATE TABLE docs (id INT, body TEXT)")
    session.execute("CREATE INDEX ON docs (body) USING inverted")
    session.execute(
        'INSERT INTO docs (id, body) VALUES '
        '(1, "visual search database"), '
        '(2, "audio retrieval engine"), '
        '(3, "visual retrieval engine"), '
        '(4, "database storage engine")'
    )
    result = session.execute('SELECT id FROM docs WHERE MATCH(body, "visual search", 3)')

    # El documento con ambos términos queda primero
    assert result.rows == [(1,), (3,)]
    assert result.index_type == "inverted"
    assert result.predicate_kind == "text_match"


def test_sql_create_index_with_vocabulary_prunes_rare_terms(tmp_path: Path) -> None:
    executor = QueryExecutor(EngineIndexFactory(), FileStorageEngine(tmp_path))
    session = Session(SqlParser(), QueryPlanner(), executor, Catalog())

    session.execute("CREATE TABLE docs (id INT, body TEXT)")
    session.execute("CREATE INDEX ON docs (body) USING inverted WITH (vocabulary = 2)")
    session.execute(
        'INSERT INTO docs (id, body) VALUES '
        '(1, "alpha alpha beta"), '
        '(2, "alpha beta"), '
        '(3, "gamma delta gamma")'
    )
    pruned = session.execute('SELECT id FROM docs WHERE MATCH(body, "gamma", 3)')
    kept = session.execute('SELECT id FROM docs WHERE MATCH(body, "alpha", 3)')

    # Solo alpha y beta sobreviven al límite de vocabulario
    assert pruned.rows == []
    assert {row[0] for row in kept.rows} == {1, 2}


@pytest.mark.parametrize("value", ['0', '-3', '2.5', '"abc"'])
def test_sql_create_index_with_invalid_vocabulary_raises(run, value) -> None:
    run("CREATE TABLE docs (id INT, body TEXT)")

    with pytest.raises(ValueError, match="vocabulary"):
        run(f"CREATE INDEX ON docs (body) USING inverted WITH (vocabulary = {value})")


def _write_noise_image(path: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 255, size=(64, 64), dtype=np.uint8)
    cv2.imwrite(str(path), image)


def test_sql_knn_by_file_uses_media_resolver(tmp_path: Path) -> None:
    from multimedia.codebook.kmeans_codebook import KMeansCodebook
    from multimedia.extractors.sift_extractor import SIFTExtractor
    from multimedia.resolver import PipelineMediaResolver

    media_dir = tmp_path / "uploads"
    media_dir.mkdir()
    _write_noise_image(media_dir / "a.png", seed=1)
    _write_noise_image(media_dir / "b.png", seed=2)
    _write_noise_image(media_dir / "c.png", seed=3)
    # La consulta es una copia exacta de a.png para fijar el primer resultado
    (media_dir / "query.png").write_bytes((media_dir / "a.png").read_bytes())

    resolver = PipelineMediaResolver(SIFTExtractor(), KMeansCodebook(k=4), media_dir)
    executor = QueryExecutor(
        EngineIndexFactory(media_resolver=resolver),
        FileStorageEngine(tmp_path / "storage"),
    )
    parser, planner = SqlParser(), QueryPlanner()

    def run(sql: str):
        return executor.execute(planner.plan(parser.parse(sql)))

    run("CREATE TABLE media (id INT, feat VECTOR)")
    run("CREATE INDEX ON media (feat) USING knn")
    run('INSERT INTO media (id, feat) VALUES (1, "a.png"), (2, "b.png"), (3, "c.png")')
    result = run('SELECT * FROM media WHERE KNN(feat, "query.png", 3)')

    hits = [record for (record,) in result.rows]
    scores = [score for _key, score in hits]
    # El filtro de candidatos puede descartar imágenes sin visual words comunes
    assert len(hits) >= 2
    assert hits[0][0] == "a.png"
    assert scores == sorted(scores, reverse=True)
    assert {key for key, _score in hits} <= {"a.png", "b.png", "c.png"}
    assert result.index_type == "knn"


def _write_sine_wav(path: Path, freq: float) -> None:
    rate = 8000
    t = np.arange(rate * 2) / rate
    signal = (12000 * np.sin(2 * np.pi * freq * t)).astype(np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(signal.tobytes())


def test_sql_knn_audio_by_file_routes_to_mfcc_resolver(tmp_path: Path) -> None:
    from multimedia.codebook.kmeans_codebook import KMeansCodebook
    from multimedia.extractors.mfcc_extractor import MFCCExtractor
    from multimedia.extractors.sift_extractor import SIFTExtractor
    from multimedia.resolver import CompositeMediaResolver, PipelineMediaResolver

    media_dir = tmp_path / "uploads"
    media_dir.mkdir()
    _write_sine_wav(media_dir / "low.wav", freq=220.0)
    _write_sine_wav(media_dir / "mid.wav", freq=880.0)
    _write_sine_wav(media_dir / "high.wav", freq=3520.0)
    # La consulta usa una frecuencia casi igual a la del tono grave
    _write_sine_wav(media_dir / "query.wav", freq=230.0)

    resolver = CompositeMediaResolver(
        [
            PipelineMediaResolver(SIFTExtractor(), KMeansCodebook(k=4), media_dir),
            PipelineMediaResolver(MFCCExtractor(), KMeansCodebook(k=4), media_dir),
        ]
    )
    executor = QueryExecutor(
        EngineIndexFactory(media_resolver=resolver),
        FileStorageEngine(tmp_path / "storage"),
    )
    parser, planner = SqlParser(), QueryPlanner()

    def run(sql: str):
        return executor.execute(planner.plan(parser.parse(sql)))

    run("CREATE TABLE tracks (id INT, feat VECTOR)")
    run("CREATE INDEX ON tracks (feat) USING knn")
    run('INSERT INTO tracks (id, feat) VALUES (1, "low.wav"), (2, "mid.wav"), (3, "high.wav")')
    result = run('SELECT * FROM tracks WHERE KNN(feat, "query.wav", 2)')

    hits = [record for (record,) in result.rows]
    scores = [score for _key, score in hits]
    assert len(hits) == 2
    assert hits[0][0] == "low.wav"
    assert scores == sorted(scores, reverse=True)
    assert result.index_type == "knn"


def test_sql_knn_by_file_without_resolver_returns_empty(run) -> None:
    run("CREATE TABLE media (id INT, feat VECTOR)")
    run("CREATE INDEX ON media (feat) USING knn")

    result = run('SELECT * FROM media WHERE KNN(feat, "query.png", 3)')

    assert result.rows == []


def test_sql_delete_removes_from_real_index(run) -> None:
    run("CREATE TABLE img (id INT, path TEXT)")
    run("CREATE INDEX ON img (id) USING hash")
    run('INSERT INTO img (id, path) VALUES (1, "a.jpg"), (2, "b.jpg")')

    deleted = run("DELETE FROM img WHERE id = 1")
    remaining = run("SELECT id FROM img")

    assert deleted.rows == [(1,)]
    assert remaining.rows == [(2,)]


@pytest.fixture
def hybrid_run(tmp_path: Path):
    from tests.mocks import MockMediaResolver

    # Parecido visual decreciente hacia la consulta [1, 0, 0, 0]
    resolver = MockMediaResolver(
        table={
            "q.png": [1.0, 0.0, 0.0, 0.0],
            "a.png": [1.0, 0.0, 0.0, 0.0],
            "b.png": [1.0, 0.5, 0.0, 0.0],
            "c.png": [1.0, 1.0, 0.0, 0.0],
            "d.png": [1.0, 2.0, 0.0, 0.0],
        }
    )
    executor = QueryExecutor(
        EngineIndexFactory(media_resolver=resolver),
        FileStorageEngine(tmp_path / "storage"),
    )
    parser, planner = SqlParser(), QueryPlanner()

    def _run(sql: str):
        return executor.execute(planner.plan(parser.parse(sql)))

    _run("CREATE TABLE tracks (id INT, feat VECTOR, lyrics TEXT)")
    _run("CREATE INDEX ON tracks (feat) USING knn")
    _run("CREATE INDEX ON tracks (lyrics) USING inverted")
    _run(
        'INSERT INTO tracks (id, feat, lyrics) VALUES '
        '(1, "a.png", "calm morning fog"), '
        '(2, "b.png", "love night moon"), '
        '(3, "c.png", "love moon river"), '
        '(4, "d.png", "love night love night love")'
    )
    return _run


def test_sql_hybrid_fusion_winner_leads_neither_branch(hybrid_run) -> None:
    visual = hybrid_run('SELECT * FROM tracks WHERE KNN(feat, "q.png", 4)')
    textual = hybrid_run('SELECT id FROM tracks WHERE MATCH(lyrics, "love night", 4)')
    fused = hybrid_run(
        'SELECT * FROM tracks WHERE HYBRID(feat, "q.png", lyrics, "love night", 3)'
    )

    # En cada rama individual gana un documento distinto de b
    visual_hits = [record for (record,) in visual.rows]
    assert visual_hits[0][0] == "a.png"
    assert textual.rows[0] == (4,)
    # La fusión corona al documento que era segundo en ambas ramas
    assert fused.columns == ["id", "feat", "lyrics", "fused_score", "visual_score", "text_score"]
    feats = [row[1] for row in fused.rows]
    assert feats == ["b.png", "d.png", "c.png"]
    assert fused.rows[0][:3] == (2, "b.png", "love night moon")
    assert fused.index_type == "hybrid"
    assert fused.predicate_kind == "hybrid"


def test_sql_hybrid_exposes_branch_scores(hybrid_run) -> None:
    result = hybrid_run(
        'SELECT * FROM tracks WHERE HYBRID(feat, "q.png", lyrics, "love night", 3)'
    )

    winner = result.rows[0]
    fused_score, visual_score, text_score = winner[3], winner[4], winner[5]
    assert fused_score > 0.0
    # El ganador aparece en ambas ramas con sus scores de origen
    assert visual_score is not None and 0.0 < visual_score <= 1.0
    assert text_score is not None and 0.0 < text_score <= 1.0
    scores = [row[3] for row in result.rows]
    assert scores == sorted(scores, reverse=True)


def test_sql_hybrid_explain_shows_both_branches(hybrid_run) -> None:
    result = hybrid_run(
        'SELECT * FROM tracks WHERE HYBRID(feat, "q.png", lyrics, "love night", 3)'
    )

    texts = [text for _depth, text in result.explain]
    assert any("Hybrid Fusion Scan using hybrid" in text for text in texts)
    branches = [text for text in texts if text.startswith("Branch:")]
    assert len(branches) == 2
    assert "KNN Index Scan on feat" in branches[0]
    assert "Text Search Scan on lyrics" in branches[1]


def test_sql_hybrid_missing_text_index_raises(run) -> None:
    run("CREATE TABLE tracks (id INT, feat VECTOR, lyrics TEXT)")
    run("CREATE INDEX ON tracks (feat) USING knn")

    with pytest.raises(ValueError, match="tracks.lyrics"):
        run('SELECT * FROM tracks WHERE HYBRID(feat, "q.png", lyrics, "love", 3)')


def test_sql_hybrid_missing_media_index_raises(run) -> None:
    run("CREATE TABLE tracks (id INT, feat VECTOR, lyrics TEXT)")
    run("CREATE INDEX ON tracks (lyrics) USING inverted")

    with pytest.raises(ValueError, match="tracks.feat"):
        run('SELECT * FROM tracks WHERE HYBRID(feat, "q.png", lyrics, "love", 3)')
