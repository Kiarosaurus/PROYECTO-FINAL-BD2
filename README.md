# Proyecto 2: Base de Datos 2

Integrantes:
- Kiara Alexandra Balcázar Santa Cruz

## Arquitectura

Interfaces (ABCs) en `multimodal-db/`:
- `core/ports/`: `StorageEngine`, `BufferManager` e `Index`.
- `core/metrics.py`: `IOStats` y `OperationResult`.
- `indices/ports.py`: tipos de `Predicate`.
- `multimedia/ports/`: `FeatureExtractor` y `Codebook`.
- `query/ports.py`: `Parser`, `Planner` y `Executor`.
- `query/plan_types.py`: `QueryPlan` y `ResultSet`.
- `query/index_factory.py`: `IndexFactory` y `IndexType`.

`tests/mocks.py` trae una versión falsa de cada interface para las pruebas.

## Pipeline de consultas

El SQL pasa por tres etapas:
- `Parser` (lark) convierte el texto en nodos AST.
- `Planner` arma un `QueryPlan` y elige el índice según el predicado.
- `Executor` corre el plan y devuelve un `ResultSet` con `IOStats`.

SQL soportado:

```sql
CREATE TABLE img (id INT, path TEXT, feat VECTOR)
CREATE INDEX ON img (feat) USING rtree
INSERT INTO img (id, path) VALUES (1, "a.jpg")
DELETE FROM img WHERE id = 5
SELECT * FROM img WHERE id BETWEEN 1 AND 9 LIMIT 10
SELECT * FROM img WHERE KNN(feat, [0.1, 0.2, 0.3], 5)
SELECT * FROM img WHERE WITHIN(box, [0, 0], [10, 10])
```

Cada predicado elige su índice: igualdad usa hash, rango usa bplus, KNN usa knn,
espacial usa rtree y texto usa inverted.

## Service

- `service/catalog.py`: `Catalog` guarda las tablas y sus índices.
- `service/session.py`: `Session` atiende una conexión y acumula `IOStats`.
- `service/dto.py`: `Schema`, `TableInfo`, `ColumnSpec` e `IndexInfo`.

## API

FastAPI en `api/`:
- `GET /health`: estado del servicio.
- `POST /query`: corre una consulta SQL.
- `POST /upload`: sube un archivo multimedia.
- `GET /files/{name}`: sirve un archivo subido sin salir de la carpeta.

Los errores salen siempre con el formato `{error, detail}`. El executor usa
índices mock mientras se conectan los reales.

## Entorno

Python 3.12:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Las dependencias están fijadas en `requirements.txt`.
