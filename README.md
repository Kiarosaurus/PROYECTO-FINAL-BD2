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

## Frontend

Next.js en `frontend/`:
- Editor de queries que llama a `POST /query`.
- Tabla con los resultados.
- Galería para imágenes y reproductor para audio, servidos desde `/files/{name}`.

## Cómo correr

Hay dos formas: todo en Docker o cada parte en local.

### Con Docker

Levanta postgres (pgvector), backend y frontend con un solo comando desde la
raíz del repo:

```bash
docker compose up --build
```

Quedan expuestos:
- Frontend: `http://localhost:3000`
- Backend: `http://localhost:8000` (health en `/health`)
- Postgres: `localhost:5432`

Variables opcionales (con sus valores por defecto):
- `STORAGE_BACKEND=file`: elige el `StorageEngine` (`file` o `postgres`).
- `POSTGRES_USER=mmdb`, `POSTGRES_PASSWORD=mmdb`, `POSTGRES_DB=multimodal`.

Para apagar todo:

```bash
docker compose down
```

Para borrar también la data de postgres:

```bash
docker compose down -v
```

### En local

Backend (Python 3.12) desde la raíz del repo:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Las dependencias están fijadas en `requirements.txt`. El backend se levanta
desde `multimodal-db/`:

```bash
cd multimodal-db
PYTHONPATH=. uvicorn api.main:app --reload --port 8000
```

Frontend en otra terminal:

```bash
cd multimodal-db/frontend
npm install
npm run dev
```

El frontend lee la URL del backend de `NEXT_PUBLIC_API_URL` y por defecto
apunta a `http://localhost:8000`.

## Población de data

El seed `multimodal-db/tests/seed_demo.py` crea una tabla `media`, sube los
archivos al endpoint `/upload` e inserta las filas. Necesita el backend
corriendo y usa `API_URL` (por defecto `http://localhost:8000`).

Tiene tres fuentes de imágenes, en orden de prioridad:

Data sintética (por defecto, sin configurar nada). Sube un PNG de 1x1 y un WAV
corto para probar la galería y el audio player:

```bash
cd multimodal-db
PYTHONPATH=. python tests/seed_demo.py
```

Data del Drive. Baja la carpeta pública del proyecto con `gdown`:

```bash
cd multimodal-db
USE_DRIVE=1 PYTHONPATH=. python tests/seed_demo.py
```

- `DRIVE_FOLDER_ID`: id de la carpeta de Drive (trae uno por defecto).
- `DRIVE_CACHE_DIR`: dónde quedan las imágenes bajadas.

Carpeta local. Si ya tienes imágenes en disco, apúntala con `SEED_IMAGES_DIR`
y tiene prioridad sobre el Drive:

```bash
cd multimodal-db
SEED_IMAGES_DIR=/ruta/a/imagenes PYTHONPATH=. python tests/seed_demo.py
```

## Pruebas

Los tests del query processor corren con pytest contra los mocks:

```bash
cd multimodal-db
pytest tests/
```
